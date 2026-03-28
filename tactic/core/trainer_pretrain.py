from pathlib import Path
import torch
import torch.multiprocessing as mp
import wandb
from loguru import logger
from sklearn.base import BaseEstimator

from tactic.evaluate.tactic_wrapper import TACTICWrapper
from tactic.evaluate.run import run_evaluate
from tactic.config.config_pretrain import ConfigPretrain
from tactic.core.get_model import get_model_pretrain
from tactic.core.get_optimizer import get_optimizer_pretrain
from tactic.core.get_scheduler import get_scheduler_pretrain
from tactic.core.losses import CrossEntropyLossExtraBatch
from tactic.core.metrics import MetricsTraining
from tactic.core.trainer_pretrain_init import (create_synthetic_dataloader, create_synthetic_dataset,
                                                     log_parameter_count, prepare_ddp_model)

import time


class TrainerPretrain(BaseEstimator):

    def __init__(
            self, 
            cfg: ConfigPretrain,
            barrier: mp.Barrier
        ):

        self.cfg = cfg
        self.barrier = barrier
        self.model_ = get_model_pretrain(cfg)
        self.model_.to(cfg.device)

        log_parameter_count(cfg, self.model_)
        self.model = prepare_ddp_model(cfg, self.model_)

        self.synthetic_dataset = create_synthetic_dataset(cfg)
        self.synthetic_dataloader = create_synthetic_dataloader(cfg, self.synthetic_dataset)

        self.optimizer = get_optimizer_pretrain(cfg, self.model)
        print(self.optimizer)
        self.scheduler = get_scheduler_pretrain(cfg, self.optimizer)
        self.loss = CrossEntropyLossExtraBatch(cfg.optim.loss_anom_weight)
        self.metrics_train = MetricsTraining()
        self.step = 0
        self.dataloader = None
        self.all_data = torch.load("tactic/evaluate/datasets/synthetic_data/test_gmm_datasets.pt")
        self.device = cfg.device
        self.start = time.time()
        

    def train(self):

        self.model.train()
        self.dataloader = iter(self.synthetic_dataloader)

        for step in range(0, self.cfg.optim.max_steps+1):
            self.step = step
            self.train_one_step()

        self.prepare_directories_and_weights()
        self.test_current_model()

    def test(self):
        self.model.eval()
        self.test_current_model()

    
    def train_one_step(self):
        self.process_next_batch()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.optim.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        if self.log_this_step() and self.cfg.is_main_process:
            self.log_training_metrics()


    def process_next_batch(self):
        for _ in range(self.cfg.optim.gradient_accumulation_steps):

            dataset = next(self.dataloader)

            x_support = dataset['x_support']
            y_support = dataset['y_support']
            x_query = dataset['x_query']
            y_query = dataset['y_query']

            x_support = x_support.to(self.cfg.device)
            y_support = y_support.to(self.cfg.device)
            x_query = x_query.to(self.cfg.device)
            y_query = y_query.to(self.cfg.device)

            pred, loss = self.model.train_forward(x_support, y_support, x_query, y_query, self.loss)
            loss = loss / self.cfg.optim.gradient_accumulation_steps
            loss.backward()

            self.update_metrics(pred.detach().cpu(), y_query.detach().cpu())
    

    def test_current_model(self):
        if not self.cfg.is_main_process:
            return
        logger.info(f"Starting test sweep")

        for syntethic_mode in [None, 'cluster', 'global', 'local']:
            wrapper = TACTICWrapper(self.model_, self.device, syn_type=str(syntethic_mode), max_dim=self.cfg.data.max_features)

            if syntethic_mode is None:
                dataset = None
            else:
                dataset = self.all_data[syntethic_mode]

            run_evaluate(
                clf=wrapper, step=self.step, model_name="TACTIC",
                dataset=dataset, seedlist=[0, 1, 2, 3, 4],
                anomaly_stratify=self.cfg.data.anoms_in_support,
                max_dim=self.cfg.data.max_features,
                run_mode=syntethic_mode
            )


    def log_this_step(self):
        return self.step % self.cfg.optim.log_every_n_steps == 0
    

    def log_training_metrics(self):

        logger.info(f"Step {self.step} | Loss: {self.metrics_train.loss:.4f} | Accuracy: {self.metrics_train.accuracy:.4f}")
        wandb.log(
            {
                "train/accuracy": self.metrics_train.accuracy,
                "train/loss": self.metrics_train.loss,
                "train/time_per_step": (time.time() - self.start) / self.cfg.optim.log_every_n_steps,
                "step": self.step
            }
        )
        wandb.log(
            self.metrics_train.per_class_accuracy
        )
        self.metrics_train.reset()
        self.start = time.time()


    def eval_this_step(self):
        return self.step % self.cfg.optim.eval_every_n_steps == 0


    def update_metrics(self, pred: torch.Tensor, y_query: torch.Tensor):
        
        self.metrics_train.update(pred, y_query)
        #self.metrics_val.update(pred, y_query)


    def move_model_to_cpu(self):

        del self.model
        self.model_.cpu()
        torch.cuda.empty_cache()

        # wait until all gpus have moved the model to cpu
        if self.cfg.use_ddp:
            torch.distributed.barrier()
        self.device = "cpu"


    def wait_and_move_model_to_gpu(self):

        # We cannot use the torch distributed barrier here, because that blocks the execution on the gpus.
        # This barrier only blocks execution on the cpu of the current process, which doesn't interfere with the validation sweep.
        self.barrier.wait()

        # see https://github.com/pytorch/pytorch/issues/104336
        self.model_.to(self.cfg.device)

        if self.cfg.use_ddp:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model_, device_ids=[self.cfg.device], find_unused_parameters=False)
        else:
            self.model = self.model_
        self.device = "cuda"


    def prepare_directories_and_weights(self) -> tuple[Path, Path]:

        weights_path = self.cfg.output_dir / 'weights' / f"model_step_{self.step}.pt"
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        self.last_weights_path = weights_path

        output_dir = self.cfg.output_dir / f"step_{self.step}"
        output_dir.mkdir(parents=True, exist_ok=True)

        state_dict = self.model_.state_dict()
        torch.save(state_dict, weights_path)

        return output_dir, weights_path


