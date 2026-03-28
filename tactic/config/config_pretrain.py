from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from omegaconf import DictConfig, OmegaConf

from tactic.config.config_data import ConfigData
from tactic.config.config_optim import ConfigOptim
from tactic.config.config_preprocessing import ConfigPreprocessing
from tactic.config.config_save_load_mixin import ConfigSaveLoadMixin
from tactic.core.enums import GeneratorName


@dataclass
class ConfigPretrain(ConfigSaveLoadMixin):
    output_dir: Path
    seed: int
    devices: list[torch.device]
    use_ddp: bool
    workers_per_gpu: int
    model: dict
    data: ConfigData
    optim: ConfigOptim
    preprocessing: ConfigPreprocessing

    device: Optional[torch.device] = None   # initialized later by ddp
    is_main_process: bool = True            # initialized later by ddp


    @classmethod
    def from_hydra(cls, cfg_hydra: DictConfig):

        output_dir = Path(cfg_hydra.output_dir)

        devices = [torch.device(device) for device in cfg_hydra.devices]
        model_settings = cfg_hydra.pretrain_model


        return cls(
            output_dir=output_dir,
            devices=devices,
            use_ddp=len(devices) > 1,
            seed=cfg_hydra.seed,
            workers_per_gpu=cfg_hydra.workers_per_gpu,
            model = OmegaConf.to_container(model_settings),
            data = ConfigData(
                use_gmm_prior=cfg_hydra.data.use_gmm_prior,
                gmm_prob_thr=cfg_hydra.data.gmm_prob_thr,
                anoms_in_support=cfg_hydra.data.anoms_in_support,
                generator=GeneratorName(cfg_hydra.data.generator),
                min_samples_support=cfg_hydra.data.min_samples_support,
                max_samples_support=cfg_hydra.data.max_samples_support,
                n_samples_query=cfg_hydra.data.n_samples_query,
                min_features=cfg_hydra.data.min_features,
                max_features=cfg_hydra.data.max_features,
                max_classes=cfg_hydra.data.max_classes,
                generator_hyperparams=OmegaConf.to_container(cfg_hydra.data.generator_hyperparams),
                query_balance_factor=cfg_hydra.data.query_balance_factor
            ),
            optim = ConfigOptim(
                max_steps=cfg_hydra.optim.max_steps,
                log_every_n_steps=cfg_hydra.optim.log_every_n_steps,
                eval_every_n_steps=cfg_hydra.optim.eval_every_n_steps,
                batch_size=cfg_hydra.optim.batch_size,
                gradient_accumulation_steps=cfg_hydra.optim.gradient_accumulation_steps,
                lr=cfg_hydra.optim.lr,
                weight_decay=cfg_hydra.optim.weight_decay,
                beta1=cfg_hydra.optim.beta1,
                beta2=cfg_hydra.optim.beta2,
                warmup_steps=cfg_hydra.optim.warmup_steps,
                cosine_scheduler=cfg_hydra.optim.cosine_scheduler,
                max_grad_norm=cfg_hydra.optim.max_grad_norm,
                use_pretrained_weights=cfg_hydra.optim.use_pretrained_weights,
                path_to_weights=cfg_hydra.optim.path_to_weights,
                loss_anom_weight=cfg_hydra.optim.loss_anom_weight
            ),
            preprocessing = ConfigPreprocessing(
                use_quantile_transformer=cfg_hydra.preprocessing.use_quantile_transformer,
                use_feature_count_scaling=cfg_hydra.preprocessing.use_feature_count_scaling,
            ),
        )
    











