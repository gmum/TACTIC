from typing import Generator, Iterator
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from tactic.config.config_pretrain import ConfigPretrain
from tactic.core.enums import GeneratorName
from tactic.data.preprocessor import Preprocessor

from tactic.data.synthetic_generator_selector import SyntheticDatasetGeneratorSelectorMixin
from tactic.data.synthetic_gmm import create_gaussian_mixture


class SyntheticDataset(torch.utils.data.IterableDataset, SyntheticDatasetGeneratorSelectorMixin):

    def __init__(
        self, 
        cfg: ConfigPretrain,
        generator_name: GeneratorName,
        min_samples_support: int,
        max_samples_support: int,
        n_samples_query: int,
        min_features: int,
        max_features: int,
        max_classes: int,
        use_quantile_transformer: bool,
        use_feature_count_scaling: bool,
        generator_hyperparams: dict
    ) -> None:
        
        self.cfg = cfg
        self.generator_name = generator_name
        self.min_samples_support = min_samples_support
        self.max_samples_support = max_samples_support
        self.n_samples_query = n_samples_query
        self.n_samples = max_samples_support + n_samples_query
        self.min_features = min_features
        self.max_features = max_features
        self.max_classes = max_classes
        self.use_quantile_transformer = use_quantile_transformer
        self.use_feature_count_scaling = use_feature_count_scaling
        self.generator_hyperparams = generator_hyperparams


    def __iter__(self) -> Iterator:

        self.synthetic_dataset_generator = self.select_synthetic_dataset_generator()
        return self.generator()


    def generator(self) -> Generator[dict[str, torch.Tensor], None, None]:
        i = 0
        error_counter = 0
        query_balance = self.cfg.data.query_balance_factor

        while True:
            try:
                i += 1

                is_other = np.random.uniform(0, 1) <= self.cfg.data.gmm_prob_thr
                gauss_data_type = np.random.randint(3)

                if self.cfg.data.use_gmm_prior and is_other:
                    if gauss_data_type == 0:
                        alpha_mean, alpha_cov, is_global = 1, 5, False
                    elif gauss_data_type == 1:
                        alpha_mean, alpha_cov, is_global = 5, 1, False
                    else: # gauss_data_type == 2
                        alpha_mean, alpha_cov, is_global = 1, 1, True

                    x, y = create_gaussian_mixture(
                        num_gaussians=20, cluster_min_points=50, cluster_max_points=500,
                        dim=self.cfg.data.max_features, n_support_min=2000, n_support_max=9000,
                        n_query=500, alpha_mean=alpha_mean, alpha_cov=alpha_cov, is_global=is_global,
                    )
                else:
                    x, y = next(self.synthetic_dataset_generator)

                if not (self.cfg.data.use_gmm_prior and is_other):
                    y = self.randomize_class_order(y)

                    unique_labels = np.unique(y)
                    unique_labels = unique_labels if unique_labels[0] != -100 else unique_labels[1:]
                    num_unique = unique_labels.shape[0]
                    if num_unique == 1:
                        continue

                    k_min = max(1, num_unique // 10)
                    k_max = num_unique + 1 if num_unique == 1 else num_unique
                    k = np.random.randint(k_min, k_max, (1,))[0]
                    perm = np.random.permutation(num_unique)
                    selected_labels = unique_labels[perm[:k]]

                    samples_mask = np.isin(y, selected_labels)
                    anomalies_mask = ~samples_mask & (y != -100)
                    y[samples_mask] = 0
                    y[anomalies_mask] = 1

                try:
                    x_support, y_support, x_query, y_query = self.split_into_support_and_query(x, y, query_balance)
                except BaseException:
                    continue

                x_support = torch.tensor(x_support, dtype=torch.float32)
                y_support = torch.tensor(y_support, dtype=torch.int64)
                x_query = torch.tensor(x_query, dtype=torch.float32)
                y_query = torch.tensor(y_query, dtype=torch.int64)

                if not self.cfg.data.anoms_in_support:
                    y_support[y_support == 1] = -100


                preprocessor = Preprocessor(
                    max_features=self.max_features,
                    use_quantile_transformer=False,
                    use_feature_count_scaling=False,
                )
                preprocessor.fit(x_support)
                x_support = preprocessor.transform(x_support)
                x_query = preprocessor.transform(x_query)

            except BaseException as e:
                error_counter += 1
                print("Error: ", error_counter, e)
                if error_counter > 10000:
                    raise e
                continue

            yield {
                'x_support': x_support,
                'y_support': y_support,
                'x_query': x_query,
                'y_query': y_query,
            }


    def randomize_class_order(self, y):
            
        curr_classes = int(y.max().item()) + 1
        new_classes = np.random.permutation(self.max_classes)
        mapping = { i: new_classes[i] for i in range(curr_classes) }
        y = np.array([mapping[i.item()] for i in y], dtype=np.int64)

        return y


    def split_into_support_and_query(self, x: np.ndarray, y: np.ndarray, query_balance: float
                                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        normal_len = len(np.where(y == 0)[0])
        anom_len = len(np.where(y == 1)[0])
        batch_size = 256

        size = max(
            batch_size * (1-query_balance) / normal_len,
            batch_size * query_balance / anom_len
        )

        x_support, x_query, y_support, y_query = train_test_split(
            x, y,
            test_size=size,
            stratify=y
        )
        normal_idx = np.where(y_query == 0)[0]
        anom_idx = np.where(y_query == 1)[0]

        normal_len = len(normal_idx)
        anom_len = len(anom_idx)

        balance_factor = (1 - query_balance) / query_balance
        anom_balanced = int(balance_factor * anom_len)
        norm_balanced = int(normal_len / balance_factor)
        if anom_balanced > normal_len:
            extra = np.random.choice(anom_idx, size=norm_balanced, replace=False)
            indices = np.concatenate([normal_idx, extra])
        else:
            extra = np.random.choice(normal_idx, size=anom_balanced, replace=False)
            indices = np.concatenate([extra, anom_idx])
        x_query = x_query[indices]
        y_query = y_query[indices]

        normal_idx = np.where(y_support == 0)[0]
        anom_idx = np.where(y_support == 1)[0]

        normal_len = len(normal_idx)
        anom_len = len(anom_idx)

        size = np.random.uniform(0.05, 0.3)
        expected_anoms = int(normal_len*size)

        if anom_len > expected_anoms:
            shuffled = np.random.permutation(anom_idx)
            anom_idx = shuffled[:expected_anoms]

        indices = np.concatenate([normal_idx, anom_idx])
        x_support = x_support[indices]
        y_support = y_support[indices]

        n_samples_support = np.random.randint(low=self.min_samples_support, high=self.max_samples_support)
        train_size = n_samples_support / len(y_support)

        if 0.0 < train_size < 1.0:
            strat = y_support if len(anom_idx) > 1 else None

            x_support, _, y_support, _ = train_test_split(
                x_support, y_support,
                test_size=1-train_size,
                stratify=strat
            )

        return x_support, y_support, x_query, y_query
    

    def randomize_feature_order(self, x_support: np.ndarray, x_query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

        curr_features = x_support.shape[1]
        new_feature_order = torch.randperm(curr_features)

        x_support = x_support[:, new_feature_order]
        x_query = x_query[:, new_feature_order]

        return x_support, x_query
    
