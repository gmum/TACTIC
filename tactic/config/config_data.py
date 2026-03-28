from dataclasses import dataclass

from tactic.core.enums import GeneratorName


@dataclass
class ConfigData:
    use_gmm_prior: bool
    gmm_prob_thr: int
    anoms_in_support: bool
    generator: GeneratorName
    min_samples_support: int
    max_samples_support: int
    n_samples_query: int
    min_features: int
    max_features: int
    max_classes: int
    generator_hyperparams: dict
    query_balance_factor: float
