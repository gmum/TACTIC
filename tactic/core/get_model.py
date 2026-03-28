import torch

from tactic.config.config_pretrain import ConfigPretrain
from tactic.model.tactic import TACTIC


def get_model_pretrain(cfg: ConfigPretrain) -> torch.nn.Module:
    return TACTIC(
        n_features=cfg.data.max_features,
        dim=cfg.model['dim'],
        n_layers=cfg.model['n_layers'],
        n_heads=cfg.model['n_heads'],
        attn_dropout=cfg.model['attn_dropout'],
        use_pretrained_weights=cfg.optim.use_pretrained_weights,
        path_to_weights=cfg.optim.path_to_weights
    )
