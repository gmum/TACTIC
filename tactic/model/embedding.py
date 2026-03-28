import einops
import torch
import torch.nn as nn


class FoundationEmbeddingX(torch.nn.Module):

    def __init__(
            self,
            dim: int,
            n_features: int,
        ) -> None:
        
        super().__init__()

        self.dim = dim
        self.n_features = n_features

        self.x_embedding = nn.Linear(n_features, dim)

    
    def forward(self, x_support: torch.Tensor, x_query__: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        batch_size = x_support.shape[0]
        n_obs_support = x_support.shape[1]
        n_obs_query__ = x_query__.shape[1]

        x_support = self.x_embedding(x_support)
        x_query__ = self.x_embedding(x_query__)

        return x_support, x_query__
