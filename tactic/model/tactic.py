import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tactic.model.embedding import FoundationEmbeddingX


class TACTIC(nn.Module):

    def __init__(
            self,
            n_features: int,
            dim: int,
            n_layers: int,
            n_heads: int,
            attn_dropout: float,
            use_pretrained_weights: bool,
            path_to_weights: str,
    ) -> None:
        super().__init__()
        n_classes = 2

        self.n_features = n_features
        self.n_classes = n_classes
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.attn_dropout = attn_dropout

        self.x_embedding = FoundationEmbeddingX(dim, n_features)

        self.layers = nn.ModuleList([])

        for _ in range(n_layers):
            att = MultiheadAttention(dim, n_heads)

            self.layers.append(nn.ModuleDict({
                'layer_norm1': nn.LayerNorm(dim),
                'attention': att,
                'layer_norm2': nn.LayerNorm(dim),
                'linear1': nn.Linear(dim, dim * 4),
                'linear2': nn.Linear(dim * 4, dim),
            }))

        self.final_layer1 = nn.Linear(dim, dim * 4)
        self.final_layer2 = nn.Linear(dim * 4, n_classes)

        if use_pretrained_weights:
            self.load_state_dict(torch.load(path_to_weights), strict=False)
        else:
            self.init_weights()

    def init_weights(self):

        for module_dict in self.layers:
            # module_dict['attention'].init_weights()
            nn.init.zeros_(module_dict['linear2'].weight)
            nn.init.zeros_(module_dict['linear2'].bias)

    def forward(self, x_support: torch.Tensor, mask: torch.Tensor, x_query: torch.Tensor):

        """
        x_support is (batch_size, n_observations_support, n_features)
        mask is (batch_size, n_observations_support)

        x_query is (batch_size, n_observations_query, n_features)

        returns:

        y_query is (batch_size, n_observations_query, n_classes)

        syntax:
        b = batch size
        n = number of observations
        d = dimension of embedding
        c = number of classes
        """

        x_query__ = x_query

        batch_size = x_support.shape[0]
        n_obs_support = x_support.shape[1]
        n_obs_query__ = x_query__.shape[1]

        padding_mask = torch.zeros((batch_size, n_obs_support), dtype=torch.bool, device=x_support.device)
        padding_mask[mask == -100] = True

        x_support, x_query__ = self.x_embedding(x_support, x_query__)

        support = x_support
        query__ = x_query__

        x, pack = einops.pack((support, query__), 'b * d')

        for module_dict in self.layers:
            x_residual = x
            support, query__ = einops.unpack(x, pack, 'b * d')
            att_support = module_dict['attention'](support, support, support, key_padding_mask=padding_mask)
            att_query__ = module_dict['attention'](query__, support, support, key_padding_mask=padding_mask)
            x = einops.pack((att_support, att_query__), 'b * d')[0]
            x = x_residual + x
            x = module_dict['layer_norm1'](x)
            x_residual = x
            x = module_dict['linear1'](x)
            x = torch.nn.functional.gelu(x)
            x = module_dict['linear2'](x)
            x = x_residual + x
            x = module_dict['layer_norm2'](x)

        x = self.final_layer1(x)
        x = F.gelu(x)
        x = self.final_layer2(x)

        support, query__ = einops.unpack(x, pack, 'b * c')

        return query__

    def create_loss(
            self, pred, y_query, base_loss_fn,
    ):
        loss_pred = base_loss_fn(pred, y_query)

        return loss_pred

    def train_forward(self, x_support: torch.Tensor, mask: torch.Tensor,
                      x_query: torch.Tensor, y_query: torch.Tensor,
                      base_loss_fn):
        pred = self.forward(x_support, mask, x_query)
        loss = self.create_loss(pred, y_query, base_loss_fn)

        return pred, loss

    def predict_proba(self, X_train, X_test):
        X_train = X_train.unsqueeze(0)
        X_test = X_test.unsqueeze(0)
        mask = torch.zeros(*X_train.shape[:2])

        with torch.no_grad():
            y_logits = self.forward(X_train, mask, X_test)
        y_probs = torch.nn.functional.softmax(y_logits, dim=-1)

        return y_probs.squeeze(0)

    def predict_score(self, X_train, X_test):
        proba = self.predict_proba(X_train, X_test).cpu().numpy()
        score = proba[:, 1]
        preds = np.argmax(proba, axis=-1)

        return score, preds

class MultiheadAttention(torch.nn.Module):

    def __init__(self, dim: int, n_heads: int) -> None:
        super().__init__()

        self.use_flash_attention = False
        self.dim = dim
        self.n_heads = n_heads

        self.att = nn.MultiheadAttention(dim, n_heads, dropout=0.0, batch_first=True)

    def init_weights(self):
        pass
        # nn.init.zeros_(self.att.out_proj.weight)
        # nn.init.zeros_(self.att.out_proj.bias)

    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            key_padding_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        b = batch size
        n = number of samples (dataset size)
        h = heads
        d = dimension of embedding

        query is (b, n, d)
        key is (b, n, d)
        value is (b, n, d)

        attention weights will be (b, h, n, n)
        output will be (b, n, d)
        """

        output = self.att(query, key, value, key_padding_mask=key_padding_mask)[0]
        return output


class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x
