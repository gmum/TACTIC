# import numpy as np
import torch
from loguru import logger
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler


class Preprocessor(TransformerMixin, BaseEstimator):
    """
    This class is used to preprocess the data before it is pushed through the model.
    The preprocessor assures that the data has the right shape and is normalized,
    This way the model always gets the same input distribution, 
    no matter whether the input data is synthetic or real.

    """

    def __init__(
            self, 
            max_features: int,
            use_quantile_transformer: bool,
            use_feature_count_scaling: bool,
        ):

        self.max_features = max_features
        self.use_quantile_transformer = use_quantile_transformer
        self.use_feature_count_scaling = use_feature_count_scaling

    def fit(self, X: torch.Tensor):

        self.compute_pre_nan_mean(X)
        X = self.impute_nan_features_with_mean(X)

        self.determine_which_features_are_singular(X)
        X = self.cutoff_singular_features(X, self.singular_features)

        self.x_min = X.min(dim=0).values
        self.x_max = X.max(dim=0).values

        assert torch.isnan(X).sum() == 0, "There are NaNs in the data after preprocessing"

        return self

    def transform(self, X: torch.Tensor):

        X = self.cutoff_singular_features(X, self.singular_features)
        X = self.impute_nan_features_with_mean(X)

        a, b = -1, 1
        X = a + (X - self.x_min) * (b - a) / (self.x_max - self.x_min)
        X = self.extend_feature_dim_to_max_features(X, self.max_features)

        assert torch.isnan(X).sum() == 0, "There are NaNs in the data after preprocessing"

        return X

    def determine_which_features_are_singular(self, x: torch.Tensor) -> None:
        self.singular_features = torch.tensor([len(torch.unique(x_col)) for x_col in x.T]) == 1


    def compute_pre_nan_mean(self, x: torch.Tensor) -> None:
        """
        Computes the mean of the data before the NaNs are imputed
        """
        self.pre_nan_mean = torch.nanmean(x, dim=0)

    def impute_nan_features_with_mean(self, x: torch.Tensor) -> torch.Tensor:

        inds = torch.where(torch.isnan(x))
        x[inds] = torch.take(self.pre_nan_mean, inds[1])
        return x


    def cutoff_singular_features(self, x: torch.Tensor, singular_features: torch.Tensor) -> torch.Tensor:
        if singular_features.any():
            x = x[:, ~singular_features]

        return x

    def extend_feature_dim_to_max_features(self, x: torch.Tensor, max_features) -> torch.Tensor:
        """
        Increases the number of features to the number of features the model has been trained on
        """
        added_zeros = torch.zeros((x.shape[0], max_features - x.shape[1]), dtype=torch.float32)
        x = torch.concatenate([x, added_zeros], dim=1)
        return x
    


    
