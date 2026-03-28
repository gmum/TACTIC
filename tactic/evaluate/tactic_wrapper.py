import numpy as np
import torch

from tactic.model.tactic import TACTIC


class TACTICWrapper:
    def __init__(self, model: TACTIC, device="cpu", max_dim=100, **kwargs):
        self.model = model
        self.device = device
        self.max_dim = max_dim
        self.bag_size = 10

    def fit(self, X_train, y_train):
        if X_train.shape[1] < self.model.n_features:
            padding_width = self.model.n_features - X_train.shape[1]
            X_train = np.pad(X_train, ((0, 0), (0, padding_width)), mode='constant')
        self.X_train = torch.from_numpy(X_train).float().to(self.device)

        return self

    def predict_score(self, X):
        if X.shape[1] < self.model.n_features:
            padding_width = self.model.n_features - X.shape[1]
            X = np.pad(X, ((0, 0), (0, padding_width)), mode='constant')
        X = torch.from_numpy(X).float().to(self.device)

        scores = []
        for _ in range(self.bag_size):
            data_shape = X.shape[1]
            if data_shape > self.model.n_features:
                chosen_indices = np.random.choice(self.X_train.shape[1], self.model.n_features, replace=False)
                X_train_cur = self.X_train[:, chosen_indices]
                X_cur = X[:, chosen_indices]
            else:
                X_train_cur = self.X_train
                X_cur = X

            proba = self.model.predict_proba(X_train_cur, X_cur).cpu().numpy()
            score = proba[:, 1]
            scores.append(score)

        score = np.mean(scores, axis=0)
        preds = score >= 0.5

        return score, preds
