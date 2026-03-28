import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler


def create_gaussian_mixture(
        num_gaussians, cluster_min_points, cluster_max_points, dim,
        n_support_min, n_support_max, n_query, alpha_mean=5, alpha_cov=5,
        is_global=False
):
    means, covs = [], []
    points = []

    dataset_gaussians = np.random.randint(2, num_gaussians + 1)

    # Ensure reasonable number of points per cluster
    #cluster_max_points = min(max_points, 2500 // dataset_gaussians + 1)
    cur_dim = np.random.randint(2, dim + 1)

    for gaussian_index in range(dataset_gaussians):
        points_per_gaussian = np.random.randint(cluster_min_points, cluster_max_points + 1)

        random_matrix = np.random.randn(cur_dim, cur_dim)
        covariance_matrix = np.dot(random_matrix, random_matrix.T)
        cov = covariance_matrix

        mean = np.random.uniform(-1, 1, size=(cur_dim, ))

        points.append(points_per_gaussian)

        means.append(mean[np.newaxis, :])
        covs.append(cov[np.newaxis, :])

    points = torch.tensor(points, dtype=torch.float32)
    # More stable probability calculation
    probs = points / torch.sum(points)

    n_support = np.random.randint(n_support_min, n_support_max)
    support_data, query_data = [], []

    for cur_mean, cur_cov, prob in zip(means, covs, probs):
        gauss_support = int(n_support*prob)
        gauss_query = int(n_query*prob)

        x_support = np.random.multivariate_normal(cur_mean[0], cur_cov[0], gauss_support)
        x_query = np.random.multivariate_normal(alpha_mean*cur_mean[0], alpha_cov*cur_cov[0], gauss_query)

        support_data.append(x_support)
        query_data.append(x_query)

    X_support = np.vstack(support_data)
    X_query = np.vstack(query_data)

    min_scaler = MinMaxScaler(feature_range=(-1, 1))
    X_support = min_scaler.fit_transform(X_support)
    X_query = min_scaler.transform(X_query)

    if is_global:
        np.random.uniform(-1.1, 1.1, size=X_query.shape).astype(np.float32)

    y_support = np.zeros(X_support.shape[0], dtype=np.int64)
    y_query = np.ones(X_query.shape[0], dtype=np.int64)
    X = np.concatenate([X_support, X_query], axis=0)
    y = np.concatenate([y_support, y_query], axis=0)

    return X, y
