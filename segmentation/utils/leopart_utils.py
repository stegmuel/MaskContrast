import faiss
import torch
from sklearn.preprocessing import StandardScaler
import numpy as np


def normalize_and_transform(feats: torch.Tensor, pca_dim: int) -> torch.Tensor:
    feats = feats.numpy()
    # Iteratively train scaler to normalize data
    bs = 100000
    num_its = (feats.shape[0] // bs) + 1
    scaler = StandardScaler()
    for i in range(num_its):
        scaler.partial_fit(feats[i * bs:(i + 1) * bs])
    print("trained scaler")
    for i in range(num_its):
        feats[i * bs:(i + 1) * bs] = scaler.transform(feats[i * bs:(i + 1) * bs])
    print(f"normalized feats to {feats.shape}")
    # Do PCA
    pca = faiss.PCAMatrix(feats.shape[-1], pca_dim)
    pca.train(feats)
    assert pca.is_trained
    transformed_val = pca.apply_py(feats)
    print(f"val feats transformed to {transformed_val.shape}")
    return torch.from_numpy(transformed_val)


def cluster(pca_dim: int, transformed_feats: np.ndarray, spatial_res: int, k: int, seed: int = 1,
            mask: torch.Tensor = None, spherical: bool = False):
    """
    Computes k-Means and retrieve assignments for each feature vector. Optionally the clusters are only computed on
    foreground vectors if a mask is provided. In this case transformed_feats is already expected to contain only the
    foreground vectors.
    """
    print(f"start clustering with {seed}")
    kmeans = faiss.Kmeans(pca_dim, k, niter=100, nredo=5, verbose=True, gpu=False, spherical=spherical, seed=seed)
    kmeans.train(transformed_feats)
    print("kmeans trained")
    _, pred_labels = kmeans.index.search(transformed_feats, 1)
    clusters = pred_labels.squeeze()
    print("index search done")

    # Apply fg mask if provided.
    if mask is not None:
        preds = torch.zeros_like(mask) + k
        preds[mask.bool()] = torch.from_numpy(clusters).float()
    else:
        preds = torch.from_numpy(clusters.reshape(-1, spatial_res, spatial_res))
    return preds