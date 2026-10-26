import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans


@torch.no_grad()
def spectral_cluster(affinity, n_clusters=2, eps=1e-8):
    """
    affinity: [B, N, N] or [N, N]
    returns: cluster labels [B, N] or [N]
    """
    single = affinity.ndim == 2
    if single:
        affinity = affinity.unsqueeze(0)

    # Symmetric, nonnegative affinity
    A = ((affinity + affinity.transpose(-1, -2)) / 2).clamp_min(0)

    # Normalized affinity: D^{-1/2} A D^{-1/2}
    d = A.sum(dim=-1).clamp_min(eps).rsqrt()
    L = d.unsqueeze(-1) * A * d.unsqueeze(-2)

    # Top-k eigenvectors
    _, eigenvectors = torch.linalg.eigh(L)
    embedding = F.normalize(
        eigenvectors[..., -n_clusters:],
        dim=-1,
        eps=eps
    )

    # K-means for each sample
    labels = torch.stack([
        torch.tensor(
            KMeans(
                n_clusters=n_clusters,
                n_init=10,
                random_state=0
            ).fit_predict(x.cpu().numpy()),
            device=affinity.device
        )
        for x in embedding
    ])

    return labels[0] if single else labels

