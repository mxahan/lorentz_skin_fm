# code for pca plot
import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.decomposition import PCA
def plot_pca_clusters(x: torch.Tensor, labels, name = None):
    """
    Args:
        x:      (n, d) — feature vectors
        labels: one of:
                  - (n, c) tensor  — one-hot encoded
                  - (n,)   tensor  — integer class indices
                  - list of str    — class name per point e.g. ["cat", "dog", ...]
    """
    assert x.shape[0] == len(labels), "x and labels must have same number of points"
    # Normalize all label formats → list of strings
    if isinstance(labels, torch.Tensor):
        if labels.dim() == 2:                          # one-hot
            class_names = labels.argmax(dim=1).cpu().tolist()
        else:                                          # integer indices
            class_names = labels.cpu().tolist()
        class_names = [str(c) for c in class_names]
    elif isinstance(labels, list):                     # string names
        class_names = [str(c) for c in labels]
    else:
        raise ValueError(f"Unsupported labels type: {type(labels)}")
    unique_classes = sorted(set(class_names), key=lambda c: (int(c) if c.isdigit() else c))
    # unique_classes= ['0', '2', '3', '8', '6', '7', '9']
    # PCA to 2D
    pca = PCA(n_components=2)
    x_2d = pca.fit_transform(x.detach().cpu().numpy())  # (n, 2)
    # Plot
    # fig, ax = plt.subplots(figsize=(8, 6))
    # for cls in unique_classes:
    #     mask = [i for i, c in enumerate(class_names) if c == cls]
    #     ax.scatter(x_2d[mask, 0], x_2d[mask, 1], label=cls, alpha=0.6, s=20)
    # ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    # ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    # ax.set_title("PCA projection")
    # ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
    # plt.tight_layout()
    # plt.show()
    fig, ax = plt.subplots(figsize=(8, 6))

    for cls in unique_classes:
        mask = np.array([c == cls for c in class_names])
    
        ax.scatter(
            x_2d[mask, 0],
            x_2d[mask, 1],
            label=cls,
            alpha=0.7,
            s=28,
            edgecolors="white",
            linewidths=0.3,
        )
    
    # Axis labels
    ax.set_xlabel(
        f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)",
        fontsize=11
    )
    ax.set_ylabel(
        f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)",
        fontsize=11
    )
    
    ax.set_title("PCA Projection", fontsize=20, pad=10)
    
    # Cleaner legend inside the plot
    legend = ax.legend(
        title="Lesion Class",
        loc="upper right",
        fontsize=15,
        title_fontsize=9,
        frameon=True,
        framealpha=0.9,
        borderpad=0.7,
        labelspacing=0.5,
        handletextpad=0.4,
        markerscale=1.2,
    )
    legend.get_frame().set_linewidth(0.8)
    # Subtle grid
    ax.grid(
        True,
        linestyle="--",
        linewidth=0.5,
        alpha=0.25)
    ax.tick_params(axis="both", labelsize=15)
    plt.tight_layout()
    # Save publication-quality vector figure
    if name is not None:
        plt.savefig(
            name,
            format="svg",
            bbox_inches="tight",
        )
    
    plt.show()