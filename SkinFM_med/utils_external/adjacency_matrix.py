# mining on manifold: https://arxiv.org/pdf/1803.11095
import torch.nn.functional as F
import torch

def euclidean_adj(features, dist_ = 0, tau= 0.1): 
    '''
    return euclidean adjacency matrix 
    input shape:  N x d (batch_size x feature_dimension)
    dist_ : 0  - dot product, 1: cosine similiarity, 2: RBF similarity
    
    '''
    
    c = features.shape[-1]
    
    flat_features = features.view(-1, c)  # (64*64, 256)
    


    if dist_ ==0:
        # Compute dot product similarity → shape (4096, 4096)
        similarity = flat_features @ flat_features.t()  # (4096, 4096)
        # Apply ReLU to ensure similarity = max(dot_product, 0)
        adjacency = torch.relu(similarity)  # (4096, 4096)
        
    if dist_ ==1:
        # Compute dot product similarity → shape (4096, 4096)
        flat_features = F.normalize(flat_features, p=2, dim=1)
        similarity = flat_features @ flat_features.t()  # (4096, 4096)
        # Step 3: Apply ReLU to ensure similarity = max(dot_product, 0)
        adjacency = torch.relu(similarity)  # (4096, 4096)
    
    
    if dist_ == 2:
        # Compute pairwise squared L2 distances
        feat_sq = (flat_features**2).sum(dim=1, keepdim=True)  # (N, 1)
        dists = feat_sq + feat_sq.t() - 2 * (flat_features @ flat_features.t())  # (N, N)
        # RBF kernel similarity
        adjacency = torch.exp(-dists / tau)  # (4096, 4096)
    

    
    adjacency.fill_diagonal_(0)
    
    return adjacency
    
H, W = 64, 64

def get_topk_neighbors(adjacency, point, k=10):
    '''
    input:
        adjacency: NxN adjacency matrix (euclidean)
        point: points to find the neighbour around
    return:
        Neighbor indices of top k
    '''
    if isinstance(point, tuple):
        h, w = point
        idx = h * W + w
    else:
        idx = point

    # Get the similarity scores for the point
    sim_scores = adjacency[idx]  # shape: (4096,)

    # Find top-k indices (excluding self if needed)
    topk = torch.topk(sim_scores, k=k, largest=True)
    neighbor_indices = topk.indices  # shape: (k,)

    return neighbor_indices  # each index ∈ [0, 4095]


def manifold_dist_adj(adjacency,  alpha = 0.99):
    '''
    calculate: (1−α)⋅(I−α⋅D^-.5 A D^-.5 )^−1
    input: 
        adjacency: NxN adjacency matrix (euclidean)
    
    return:
        Graph Similarity matrix: after random walk
    '''
    degree = adjacency.sum(dim=1)  # shape: (N,)
    # Compute D^{-1/2} safely (add epsilon to avoid division by 0)
    eps = 1e-10
    D_inv_sqrt = torch.diag(torch.pow(degree + eps, -0.5))  # shape: (N, N)

    # Symmetric normalization
    A_sym = D_inv_sqrt @ adjacency @ D_inv_sqrt
    
    I = torch.eye(A_sym.size(0), device=adjacency.device)

    # Compute the inverse
    inv_term = torch.inverse(I - alpha * A_sym)

    # Final result
    result = (1 - alpha) * inv_term
    return result

def neighbor_mask_image(neighbor_indices, H=64, W=64):
    mask = torch.zeros(H * W, dtype=torch.float32)
    mask[neighbor_indices] = 1.0
    return mask.view(H, W)