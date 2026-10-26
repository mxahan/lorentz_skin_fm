"""Sarkar's combinatorial construction."""
import numpy as np
import scipy
import networkx as nx

MIN_NORM = 1e-8


# ################# CIRCLE INVERSIONS ########################

def reflect_at_zero(x, mu):  # Note: this differs from geom.poincare.reflect_at_zero because it's numpy instead of torch
    """
    Image of x by circle inversion that takes mu to the origin
    """
    mu_sqnorm = np.sum(mu ** 2)
    a = mu / mu_sqnorm.clip(min=1e-8)
    a_sqnorm = np.sum(a ** 2)
    r2 = a_sqnorm - np.longdouble([1.])
    xa_sqnorm = np.maximum(np.sum((x - a) ** 2, axis=-1, keepdims=True), MIN_NORM)
    return (r2 / xa_sqnorm) * (x - a) + a


def reflect_through_zero(p, q, x):
    """ Image of x under reflection that takes p (normalized) to q (normalized) and 0 to 0. """

    p_ = p / np.linalg.norm(p, axis=-1, keepdims=True).clip(min=1e-15)
    q_ = q / np.linalg.norm(q, axis=-1, keepdims=True).clip(min=1e-15)
    # print("norm p, q", np.linalg.norm(p_), np.linalg.norm(q_))
    r = q_ - p_
    # Magnitude of x in direction of r
    m = np.sum(r * x, axis=-1, keepdims=True) / np.sum(r * r, axis=-1, keepdims=True)
    return x - 2 * r * m


def test_reflect():
    pass

## ################# select distant samples #####################

def points_sample(N = 100, dim = 512):
    # Uniform
    # x = np.random.rand(N, dim) 
    # Half-normal
    # x = np.abs(np.random.randn(N, dim)) # provide +ve and -ve values
    # Exponential
    # x = np.random.exponential(size=(N, dim))
    shape = 0.2  # try 0.5, 1, 2, 5
    x = np.random.gamma(shape=shape, scale=1.0, size=(N, dim))
    
    x = x / np.linalg.norm(x, axis=1, keepdims=True)       # unit sphere
    return x

    
def select_separated(x, N1, n_starts= 3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)       # unit sphere
    best, best_score = None, -np.inf

    for start in np.random.choice(len(x), min(n_starts, len(x)), replace=False):
        selected = [start]                                 # different initial points

        for _ in range(N1 - 1):
            sim = x @ x[selected].T                        # cosine similarity
            score = sim.max(axis=1)                        # closest selected point
            score[selected] = np.inf                       # exclude selected
            selected.append(np.argmin(score))              # farthest candidate

        S = x[selected]                                    # selected subset
        sim = S @ S.T
        np.fill_diagonal(sim, -np.inf)
        min_angle = np.arccos(np.clip(sim.max(), -1, 1))   # worst pair separation

        if min_angle > best_score:
            best, best_score = selected, min_angle

    return x[best]

# ################# SARKAR CONSTRUCTION ########################

def pick_root(tree):
    graph_distances = np.array(nx.floyd_warshall_numpy(tree).astype(np.float32))
    j_ids = np.argmax(graph_distances, axis=1)
    i = np.argmax(graph_distances[np.arange(tree.number_of_nodes()), j_ids])
    j = j_ids[i]
    path = nx.shortest_path(tree, i, j)
    length = len(path)
    root = path[length // 2]
    return root


def place_children(z_parent, n_children, scaling, dim=2, coding=True):
    """Embeds children of node embedded at the origin.
    Assumes z is embedding of parent of node at the origin.
    children are at disrance scale/2 from their parent in hyperbolic metric.
    """
    if dim == 2:
        if z_parent is None:
            theta_parent = 0
            n_neighbors = n_children
        else:
            theta_parent = np.angle(z_parent[0] + z_parent[1] * 1j)
            n_neighbors = n_children + 1
        theta_children = [theta_parent + 2 * np.longdouble(np.pi) * (i + 1) / np.longdouble(n_neighbors) for i in
                          range(n_children)]
        z_children = []
        for theta_child in theta_children:
            z_children.append(scaling * np.array([np.cos(theta_child), np.sin(theta_child)]))
        return z_children
    else:
        normalize = lambda x: x / np.linalg.norm(x, keepdims=True)
        if coding:
            N = 2 ** int(np.ceil(np.log(dim) / np.log(2)))
            H = scipy.linalg.hadamard(N)
            if z_parent is not None:
                par_ = np.concatenate((z_parent, np.zeros(N - dim)))
                H = reflect_through_zero(H[0, :], par_, H)
                # print("reflecting H0 onto parent", scaling * normalize(H[0,:]) - par_)
            z_children = [H[i, :dim] for i in range(1, min(n_children + 1, N))]
            if n_children > N - 1:
                z_children += [np.random.randn(dim) for _ in range(n_children - N + 1)]
            z_children = [scaling * normalize(c) for c in z_children]
        else:
            #   # negative values (comment one of the block)
            # z_children = [scaling * normalize(np.random.randn(dim)) for _ in range(n_children)] # earlier version, provide both +- values

            #   # only positive values
            z_children = points_sample(N = 4* n_children, dim =  dim)
            z_children = select_separated(z_children, N1 = n_children)
            z_children = [scaling * normalize(c) for c in z_children]

        return z_children


def sarkar(tree, tau=1, root=14, dim=2, coding=False, seed=1234):
    """Embeds a tree in H_d using Sarkar's construction.

    Args:
        tree: nx.Graph object representing the tree structure to embed.
        root: index of the root node in the tree object.
        tau: scale of hyperbolic embeddings, parent-child will be placed at
             hyperbolic distance tau from each other.
    """
    np.random.seed(seed)
    if root is None:
        # pick root in Sarkar as node on longest path
        root = pick_root(tree)

    # Initialize embeddings array
    z = np.zeros((tree.number_of_nodes(), dim), dtype=np.float64)
    scaling = np.tanh(tau / 2)  # Euclidean distance corresponding to hyperbolic distance of tau
    # bfs traversal
    bfs_tree_rev = nx.reverse_view(nx.bfs_tree(tree, root))
    for current, children in nx.bfs_successors(tree, root):
        if current == root:
            z[root] = np.zeros(dim)
            z_children = place_children(None, len(children), scaling, dim=dim, coding=coding)
            for i, child_idx in enumerate(children):
                z[child_idx] = z_children[i]
        else:
            z_current = z[current]
            z_parent = z[list(bfs_tree_rev.neighbors(current))[0]]

            # inversion that maps current to the origin
            z_parent = reflect_at_zero(z_parent, z_current)
            z_children = place_children(z_parent, len(children), scaling, dim=dim, coding=coding)
            for i, child_idx in enumerate(children):
                z[child_idx] = reflect_at_zero(z_children[i], z_current)
    return z


def gen_sarkar(tree, tau=1, root=14, dim=16, coding=False, seed=1234, stage_scale=None):
    """Sarkar embedding with edge length at depth d: tau * stage_scale[d]."""
    np.random.seed(seed)
    root = pick_root(tree) if root is None else root
    z = np.zeros((tree.number_of_nodes(), dim), dtype=np.float64)
    stage_scale = ([1.0] * (tree.number_of_nodes() + 1) if stage_scale is None
                   else np.asarray(stage_scale, dtype=np.float64))

    depth = nx.single_source_shortest_path_length(tree, root)
    bfs_rev = nx.reverse_view(nx.bfs_tree(tree, root))

    for current, children in nx.bfs_successors(tree, root):
        child_depth = depth[current] + 1
        if child_depth >= len(stage_scale):
            raise ValueError(f"No stage scale supplied for depth {child_depth}.")
            
        scaling = np.tanh(tau * stage_scale[child_depth] / 2)

        if current == root:
            z_children = place_children(None, len(children), scaling, dim=dim, coding=coding)
            for i, child in enumerate(children): z[child] = z_children[i]
        else:
            z_current = z[current]
            z_parent = z[list(bfs_rev.neighbors(current))[0]]
            z_parent = reflect_at_zero(z_parent, z_current)
            z_children = place_children(z_parent, len(children), scaling, dim=dim, coding=coding)
            for i, child in enumerate(children):
                z[child] = reflect_at_zero(z_children[i], z_current)
    return z