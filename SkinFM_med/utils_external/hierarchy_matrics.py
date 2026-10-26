import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence


@dataclass
class Hierarchy:
    leaf_names: List[str]
    lca_heights: torch.Tensor   # [C, C]
    depth_nodes: torch.Tensor   # [C, max_depth + 1]
    max_depth: int


def _parse_tree(tree: Dict[str, Any]):
    """
    Converts a nested tree into parent/children dictionaries.

    Supported structure:
        {"root": {
            "group_1": ["leaf_1", "leaf_2"],
            "group_2": {
                "subgroup": ["leaf_3", "leaf_4"]
            }
        }}
    """
    if len(tree) != 1:
        raise ValueError("TREE must contain exactly one root node.")

    root = next(iter(tree))
    parent = {}
    children = {}

    def add_node(node):
        children.setdefault(node, [])

    def connect(parent_node, child_node):
        add_node(parent_node)
        add_node(child_node)

        if child_node in parent:
            raise ValueError(
                f"Node {child_node!r} has more than one parent."
            )

        parent[child_node] = parent_node
        children[parent_node].append(child_node)

    def visit(node, descendants):
        add_node(node)

        if descendants is None:
            return

        if isinstance(descendants, str):
            connect(node, descendants)
            return

        if isinstance(descendants, list):
            for item in descendants:
                if isinstance(item, str):
                    connect(node, item)
                elif isinstance(item, dict):
                    for child, subtree in item.items():
                        connect(node, child)
                        visit(child, subtree)
                else:
                    raise TypeError(
                        f"Unsupported tree item: {type(item)}"
                    )
            return

        if isinstance(descendants, dict):
            for child, subtree in descendants.items():
                connect(node, child)
                visit(child, subtree)
            return

        raise TypeError(
            f"Unsupported descendants for {node!r}: "
            f"{type(descendants)}"
        )

    visit(root, tree[root])
    return root, parent, children


def build_hierarchy(
    tree: Dict[str, Any],
    nodes: Sequence[str],
    num_leaf_classes: int,
) -> Hierarchy:
    """
    Builds the matrices required by the hierarchical metrics.

    Requirements
    ------------
    - output column j corresponds to NODES[j].
    - The first `num_leaf_classes` entries of NODES are all tree leaves.
    """
    root, parent, children = _parse_tree(tree)

    if len(set(nodes)) != len(nodes):
        raise ValueError("NODES contains duplicate names.")

    tree_nodes = set(children)

    missing = tree_nodes - set(nodes)
    extra = set(nodes) - tree_nodes

    if missing:
        raise ValueError(
            f"Tree nodes missing from NODES: {sorted(missing)}"
        )

    if extra:
        raise ValueError(
            f"NODES entries missing from TREE: {sorted(extra)}"
        )

    tree_leaves = {
        node for node, child_nodes in children.items()
        if len(child_nodes) == 0
    }

    leaf_names = list(nodes[:num_leaf_classes])

    if set(leaf_names) != tree_leaves:
        missing_leaves = tree_leaves - set(leaf_names)
        nonleaves = set(leaf_names) - tree_leaves

        raise ValueError(
            "The first output.shape[1] entries of NODES must be "
            "exactly the tree leaves. "
            f"Missing leaves: {sorted(missing_leaves)}; "
            f"non-leaf entries: {sorted(nonleaves)}."
        )

    # Root-to-leaf paths
    paths = {}

    for leaf in leaf_names:
        path = [leaf]
        current = leaf

        while current != root:
            if current not in parent:
                raise ValueError(
                    f"Node {current!r} is disconnected from root {root!r}."
                )

            current = parent[current]
            path.append(current)

        paths[leaf] = list(reversed(path))

    max_depth = max(len(path) - 1 for path in paths.values())

    # Assign integer IDs for depth-wise comparisons.
    node_to_id = {
        node: index for index, node in enumerate(nodes)
    }

    # Pad short paths by repeating the ended leaf.
    depth_nodes = torch.empty(
        num_leaf_classes,
        max_depth + 1,
        dtype=torch.long,
    )

    for leaf_index, leaf in enumerate(leaf_names):
        path = paths[leaf]
        padded_path = path + [leaf] * (max_depth + 1 - len(path))

        depth_nodes[leaf_index] = torch.tensor(
            [node_to_id[node] for node in padded_path],
            dtype=torch.long,
        )

    # Standard node height:
    # leaf = 0; internal node = 1 + maximum child height.
    node_heights = {}

    def get_height(node):
        if node in node_heights:
            return node_heights[node]

        if not children[node]:
            height = 0
        else:
            height = 1 + max(
                get_height(child) for child in children[node]
            )

        node_heights[node] = height
        return height

    get_height(root)

    # Ancestor sets for LCA lookup
    ancestor_sets = {}

    for leaf, path in paths.items():
        ancestor_sets[leaf] = set(path)

    lca_heights = torch.empty(
        num_leaf_classes,
        num_leaf_classes,
        dtype=torch.float32,
    )

    for i, true_leaf in enumerate(leaf_names):
        true_path = paths[true_leaf]

        for j, predicted_leaf in enumerate(leaf_names):
            common_ancestors = ancestor_sets[predicted_leaf]

            # Search upward from the true leaf; the first shared
            # ancestor is the LCA.
            lca = next(
                node
                for node in reversed(true_path)
                if node in common_ancestors
            )

            lca_heights[i, j] = node_heights[lca]

    return Hierarchy(
        leaf_names=leaf_names,
        lca_heights=lca_heights,
        depth_nodes=depth_nodes,
        max_depth=max_depth,
    )


@torch.no_grad()
def mistake_severity(
    output: torch.Tensor,
    labels: torch.Tensor,
    hierarchy: Hierarchy,
) -> float:
    labels = _validate_metric_inputs(output, labels, hierarchy)
    predictions = output.argmax(dim=1)
    mistakes = predictions.ne(labels)

    if not mistakes.any().item():
        return 0.0

    lca_heights = hierarchy.lca_heights.to(output.device)

    return lca_heights[
        labels[mistakes],
        predictions[mistakes],
    ].float().mean().item()


@torch.no_grad()
def hierarchical_distance_topk(
    output: torch.Tensor,
    labels: torch.Tensor,
    hierarchy: Hierarchy,
    k: int = 10,
) -> float:
    labels = _validate_metric_inputs(output, labels, hierarchy)

    if not isinstance(k, int) or k <= 0:
        raise ValueError(f"k must be a positive integer, got {k}.")

    effective_k = min(k, output.shape[1])
    topk_predictions = output.topk(effective_k, dim=1).indices
    lca_heights = hierarchy.lca_heights.to(output.device)

    distances = lca_heights[
        labels.unsqueeze(1),
        topk_predictions,
    ]

    return distances.float().mean().item()

def _validate_metric_inputs(
    output: torch.Tensor,
    labels: torch.Tensor,
    hierarchy: Hierarchy,
):
    if output.ndim != 2:
        raise ValueError(
            f"output must have shape [N, C], got {output.shape}."
        )

    labels = labels.reshape(-1).to(output.device, dtype=torch.long)

    if output.shape[0] != labels.numel():
        raise ValueError(
            f"Sample mismatch: {output.shape[0]} outputs and "
            f"{labels.numel()} labels."
        )

    if output.shape[0] == 0:
        raise ValueError("Cannot evaluate empty tensors.")

    num_classes = output.shape[1]

    if num_classes != len(hierarchy.leaf_names):
        raise ValueError(
            f"output has {num_classes} classes, but hierarchy has "
            f"{len(hierarchy.leaf_names)} leaves."
        )

    if labels.min().item() < 0 or labels.max().item() >= num_classes:
        raise ValueError(
            f"Labels must be leaf indices in [0, {num_classes - 1}]."
        )

    return labels

@torch.no_grad()
def accuracy_at_depth(
    output: torch.Tensor,
    labels: torch.Tensor,
    hierarchy: Hierarchy,
) -> Dict[int, float]:
    labels = _validate_metric_inputs(output, labels, hierarchy)
    predictions = output.argmax(dim=1)

    depth_nodes = hierarchy.depth_nodes.to(output.device)

    true_paths = depth_nodes[labels]
    predicted_paths = depth_nodes[predictions]

    accuracies = (
        true_paths.eq(predicted_paths)
        .to(torch.float32)
        .mean(dim=0)
    )

    return {
        depth: accuracy.item()
        for depth, accuracy in enumerate(accuracies)
    }


@torch.no_grad()
def hierarchical_metrics(
    output: torch.Tensor,
    labels: torch.Tensor,
    tree: Dict[str, Any],
    nodes: Sequence[str],
    *,
    mistake_severity_on: bool = True,
    hierarchical_distance_on: bool = True,
    depth_accuracy_on: bool = True,
    topk: int = 10,
) -> Dict:
    if output.ndim != 2:
        raise ValueError(
            f"output must have shape [N, C], got {output.shape}."
        )

    labels = labels.reshape(-1).to(output.device, dtype=torch.long)

    if output.shape[0] != labels.numel():
        raise ValueError("output and labels have different sample counts.")

    num_classes = output.shape[1]

    if labels.numel() == 0:
        raise ValueError("Cannot evaluate empty tensors.")

    if labels.min() < 0 or labels.max() >= num_classes:
        raise ValueError(
            f"Leaf labels must be in [0, {num_classes - 1}]."
        )

    hierarchy = build_hierarchy(
        tree=tree,
        nodes=nodes,
        num_leaf_classes=num_classes,
    )

    results = {}

    if mistake_severity_on:
        results["mistake_severity"] = mistake_severity(
            output, labels, hierarchy
        )

    if hierarchical_distance_on:
        results[f"hierarchical_distance_top_{min(topk, num_classes)}"] = (
            hierarchical_distance_topk(
                output, labels, hierarchy, k=topk
            )
        )

    if depth_accuracy_on:
        results["accuracy_at_depth"] = accuracy_at_depth(
            output, labels, hierarchy
        )

    return results