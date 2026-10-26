# https://github.com/uiuctml/HypStructure/blob/main/hierarchy/data.py
from typing import *
import numpy as np

import networkx as nx
import os
import json

class Hierarchy():
    '''
    Class representing a general ImageNet-style hierarchy, modified from 
    https://github.com/MadryLab/robustness/blob/master/robustness/tools/breeds_helpers.py 
    '''
    def __init__(self, dataset_name, info_dir='/path/to/repository/', *args, **kw):
        """
        Args:
            info_dir (str) : Base path to datasets and hierarchy information files. Contains a 
                "class_hierarchy.txt" file with one edge per line, a
                "node_names.txt" mapping nodes to names, and "dataset_class_info.json".
        """
        super(Hierarchy, self).__init__(*args, **kw)
        
        self.dataset_name = dataset_name
        self.info_dir = info_dir
        
        REQUIRED_FILES = [f'{self.dataset_name}/dataset_class_info.json',
                  f'{self.dataset_name}/class_hierarchy.txt',
                  f'{self.dataset_name}/node_names.txt']

        # for f in REQUIRED_FILES:
        #     if not os.path.exists(os.path.join(self.info_dir, f)):
        #         self.generate_files()
            
        # Details about dataset class names (leaves), IDS
        with open(os.path.join(self.info_dir, f"{self.dataset_name}/dataset_class_info.json")) as f:
            class_info = json.load(f)

        # Hierarchy represented as edges between parent & child nodes.
        with open(os.path.join(self.info_dir, f'{self.dataset_name}/class_hierarchy.txt')) as f:
            edges = [l.strip().split() for l in f.readlines()]

        # Information (names, IDs) for intermediate nodes in hierarchy.
        with open(os.path.join(self.info_dir,  f'{self.dataset_name}/node_names.txt')) as f:
            mapping = [l.strip().split('\t') for l in f.readlines()]


        # Original dataset classes
        self.LEAF_IDS = [c[1] for c in class_info] # wnid
        self.LEAF_ID_TO_NAME = {c[1]: c[2] for c in class_info} # wnid : name
        self.LEAF_ID_TO_NUM = {c[1]: c[0] for c in class_info} # wnid : labelid
        self.LEAF_NUM_TO_NAME = {c[0]: c[2] for c in class_info} # labelid : name

        # Full hierarchy
        self.HIER_NODE_NAME = {w[0]: w[1] for w in mapping} # wnid : name
        self.NAME_TO_NODE_ID = {w[1]: w[0] for w in mapping} # name : wnid
        self.graph = self._make_parent_graph(self.LEAF_IDS, edges)

        # make label mapping
        self.label_map = self.get_label_mapping()
        # generate tree distance
        self.tree_dist = self.generate_tree_dist()
        # leave node names
        self.leaf_names = [c[2] for c in class_info] 
        


    @staticmethod
    def _make_parent_graph(nodes, edges):
        """
        Obtain networkx representation of class hierarchy.

        Args:
            nodes [str] : List of node names to traverse upwards.
            edges [(str, str)] : Tuples of parent-child pairs.

        Return:
            networkx representation of the graph.
        """

        # create full graph
        full_graph_dir = {}
        for p, c in edges:
            if p not in full_graph_dir:
                full_graph_dir[p] = {c: 1}
            else:
                full_graph_dir[p].update({c: 1})
                    
        FG = nx.DiGraph(full_graph_dir)

        # perform backward BFS to get the relevant graph
        graph_dir = {}
        todo = [n for n in nodes if n in FG.nodes()] # skip nodes not in graph
        while todo:
            curr = todo
            todo = []
            for w in curr:
                for p in FG.predecessors(w):
                    if p not in graph_dir:
                        graph_dir[p] = {w: 1}
                    else:
                        graph_dir[p].update({w: 1})
                    todo.append(p)
            todo = set(todo)
        
        return nx.DiGraph(graph_dir)

    def get_root_node(self):
        for node in self.graph.nodes():
            if self.graph.in_degree(node) == 0:
                return node
    
    def get_ancestors(self, node): # leaf to root path
        return nx.shortest_path(self.graph, source=self.get_root_node(), target=node)[::-1]
    
    def find_leaf_nodes(self):
        return [node for node in self.graph.nodes() if self.graph.out_degree(node) == 0]
    
    def get_label_mapping(self):
        leaf_nodes = self.find_leaf_nodes()
        leaf_ancestors = {}
        max_path = -1
        for leaf in leaf_nodes:
            ancestors = self.get_ancestors(leaf)
            # here leaf is wnid
            # convert to label id
            if len(ancestors) > max_path:
                max_path = len(ancestors)
            leaf_ancestors[self.LEAF_ID_TO_NUM[leaf]] = ancestors
        
        label_map = np.empty((len(leaf_nodes), max_path),dtype=object)
        for leaf in leaf_nodes:
            true_path = leaf_ancestors[self.LEAF_ID_TO_NUM[leaf]]
            padded_path = [''] * (max_path - len(true_path)) + true_path
            label_map[self.LEAF_ID_TO_NUM[leaf]] = padded_path

        return label_map


    def generate_tree_dist(self):
        distances = {}
        all_distances = dict(nx.all_pairs_shortest_path_length(self.graph.to_undirected()))
        for node1, paths in all_distances.items():
            for node2, dist in paths.items():
                if node1 != node2:
                    node_pair = (node1, node2)
                    rev_pair = (node2, node1)
                    if node_pair not in distances:
                        distances[node_pair] = dist
                        distances[rev_pair] = dist
        return distances
    
    
    def generate_files(self):
        # specify how to generate files for each dataset
        return None





## modified hierarchy classes section to facilitate the weighted version 


class Hierarchy_W():
    """
    Hierarchy with optional weighted edges.

    class_hierarchy.txt supports either:

        parent child
        parent child weight

    If weight is missing, weight = 1.0.
    """

    def __init__(self, dataset_name, info_dir='/path/to/repository/', *args, **kw):
        super(Hierarchy_W, self).__init__(*args, **kw)

        self.dataset_name = dataset_name
        self.info_dir = info_dir

        REQUIRED_FILES = [
            f'{self.dataset_name}/dataset_class_info.json',
            f'{self.dataset_name}/class_hierarchy.txt',
            f'{self.dataset_name}/node_names.txt',
        ]

        with open(os.path.join(self.info_dir, f"{self.dataset_name}/dataset_class_info.json")) as f:
            class_info = json.load(f)

        # with open(os.path.join(self.info_dir, f'{self.dataset_name}/class_hierarchy.txt')) as f:
        with open(os.path.join(self.info_dir, f'{self.dataset_name}/class_hierarchy_w.txt')) as f:
            raw_edges = [l.strip().split() for l in f.readlines() if l.strip()]

        edges = self._parse_edges(raw_edges)

        with open(os.path.join(self.info_dir, f'{self.dataset_name}/node_names.txt')) as f:
            mapping = [l.strip().split('\t') for l in f.readlines() if l.strip()]

        self.LEAF_IDS = [c[1] for c in class_info]
        self.LEAF_ID_TO_NAME = {c[1]: c[2] for c in class_info}
        self.LEAF_ID_TO_NUM = {c[1]: c[0] for c in class_info}
        self.LEAF_NUM_TO_NAME = {c[0]: c[2] for c in class_info}

        self.HIER_NODE_NAME = {w[0]: w[1] for w in mapping}
        self.NAME_TO_NODE_ID = {w[1]: w[0] for w in mapping}

        self.graph = self._make_parent_graph(self.LEAF_IDS, edges)

        self.label_map = self.get_label_mapping()
        self.tree_dist = self.generate_tree_dist()
        self.leaf_names = [c[2] for c in class_info]

    @staticmethod
    def _parse_edges(raw_edges):
        """
        Parse hierarchy edges.

        Supports:
            parent child
            parent child weight

        Returns:
            list of (parent, child, weight)
        """
        edges = []

        for e in raw_edges:
            if len(e) == 2:
                parent, child = e
                weight = 1.0
            elif len(e) == 3:
                parent, child, weight = e
                weight = float(weight)
            else:
                raise ValueError(f"Bad edge format: {e}")

            edges.append((parent, child, weight))

        return edges

    @staticmethod
    def _make_parent_graph(nodes, edges):
        """
        Obtain weighted networkx representation of class hierarchy.

        Args:
            nodes: leaf node IDs to traverse upward.
            edges: list of (parent, child, weight).

        Returns:
            nx.DiGraph containing only ancestors of relevant leaf nodes.
        """

        full_graph_dir = {}

        for parent, child, weight in edges:
            if parent not in full_graph_dir:
                full_graph_dir[parent] = {child: {"weight": weight}}
            else:
                full_graph_dir[parent][child] = {"weight": weight}

        FG = nx.DiGraph(full_graph_dir)

        graph_dir = {}

        todo = [n for n in nodes if n in FG.nodes()]

        while todo:
            curr = todo
            todo = []

            for child in curr:
                for parent in FG.predecessors(child):
                    weight = FG[parent][child].get("weight", 1.0)

                    if parent not in graph_dir:
                        graph_dir[parent] = {child: {"weight": weight}}
                    else:
                        graph_dir[parent][child] = {"weight": weight}

                    todo.append(parent)

            todo = list(set(todo))

        return nx.DiGraph(graph_dir)

    def get_root_node(self):
        roots = [node for node in self.graph.nodes() if self.graph.in_degree(node) == 0]

        if len(roots) == 0:
            raise ValueError("No root node found.")

        if len(roots) > 1:
            raise ValueError(f"Multiple root nodes found: {roots}")

        return roots[0]

    def get_ancestors(self, node):
        """
        Return leaf-to-root path.
        """
        return nx.shortest_path(
            self.graph,
            source=self.get_root_node(),
            target=node
        )[::-1]

    def find_leaf_nodes(self):
        return [node for node in self.graph.nodes() if self.graph.out_degree(node) == 0]

    def get_label_mapping(self):
        leaf_nodes = self.find_leaf_nodes()

        leaf_ancestors = {}
        max_path = -1

        for leaf in leaf_nodes:
            ancestors = self.get_ancestors(leaf)

            if len(ancestors) > max_path:
                max_path = len(ancestors)

            leaf_ancestors[self.LEAF_ID_TO_NUM[leaf]] = ancestors

        label_map = np.empty((len(leaf_nodes), max_path), dtype=object)

        for leaf in leaf_nodes:
            label_id = self.LEAF_ID_TO_NUM[leaf]
            true_path = leaf_ancestors[label_id]
            padded_path = [''] * (max_path - len(true_path)) + true_path
            label_map[label_id] = padded_path

        return label_map

    def generate_tree_dist(self):
        """
        Generate weighted tree distances.

        Uses edge attribute "weight".
        If all weights are 1.0, this gives the same result as the old
        unweighted shortest path length.
        """
        distances = {}

        undirected_graph = self.graph.to_undirected()

        all_distances = dict(
            nx.all_pairs_dijkstra_path_length(
                undirected_graph,
                weight="weight"
            )
        )

        for node1, paths in all_distances.items():
            for node2, dist in paths.items():
                if node1 != node2:
                    distances[(node1, node2)] = float(dist)
                    distances[(node2, node1)] = float(dist)

        return distances

    def generate_files(self):
        return None



class Hierarchy_ZH:
    """
    Build CPCC hierarchy directly from CLASS_COLS, TREE, NODES.
    Inputs:
        class_cols : leaf classes in training-label order
        tree       : nested hierarchy dictionary
        nodes      : all nodes in node-ID order
        weights    : optional depth weights; e.g. [0, 0.5, 1].
                     None -> all edge weights = 1.0
    Outputs:
        graph      : weighted nx.DiGraph (parent -> child)
        label_map  : leaf-to-root node-ID paths [num_classes, max_depth]
        tree_dist  : weighted pairwise tree distances
        leaf_names : class names in training-label order
    """

    def __init__(self, class_cols, tree, nodes, weights=None):
        self.class_cols, self.tree, self.nodes, self.weights = class_cols, tree, nodes, weights

        # Node mappings: name <-> string node ID
        self.NAME_TO_NODE_ID = {name: str(i) for i, name in enumerate(nodes)}
        self.HIER_NODE_NAME = {str(i): name for i, name in enumerate(nodes)}

        # Leaf mappings: training label order follows class_cols
        self.LEAF_IDS = [self.NAME_TO_NODE_ID[x] for x in class_cols]
        self.LEAF_ID_TO_NAME = {self.NAME_TO_NODE_ID[x]: x for x in class_cols}
        self.LEAF_ID_TO_NUM = {self.NAME_TO_NODE_ID[x]: i for i, x in enumerate(class_cols)}
        self.LEAF_NUM_TO_NAME = {i: x for i, x in enumerate(class_cols)}
        self.leaf_names = list(class_cols)

        edges = self._extract_edges(tree)                         # [(parent_id, child_id, weight), ...]
        self.graph = self._make_parent_graph(self.LEAF_IDS, edges)
        self.label_map = self.get_label_mapping()                 # [num_classes, max_depth], leaf -> root
        self.tree_dist = self.generate_tree_dist()                # {(node1,node2): weighted_distance}


    def _extract_edges(self, tree, depth=0):
        """Nested TREE -> weighted (parent_id, child_id, weight) edges."""
        edges = []
        for parent, children in tree.items():
            if not isinstance(children, (dict, list)): raise TypeError(f"Children of {parent!r} must be dict/list.")

            child_names = children.keys() if isinstance(children, dict) else children
            weight = 1.0 if self.weights is None else self.weights[depth + 1]  # child-depth weight

            edges += [(self.NAME_TO_NODE_ID[parent], self.NAME_TO_NODE_ID[c], float(weight)) for c in child_names]
            if isinstance(children, dict): edges += self._extract_edges(children, depth + 1)

        return edges


    @staticmethod
    def _make_parent_graph(nodes, edges):
        """Keep weighted parent->child graph containing the supplied leaves and their ancestors."""
        FG = nx.DiGraph()
        FG.add_weighted_edges_from(edges)

        graph = nx.DiGraph()
        todo = list(nodes)

        while todo:
            parents = []
            for child in todo:
                for parent in FG.predecessors(child):
                    graph.add_edge(parent, child, weight=FG[parent][child].get("weight", 1.0))
                    parents.append(parent)
            todo = list(set(parents))

        return graph


    def get_root_node(self):
        """Return unique root node ID."""
        roots = [x for x in self.graph if self.graph.in_degree(x) == 0]
        if len(roots) != 1: raise ValueError(f"Expected one root, found: {roots}")
        return roots[0]


    def get_ancestors(self, node):
        """Input: node ID -> Output: [leaf, ..., root]."""
        return nx.shortest_path(self.graph, self.get_root_node(), node)[::-1]


    def find_leaf_nodes(self):
        """Return graph leaf node IDs."""
        return [x for x in self.graph if self.graph.out_degree(x) == 0]


    def get_label_mapping(self):
        """Output: label_map[label_id] = padded [leaf, ..., root] node-ID path."""
        leaves = self.find_leaf_nodes()
        paths = {self.LEAF_ID_TO_NUM[x]: self.get_ancestors(x) for x in leaves}
        max_depth = max(map(len, paths.values()))

        label_map = np.empty((len(leaves), max_depth), dtype=object)
        for label_id, path in paths.items(): label_map[label_id] = [''] * (max_depth - len(path)) + path
        return label_map


    def generate_tree_dist(self):
        """Output: {(node1,node2): weighted shortest-path distance}; weights=None -> unit edges."""
        graph = self.graph.to_undirected()
        all_dist = nx.all_pairs_dijkstra_path_length(graph, weight="weight")

        return {
            (u, v): float(d)
            for u, paths in all_dist
            for v, d in paths.items()
            if u != v
        }

