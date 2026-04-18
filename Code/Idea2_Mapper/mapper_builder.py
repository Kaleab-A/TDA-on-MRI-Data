"""
Code/Idea2_Mapper/mapper_builder.py
Constructs Mapper graphs for individual subjects or populations.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

from Code.Idea2_Mapper.lens_functions import LensFactory
from Parameters.params_idea2 import Idea2Params


class MapperBuilder:
    """
    Builds Mapper graphs from fMRI time-point clouds.

    The Mapper algorithm:
    1. Apply a lens function to reduce the data to 1D.
    2. Cover the 1D range with overlapping intervals.
    3. Within each interval, cluster the preimages.
    4. Connect nodes (clusters) that share points.
    """

    def __init__(self, params: Idea2Params):
        self.params = params
        self._lens = LensFactory.create(
            params.lens_function,
            component=params.pca_component,
            window=params.variance_window,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_for_subject(
        self,
        time_series: np.ndarray,
        n_intervals: Optional[int] = None,
        overlap_fraction: Optional[float] = None,
    ) -> dict:
        """
        Build a Mapper graph for a single subject.

        Parameters
        ----------
        time_series : np.ndarray, shape (n_timepoints, n_rois)

        Returns
        -------
        dict with keys:
            'nodes'       : {node_id: List[int]}  — point indices per node
            'edges'       : List[Tuple[int, int]]
            'lens_values' : np.ndarray, shape (n_timepoints, 1)
            'n_nodes'     : int
            'n_edges'     : int
        """
        n_intervals = n_intervals or self.params.n_intervals
        overlap_fraction = overlap_fraction or self.params.overlap_fraction

        # Optional PCA reduction
        X = self._maybe_reduce(time_series)

        # Lens
        lens_values = self._lens.fit_transform(X)  # (n_timepoints, 1)

        # Cover + cluster
        nodes, edges = self._build_cover_and_cluster(
            X, lens_values.flatten(), n_intervals, overlap_fraction)

        return {
            "nodes": nodes,
            "edges": edges,
            "lens_values": lens_values,
            "n_nodes": len(nodes),
            "n_edges": len(edges),
        }

    def build_population_mapper(
        self,
        all_time_series: List[np.ndarray],
        labels: np.ndarray,
    ) -> dict:
        """
        Build a single Mapper graph by pooling all subjects' time points.
        Each point is annotated with its subject index and ADHD label.
        """
        X_list, subject_ids, label_ids = [], [], []
        for subj_idx, (ts, lbl) in enumerate(zip(all_time_series, labels)):
            X = self._maybe_reduce(ts)
            X_list.append(X)
            subject_ids.extend([subj_idx] * len(X))
            label_ids.extend([lbl] * len(X))

        X_all = np.vstack(X_list)
        subject_ids = np.array(subject_ids)
        label_ids = np.array(label_ids)

        lens_values = self._lens.fit_transform(X_all).flatten()
        nodes, edges = self._build_cover_and_cluster(
            X_all, lens_values,
            self.params.n_intervals, self.params.overlap_fraction,
        )

        # Annotate nodes with ADHD fraction
        node_adhd_fraction = {}
        for node_id, point_indices in nodes.items():
            node_labels = label_ids[point_indices]
            node_adhd_fraction[node_id] = float(np.mean(node_labels == 1))

        return {
            "nodes": nodes,
            "edges": edges,
            "lens_values": lens_values,
            "node_adhd_fraction": node_adhd_fraction,
            "label_ids": label_ids,
            "subject_ids": subject_ids,
            "n_nodes": len(nodes),
            "n_edges": len(edges),
        }

    def compute_graph_statistics(self, mapper_graph: dict) -> dict:
        """Extract scalar topology metrics from a Mapper graph."""
        nodes = mapper_graph["nodes"]
        edges = mapper_graph["edges"]
        n_nodes = len(nodes)
        n_edges = len(edges)
        node_sizes = [len(pts) for pts in nodes.values()]

        # Connected components via union-find
        n_components = self._count_components(list(nodes.keys()), edges)

        return {
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "n_components": n_components,
            "mean_node_size": float(np.mean(node_sizes)) if node_sizes else 0.0,
            "max_node_size": int(max(node_sizes)) if node_sizes else 0,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _maybe_reduce(self, X: np.ndarray) -> np.ndarray:
        """Optionally reduce dimensionality with PCA before lens."""
        n_comp = min(self.params.pca_n_components, X.shape[0], X.shape[1])
        if n_comp < X.shape[1]:
            pca = PCA(n_components=n_comp, random_state=self.params.random_seed)
            return pca.fit_transform(X)
        return X

    def _build_cover_and_cluster(
        self,
        X: np.ndarray,
        lens: np.ndarray,
        n_intervals: int,
        overlap_fraction: float,
    ) -> Tuple[Dict[int, List[int]], List[Tuple[int, int]]]:
        """
        Cover [min_lens, max_lens] with overlapping intervals,
        cluster points in each interval, build graph.
        """
        l_min, l_max = lens.min(), lens.max()
        step = (l_max - l_min) / n_intervals
        half_overlap = overlap_fraction * step

        node_id = 0
        nodes: Dict[int, List[int]] = {}
        point_to_nodes: Dict[int, List[int]] = defaultdict(list)

        for i in range(n_intervals):
            center = l_min + (i + 0.5) * step
            lo = center - step / 2 - half_overlap
            hi = center + step / 2 + half_overlap
            in_interval = np.where((lens >= lo) & (lens <= hi))[0]
            if len(in_interval) == 0:
                continue

            X_interval = X[in_interval]
            n_clusters = min(self.params.n_clusters, len(in_interval))
            if n_clusters < 2:
                # Single cluster
                nodes[node_id] = list(in_interval)
                for pt in in_interval:
                    point_to_nodes[int(pt)].append(node_id)
                node_id += 1
            else:
                clustering = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    linkage=self.params.clusterer_linkage,
                )
                cluster_labels = clustering.fit_predict(X_interval)
                for cl in range(n_clusters):
                    members = in_interval[cluster_labels == cl]
                    if len(members) == 0:
                        continue
                    nodes[node_id] = list(members)
                    for pt in members:
                        point_to_nodes[int(pt)].append(node_id)
                    node_id += 1

        # Edges: nodes sharing at least one point
        edges_set: set = set()
        for node_list in point_to_nodes.values():
            for a in range(len(node_list)):
                for b in range(a + 1, len(node_list)):
                    e = (min(node_list[a], node_list[b]),
                         max(node_list[a], node_list[b]))
                    edges_set.add(e)
        edges = list(edges_set)
        return nodes, edges

    @staticmethod
    def _count_components(node_ids: List[int],
                           edges: List[Tuple[int, int]]) -> int:
        """Union-Find connected component count."""
        parent = {n: n for n in node_ids}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            parent[find(a)] = find(b)

        for u, v in edges:
            if u in parent and v in parent:
                union(u, v)
        return len({find(n) for n in node_ids})
