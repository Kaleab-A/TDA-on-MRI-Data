"""
Code/Idea1_PH_FC/distance_analyzer.py
Group-level statistical comparisons of persistence diagrams.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from persim import wasserstein, bottleneck
from sklearn.cluster import KMeans

from Core.utils import OutputManager, StatisticsHelper, SubjectRecord
from Parameters.params_idea1 import Idea1Params


class PersistenceDistanceAnalyzer:
    """
    Computes pairwise Wasserstein/bottleneck distances between persistence
    diagrams and runs group-level statistical comparisons.
    """

    def __init__(self, params: Idea1Params, output_manager: OutputManager):
        self.params = params
        self.output_manager = output_manager

    # ------------------------------------------------------------------
    # Pairwise distances
    # ------------------------------------------------------------------

    def compute_wasserstein_distance(
        self, dgm_a: np.ndarray, dgm_b: np.ndarray, order: int = 1
    ) -> float:
        """Wasserstein distance between two persistence diagrams."""
        if dgm_a.size == 0 and dgm_b.size == 0:
            return 0.0
        if dgm_a.size == 0:
            dgm_a = np.zeros((1, 2))
        if dgm_b.size == 0:
            dgm_b = np.zeros((1, 2))
        return float(wasserstein(dgm_a, dgm_b, matching=False))

    def compute_bottleneck_distance(
        self, dgm_a: np.ndarray, dgm_b: np.ndarray
    ) -> float:
        """Bottleneck distance between two persistence diagrams."""
        if dgm_a.size == 0 and dgm_b.size == 0:
            return 0.0
        if dgm_a.size == 0:
            dgm_a = np.zeros((1, 2))
        if dgm_b.size == 0:
            dgm_b = np.zeros((1, 2))
        return float(bottleneck(dgm_a, dgm_b, matching=False))

    def compute_pairwise_matrix(
        self, diagrams: List[np.ndarray], metric: str = "wasserstein"
    ) -> np.ndarray:
        """
        Compute (n_subjects, n_subjects) pairwise distance matrix.

        Parameters
        ----------
        metric : 'wasserstein' or 'bottleneck'
        """
        n = len(diagrams)
        D = np.zeros((n, n))
        dist_fn = (self.compute_wasserstein_distance
                   if metric == "wasserstein"
                   else self.compute_bottleneck_distance)
        for i in range(n):
            for j in range(i + 1, n):
                d = dist_fn(diagrams[i], diagrams[j])
                D[i, j] = d
                D[j, i] = d
        return D

    # ------------------------------------------------------------------
    # Experiment 1: Group comparison
    # ------------------------------------------------------------------

    def group_comparison_experiment(
        self,
        diagrams_per_dim: Dict[int, List[np.ndarray]],
        labels: np.ndarray,
    ) -> dict:
        """
        For each homological dimension, compute total persistence per subject
        and test ADHD vs control with Mann-Whitney U.

        Returns dict with per-dimension results and saves a CSV.
        """
        from Code.Idea1_PH_FC.persistence_computer import PHFCComputer
        adhd_mask = labels == 1
        rows = []
        results = {}
        for dim, diagrams in diagrams_per_dim.items():
            tp = np.array([PHFCComputer.total_persistence(PHFCComputer, d) for d in diagrams])
            adhd_tp = tp[adhd_mask]
            ctrl_tp = tp[~adhd_mask]
            stats = StatisticsHelper.mann_whitney_u(adhd_tp, ctrl_tp)
            stats["dimension"] = dim
            stats["adhd_mean"] = float(np.mean(adhd_tp))
            stats["control_mean"] = float(np.mean(ctrl_tp))
            stats["cohen_d"] = StatisticsHelper.cohen_d(adhd_tp, ctrl_tp)
            rows.append(stats)
            results[dim] = {"total_persistence": tp, "stats": stats,
                            "adhd_tp": adhd_tp, "ctrl_tp": ctrl_tp}
            print(f"  H{dim} group comparison: p={stats['pvalue']:.4f}, "
                  f"d={stats['cohen_d']:.3f}")

        df = pd.DataFrame(rows)
        self.output_manager.save_dataframe(df, "group_comparison_total_persistence.csv")
        return results

    # ------------------------------------------------------------------
    # Experiment 2: H0 vs H1 comparison
    # ------------------------------------------------------------------

    def h0_vs_h1_experiment(
        self, diagrams_per_dim: Dict[int, List[np.ndarray]]
    ) -> dict:
        """
        Compare H0 vs H1 total persistence distributions across all subjects.
        Saves summary statistics.
        """
        from Core.base_tda import BasePersistenceComputer as BPC
        rows = []
        for dim, diagrams in diagrams_per_dim.items():
            tp_values = np.array([BPC.total_persistence(d) for d in diagrams])
            rows.append({
                "dimension": dim,
                "mean": float(np.mean(tp_values)),
                "std": float(np.std(tp_values)),
                "median": float(np.median(tp_values)),
                "min": float(np.min(tp_values)),
                "max": float(np.max(tp_values)),
            })
        df = pd.DataFrame(rows)
        self.output_manager.save_dataframe(df, "h0_h1_h2_summary.csv")
        return {"summary": df, "diagrams_per_dim": diagrams_per_dim}

    # ------------------------------------------------------------------
    # Experiment 3: Subtype analysis
    # ------------------------------------------------------------------

    def subtype_analysis_experiment(
        self, diagrams: List[np.ndarray], records: List[SubjectRecord]
    ) -> dict:
        """
        Cluster ADHD subjects by H1 total persistence.
        Uses KMeans with 2 clusters (inattentive vs combined).
        """
        from Core.base_tda import BasePersistenceComputer as BPC
        tp_values = np.array([BPC.total_persistence(d) for d in diagrams]).reshape(-1, 1)

        rng = np.random.default_rng(self.params.random_seed)
        km = KMeans(n_clusters=2, random_state=self.params.random_seed, n_init=10)
        cluster_labels = km.fit_predict(tp_values)

        rows = []
        for rec, tp, cl in zip(records, tp_values.flatten(), cluster_labels):
            rows.append({
                "subject_id": rec.subject_id,
                "adhd_label": rec.adhd_label,
                "h1_total_persistence": tp,
                "cluster": int(cl),
                "adhd_measure": rec.adhd_measure,
            })
        df = pd.DataFrame(rows)
        self.output_manager.save_dataframe(df, "subtype_clusters.csv")

        # Compute cluster separation score
        from sklearn.metrics import silhouette_score
        if len(np.unique(cluster_labels)) > 1:
            sil = float(silhouette_score(tp_values, cluster_labels))
        else:
            sil = float("nan")
        print(f"  Subtype clustering silhouette score: {sil:.3f}")
        return {"cluster_df": df, "silhouette_score": sil,
                "cluster_labels": cluster_labels}

    # ------------------------------------------------------------------
    # Pairwise Wasserstein matrix (for heatmap)
    # ------------------------------------------------------------------

    def compute_group_wasserstein_matrix(
        self, diagrams: List[np.ndarray], labels: np.ndarray
    ) -> np.ndarray:
        """Pairwise Wasserstein matrix sorted by group label."""
        order = np.argsort(labels)
        sorted_diagrams = [diagrams[i] for i in order]
        return self.compute_pairwise_matrix(sorted_diagrams, metric="wasserstein")
