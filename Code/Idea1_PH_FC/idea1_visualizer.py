"""
Code/Idea1_PH_FC/idea1_visualizer.py
Visualizations for Idea 1 — Persistent Homology on FC matrices.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from Core.base_visualizer import BaseVisualizer
from Core.utils import OutputManager


class Idea1Visualizer(BaseVisualizer):
    """All plots specific to Idea 1."""

    def __init__(self, output_manager: OutputManager, case_label: str = "ADHD"):
        super().__init__(output_manager)
        self.case_label = case_label
        # Override base style: no grid, larger fonts
        plt.rcParams.update({
            "axes.grid": False,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "legend.fontsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "font.size": 12,
        })

    # ------------------------------------------------------------------
    # FC / distance matrix
    # ------------------------------------------------------------------

    def plot_fc_matrix(self, fc_matrix: np.ndarray, subject_id: str) -> None:
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(fc_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax, label="Correlation")
        ax.set_title(f"FC Matrix — {subject_id}", fontsize=14)
        ax.set_xlabel("ROI index", fontsize=13)
        ax.set_ylabel("ROI index", fontsize=13)
        ax.grid(False)
        self.save_figure(fig, f"fc_matrix_{subject_id}.png")

    def plot_distance_matrix(self, dist_matrix: np.ndarray, subject_id: str) -> None:
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(dist_matrix, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax, label="Distance")
        ax.set_title(f"Distance Matrix (1-|FC|) — {subject_id}", fontsize=14)
        ax.grid(False)
        self.save_figure(fig, f"distance_matrix_{subject_id}.png")

    # ------------------------------------------------------------------
    # Persistence diagrams
    # ------------------------------------------------------------------

    def plot_group_diagrams(
        self,
        diagrams_adhd: List[np.ndarray],
        diagrams_ctrl: List[np.ndarray],
        dim: int,
        title: str = "",
    ) -> None:
        """Overlay all case and control persistence diagrams for one dimension."""
        fig, ax = plt.subplots(figsize=(6, 6))
        for dgm in diagrams_ctrl:
            if dgm.size > 0:
                ax.scatter(dgm[:, 0], dgm[:, 1], c="#2471A3", alpha=0.75, s=18)
        for dgm in diagrams_adhd:
            if dgm.size > 0:
                ax.scatter(dgm[:, 0], dgm[:, 1], c="#C0392B", alpha=0.75, s=18)
        all_vals = np.concatenate([
            np.concatenate([d for d in diagrams_adhd + diagrams_ctrl if d.size > 0])
        ])
        max_val = float(all_vals.max()) * 1.05 if all_vals.size > 0 else 1.0
        ax.plot([0, max_val], [0, max_val], "k--", lw=1.0)
        ax.set_xlim(0, max_val)
        ax.set_ylim(0, max_val)
        ax.set_xlabel("Birth", fontsize=13)
        ax.set_ylabel("Death", fontsize=13)
        ax.set_title(title or f"H{dim} Persistence Diagrams", fontsize=14)
        ax.grid(False)
        # Legend proxies
        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#C0392B",
                   markersize=10, label=self.case_label),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#2471A3",
                   markersize=10, label="Control"),
        ]
        ax.legend(handles=handles, fontsize=12)
        self.save_figure(fig, f"persistence_diagrams_H{dim}.png")

    # ------------------------------------------------------------------
    # Group comparison
    # ------------------------------------------------------------------

    def plot_total_persistence_comparison(
        self,
        results: dict,   # output of group_comparison_experiment
        dims: List[int] = [0, 1, 2],
    ) -> None:
        """Multi-panel boxplot of total persistence per dimension."""
        n_dims = len(dims)
        fig, axes = plt.subplots(1, n_dims, figsize=(4 * n_dims, 4), sharey=False)
        if n_dims == 1:
            axes = [axes]
        for ax, dim in zip(axes, dims):
            if dim not in results:
                continue
            adhd_tp = results[dim]["adhd_tp"]
            ctrl_tp = results[dim]["ctrl_tp"]
            pvalue = results[dim]["stats"]["pvalue"]
            ax.grid(False)
            self.plot_group_comparison(adhd_tp, ctrl_tp,
                                       f"H{dim} Total Persistence",
                                       ax=ax, pvalue=pvalue,
                                       case_label=self.case_label)
        fig.suptitle(f"Total Persistence: {self.case_label} vs Control",
                     y=1.02, fontsize=14)
        fig.tight_layout()
        self.save_figure(fig, "total_persistence_comparison.png")

    # ------------------------------------------------------------------
    # Wasserstein heatmap
    # ------------------------------------------------------------------

    def plot_wasserstein_heatmap(
        self, W: np.ndarray, labels: np.ndarray, dim: int = 1
    ) -> None:
        """
        Clustered heatmap of pairwise Wasserstein distances.
        Labels are sorted: control (0) then case (1).
        """
        n = len(labels)
        order = np.argsort(labels)
        W_sorted = W[np.ix_(order, order)]
        sorted_labels = labels[order]

        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(W_sorted, cmap="YlOrRd", aspect="auto")
        plt.colorbar(im, ax=ax, label="Wasserstein distance")

        # Divider line between groups
        n_ctrl = int(np.sum(sorted_labels == 0))
        ax.axhline(n_ctrl - 0.5, color="#1A5276", lw=2.0, ls="--")
        ax.axvline(n_ctrl - 0.5, color="#1A5276", lw=2.0, ls="--")

        case_prefix = self.case_label[0].upper()
        tick_labels = [f"C{i}" if l == 0 else f"{case_prefix}{i}"
                       for i, l in enumerate(sorted_labels)]
        ax.set_xticks(range(n))
        ax.set_xticklabels(tick_labels, rotation=90, fontsize=7)
        ax.set_yticks(range(n))
        ax.set_yticklabels(tick_labels, fontsize=7)
        ax.set_title(f"Pairwise Wasserstein Distance Matrix (H{dim})", fontsize=14)
        ax.grid(False)
        self.save_figure(fig, f"wasserstein_heatmap_H{dim}.png")

    # ------------------------------------------------------------------
    # Atlas scale comparison
    # ------------------------------------------------------------------

    def plot_atlas_scale_comparison(
        self, results_per_atlas: Dict[str, Dict[int, float]]
    ) -> None:
        """
        Bar chart: mean total persistence per atlas scale per dimension.

        Parameters
        ----------
        results_per_atlas : {atlas_name: {dim: mean_total_persistence}}
        """
        atlas_names = list(results_per_atlas.keys())
        dims = sorted(next(iter(results_per_atlas.values())).keys())

        x = np.arange(len(atlas_names))
        width = 0.25
        colors = ["#1F618D", "#1E8449", "#922B21"]
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, dim in enumerate(dims):
            means = [results_per_atlas[a][dim] for a in atlas_names]
            ax.bar(x + i * width, means, width, label=f"H{dim}", color=colors[i])
        ax.set_xticks(x + width)
        ax.set_xticklabels(atlas_names, fontsize=12)
        ax.set_ylabel("Mean Total Persistence", fontsize=13)
        ax.set_title("Total Persistence by Atlas Scale", fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(False)
        fig.tight_layout()
        self.save_figure(fig, "atlas_scale_comparison.png")

    # ------------------------------------------------------------------
    # Subtype clusters
    # ------------------------------------------------------------------

    def plot_subtype_clusters(
        self,
        tp_values: np.ndarray,
        cluster_labels: np.ndarray,
        adhd_labels: np.ndarray,
        dim: int = 1,
    ) -> None:
        """Scatter of H{dim} total persistence colored by cluster and case status."""
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        colors_cluster = ["#C0392B", "#2471A3"]
        for cl in np.unique(cluster_labels):
            mask = cluster_labels == cl
            axes[0].scatter(np.where(mask)[0], tp_values[mask],
                            c=colors_cluster[cl], label=f"Cluster {cl}", s=50)
        axes[0].set_title(f"H{dim} Total Persistence by Cluster", fontsize=14)
        axes[0].set_xlabel("Subject index", fontsize=13)
        axes[0].set_ylabel("Total Persistence", fontsize=13)
        axes[0].legend(fontsize=12)
        axes[0].grid(False)

        colors_adhd = np.where(adhd_labels == 1, "#C0392B", "#2471A3")
        axes[1].scatter(cluster_labels, tp_values, c=colors_adhd, s=50)
        axes[1].set_xlabel("Cluster", fontsize=13)
        axes[1].set_ylabel("Total Persistence", fontsize=13)
        axes[1].set_title(f"H{dim} Clusters vs {self.case_label} Label", fontsize=14)
        axes[1].grid(False)
        from matplotlib.lines import Line2D
        axes[1].legend(handles=[
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#C0392B",
                   markersize=10, label=self.case_label),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#2471A3",
                   markersize=10, label="Control"),
        ], fontsize=12)
        fig.tight_layout()
        self.save_figure(fig, f"subtype_clusters_H{dim}.png")
