"""
Code/Idea5_EulerCharacteristic/idea5_visualizer.py
Visualizations for Idea 5 — Euler characteristic and Betti curves.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from Core.base_visualizer import BaseVisualizer
from Core.utils import OutputManager


class Idea5Visualizer(BaseVisualizer):

    def __init__(self, output_manager: OutputManager):
        super().__init__(output_manager)

    # ------------------------------------------------------------------
    # FC matrix heatmaps
    # ------------------------------------------------------------------

    def plot_group_fc_matrices(
        self,
        fc_adhd: np.ndarray,
        fc_ctrl: np.ndarray,
        roi_labels: List[str],
        filename: str = "fc_matrices_group.png",
    ) -> None:
        """
        Side-by-side heatmaps of group-averaged FC matrices (ADHD vs Control)
        plus a difference matrix (ADHD - Control).
        """
        diff = fc_adhd - fc_ctrl
        vmax = np.percentile(np.abs(np.concatenate([fc_adhd.ravel(),
                                                     fc_ctrl.ravel()])), 98)
        dmax = np.percentile(np.abs(diff), 98)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        titles = ["Mean FC — ADHD", "Mean FC — Control", "Difference (ADHD − Control)"]
        matrices = [fc_adhd, fc_ctrl, diff]
        cmaps = ["RdBu_r", "RdBu_r", "RdBu_r"]
        vlims = [(-vmax, vmax), (-vmax, vmax), (-dmax, dmax)]

        n_rois = len(roi_labels)
        tick_step = max(1, n_rois // 10)   # show at most ~10 tick labels
        tick_pos = list(range(0, n_rois, tick_step))
        tick_labels = [roi_labels[i] for i in tick_pos]

        for ax, mat, title, cmap, (vmin, vm) in zip(
                axes, matrices, titles, cmaps, vlims):
            im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vm, aspect="auto")
            ax.set_title(title, fontsize=11)
            ax.set_xticks(tick_pos)
            ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)
            ax.set_yticks(tick_pos)
            ax.set_yticklabels(tick_labels, fontsize=7)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.suptitle("Group-Averaged Functional Connectivity (MSDL, 39 ROIs)",
                     fontsize=13)
        fig.tight_layout()
        self.save_figure(fig, filename)

    # ------------------------------------------------------------------
    # Betti curves + EC
    # ------------------------------------------------------------------

    def plot_group_betti_curves(
        self,
        group_curves: dict,
        filename: str = "betti_curves_group.png",
    ) -> None:
        """Mean +/- std b0, b1, b2 curves for ADHD vs Control."""
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
        eps = group_curves.get("epsilon", np.array([]))
        keys = ["betti_0", "betti_1", "betti_2"]
        labels_plot = ["b0 (components)", "b1 (loops)", "b2 (voids)"]
        colors = {"adhd": "#e8a09a", "control": "#9abfe8"}

        for ax, key, label in zip(axes, keys, labels_plot):
            for group in ("adhd", "control"):
                g = group_curves.get(group, {})
                if not g:
                    continue
                mean = g.get(key)
                std = g.get(f"{key}_std")
                if mean is None:
                    continue
                ax.plot(eps, mean, color=colors[group],
                        label=group.capitalize())
                if std is not None:
                    ax.fill_between(eps, mean - std, mean + std,
                                    color=colors[group], alpha=0.2)
            ax.set_xlabel("Filtration epsilon")
            ax.set_ylabel("Betti number")
            ax.set_title(label)
            ax.legend()
        fig.suptitle("Betti Curves: ADHD vs Control", fontsize=13)
        fig.tight_layout()
        self.save_figure(fig, filename)

    def plot_group_ec_curves(
        self,
        group_curves: dict,
        filename: str = "euler_characteristic_curves.png",
    ) -> None:
        """Mean ± std Euler characteristic curves."""
        fig, ax = plt.subplots(figsize=(8, 4))
        eps = group_curves.get("epsilon", np.array([]))
        colors = {"adhd": "#e8a09a", "control": "#9abfe8"}
        for group in ("adhd", "control"):
            g = group_curves.get(group, {})
            ec = g.get("euler")
            ec_std = g.get("euler_std")
            if ec is None:
                continue
            ax.plot(eps, ec, color=colors[group], label=group.capitalize(), lw=2)
            if ec_std is not None:
                ax.fill_between(eps, ec - ec_std, ec + ec_std,
                                color=colors[group], alpha=0.2)
        ax.axhline(0, color="black", ls="--", lw=0.8)
        ax.set_xlabel("Filtration epsilon")
        ax.set_ylabel("Euler characteristic X(epsilon)")
        ax.set_title("Euler Characteristic Curves: ADHD vs Control")
        ax.legend()
        fig.tight_layout()
        self.save_figure(fig, filename)

    # ------------------------------------------------------------------
    # Severity correlation
    # ------------------------------------------------------------------

    def plot_severity_correlation(
        self,
        corr_result: dict,
        dim: int,
        filename: str = None,
    ) -> None:
        """Scatter plot: area under β_dim curve vs ADHD severity score."""
        if "areas" not in corr_result:
            return
        fname = filename or f"severity_correlation_H{dim}.png"
        areas = corr_result["areas"]
        # We don't have severity_scores here — assume caller passes them or
        # the result dict contains them
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(areas, areas * 0, alpha=0.0)  # placeholder
        ax.set_xlabel(f"Area under β{dim} curve")
        ax.set_ylabel("ADHD severity score")
        ax.set_title(f"H{dim} area vs ADHD Severity\n"
                     f"ρ={corr_result.get('correlation', 'N/A'):.3f}, "
                     f"p={corr_result.get('pvalue', 'N/A'):.3f}")
        self.save_figure(fig, fname)

    def plot_severity_scatter(
        self,
        areas: np.ndarray,
        severity: np.ndarray,
        labels: np.ndarray,
        dim: int,
        corr: float,
        pvalue: float,
        filename: str = None,
    ) -> None:
        """Proper scatter: area vs severity, colored by ADHD label."""
        fname = filename or f"severity_scatter_H{dim}.png"
        fig, ax = plt.subplots(figsize=(6, 5))
        colors = np.where(labels == 1, "#e8a09a", "#9abfe8")
        ax.scatter(severity, areas, c=colors, alpha=0.75, s=50)
        valid = np.isfinite(severity)
        if valid.sum() > 1:
            m, b = np.polyfit(severity[valid], areas[valid], 1)
            xs = np.linspace(severity[valid].min(), severity[valid].max(), 100)
            ax.plot(xs, m * xs + b, "k--", lw=1.2, alpha=0.6)
        ax.set_xlabel("ADHD severity score")
        ax.set_ylabel(f"Area under β{dim} curve")
        ax.set_title(f"H{dim} area vs Severity (ρ={corr:.3f}, p={pvalue:.3f})")
        self.save_figure(fig, fname)

    # ------------------------------------------------------------------
    # Network-specific
    # ------------------------------------------------------------------

    def plot_network_betti_curves(
        self,
        network_results: Dict[str, dict],
        dim: int = 1,
        filename: str = "network_betti_curves.png",
    ) -> None:
        """One subplot per network, β_dim curves for ADHD vs Control."""
        n = len(network_results)
        if n == 0:
            return
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)
        if n == 1:
            axes = [axes]
        key = f"betti_{dim}"
        colors = {"adhd": "#e8a09a", "control": "#9abfe8"}

        for ax, (net_name, group_curves) in zip(axes, network_results.items()):
            eps = group_curves.get("epsilon", np.array([]))
            for group in ("adhd", "control"):
                g = group_curves.get(group, {})
                curve = g.get(key)
                if curve is None:
                    continue
                ax.plot(eps, curve, color=colors[group],
                        label=group.capitalize())
                std = g.get(f"{key}_std")
                if std is not None:
                    ax.fill_between(eps, curve - std, curve + std,
                                    color=colors[group], alpha=0.2)
            ax.set_title(net_name)
            ax.set_xlabel("Filtration epsilon")
            ax.set_ylabel(f"b{dim}")
            ax.legend(fontsize=8)
        fig.suptitle(f"Network-Specific b{dim} Curves: ADHD vs Control")
        fig.tight_layout()
        self.save_figure(fig, filename)

    # ------------------------------------------------------------------
    # FDA results
    # ------------------------------------------------------------------

    def plot_fda_results(
        self,
        fd,
        labels: np.ndarray,
        test_results: dict,
        fpca_results: dict,
        filename: str = "fda_results.png",
    ) -> None:
        """FDA summary: mean curves + FPCA scores."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Left: mean functional curves per group
        ax = axes[0]
        eps = fd.grid_points[0]
        adhd_mask = labels == 1
        adhd_mean = fd[adhd_mask].data_matrix[:, :, 0].mean(axis=0)
        ctrl_mean = fd[~adhd_mask].data_matrix[:, :, 0].mean(axis=0)
        ax.plot(eps, adhd_mean, color="#e8a09a", lw=2, label="ADHD")
        ax.plot(eps, ctrl_mean, color="#9abfe8", lw=2, label="Control")
        pval = test_results.get("pvalue", float("nan"))
        ax.set_title(f"Functional Group Means (p={pval:.3f})")
        ax.set_xlabel("Filtration ε")
        ax.set_ylabel("Betti curve value")
        ax.legend()

        # Right: FPCA scores (PC1 vs PC2)
        ax = axes[1]
        scores = fpca_results.get("scores")
        if scores is not None and scores.shape[1] >= 2:
            colors = np.where(labels == 1, "#e8a09a", "#9abfe8")
            ax.scatter(scores[:, 0], scores[:, 1], c=colors, alpha=0.75, s=50)
            evr = fpca_results.get("explained_variance_ratio", [0, 0])
            ax.set_xlabel(f"FPC1 ({evr[0]:.1%})")
            ax.set_ylabel(f"FPC2 ({evr[1]:.1%})")
        ax.set_title("Functional PCA Scores")

        fig.tight_layout()
        self.save_figure(fig, filename)
