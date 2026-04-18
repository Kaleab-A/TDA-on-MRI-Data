"""
Code/Idea3_SlidingWindow/idea3_visualizer.py
Visualizations for Idea 3 — Sliding-window TDA.
"""

from __future__ import annotations

from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from Core.base_visualizer import BaseVisualizer
from Core.utils import OutputManager


class Idea3Visualizer(BaseVisualizer):

    def __init__(self, output_manager: OutputManager):
        super().__init__(output_manager)

    def plot_temporal_h1(
        self, results: dict, filename: str = "temporal_h1_evolution.png"
    ) -> None:
        """Mean loop score over time ± std for ADHD vs Control."""
        fig, ax = plt.subplots(figsize=(10, 4))
        adhd_mean = results.get("adhd_mean")
        adhd_std = results.get("adhd_std")
        ctrl_mean = results.get("ctrl_mean")
        ctrl_std = results.get("ctrl_std")

        if adhd_mean is not None and len(adhd_mean) > 0:
            t = np.arange(len(adhd_mean))
            ax.plot(t, adhd_mean, color="#e8a09a", label="ADHD")
            ax.fill_between(t, adhd_mean - adhd_std, adhd_mean + adhd_std,
                            color="#e8a09a", alpha=0.25)
        if ctrl_mean is not None and len(ctrl_mean) > 0:
            t = np.arange(len(ctrl_mean))
            ax.plot(t, ctrl_mean, color="#9abfe8", label="Control")
            ax.fill_between(t, ctrl_mean - ctrl_std, ctrl_mean + ctrl_std,
                            color="#9abfe8", alpha=0.25)
        ax.set_xlabel("Window index")
        ax.set_ylabel("Loop score (H1 total persistence)")
        ax.set_title("Temporal H1 Evolution: ADHD vs Control")
        ax.legend()
        fig.tight_layout()
        self.save_figure(fig, filename)

    def plot_loop_score_comparison(
        self, results: dict, filename: str = "loop_score_comparison.png"
    ) -> None:
        """Multi-panel group comparison for loop score metrics."""
        metrics = list(results["results"].keys())
        n = len(metrics)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
        if n == 1:
            axes = [axes]
        for ax, metric in zip(axes, metrics):
            adhd_vals = results["results"][metric]["adhd"]
            ctrl_vals = results["results"][metric]["ctrl"]
            pvalue = results["results"][metric]["test"]["pvalue"]
            self.plot_group_comparison(adhd_vals, ctrl_vals, metric,
                                       ax=ax, pvalue=pvalue)
        fig.suptitle("Loop Score Biomarkers: ADHD vs Control", y=1.02)
        fig.tight_layout()
        self.save_figure(fig, filename)

    def plot_window_size_sweep(
        self, results: dict, filename: str = "window_size_sweep.png"
    ) -> None:
        """Mean loop score vs window length for both groups."""
        df = results["dataframe"]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(df["window_length"], df["adhd_mean"],
                     "o-", color="#e8a09a", label="ADHD")
        axes[0].plot(df["window_length"], df["ctrl_mean"],
                     "s-", color="#9abfe8", label="Control")
        axes[0].set_xlabel("Window length (TRs)")
        axes[0].set_ylabel("Mean loop score")
        axes[0].set_title("Loop Score vs Window Size")
        axes[0].legend()

        axes[1].plot(df["window_length"], df["pvalue"], "ko-")
        axes[1].axhline(0.05, ls="--", color="red", label="p=0.05")
        axes[1].set_xlabel("Window length (TRs)")
        axes[1].set_ylabel("Mann-Whitney p-value")
        axes[1].set_title("Statistical Significance vs Window Size")
        axes[1].legend()
        fig.tight_layout()
        self.save_figure(fig, filename)

    def plot_autocorrelation(
        self,
        results: dict,
        labels: np.ndarray,
        filename: str = "loop_score_autocorrelation.png",
    ) -> None:
        """Plot group-mean ACF of loop score sequences."""
        fig, ax = plt.subplots(figsize=(8, 4))
        adhd_acfs = results.get("adhd_acfs", [])
        ctrl_acfs = results.get("ctrl_acfs", [])

        if adhd_acfs:
            adhd_arr = np.vstack(adhd_acfs)
            lags = np.arange(adhd_arr.shape[1])
            ax.plot(lags, adhd_arr.mean(axis=0), color="#e8a09a", label="ADHD")
            ax.fill_between(lags,
                            adhd_arr.mean(axis=0) - adhd_arr.std(axis=0),
                            adhd_arr.mean(axis=0) + adhd_arr.std(axis=0),
                            color="#e8a09a", alpha=0.2)
        if ctrl_acfs:
            ctrl_arr = np.vstack(ctrl_acfs)
            lags = np.arange(ctrl_arr.shape[1])
            ax.plot(lags, ctrl_arr.mean(axis=0), color="#9abfe8", label="Control")
            ax.fill_between(lags,
                            ctrl_arr.mean(axis=0) - ctrl_arr.std(axis=0),
                            ctrl_arr.mean(axis=0) + ctrl_arr.std(axis=0),
                            color="#9abfe8", alpha=0.2)
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.set_xlabel("Lag")
        ax.set_ylabel("Autocorrelation")
        ax.set_title("Loop Score Autocorrelation: ADHD vs Control")
        ax.legend()
        if "ks_pvalue" in results:
            ax.text(0.98, 0.95,
                    f"KS p={results['ks_pvalue']:.3f}",
                    transform=ax.transAxes, ha="right", va="top")
        fig.tight_layout()
        self.save_figure(fig, filename)
