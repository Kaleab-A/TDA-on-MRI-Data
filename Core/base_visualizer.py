"""
Core/base_visualizer.py
Shared matplotlib/seaborn setup with reusable plot methods.
"""

from __future__ import annotations

from typing import List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend (safe for scripts)
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from Core.utils import OutputManager


class BaseVisualizer:
    """
    Base class for all Idea-specific visualizers.

    Provides:
    - Consistent style configuration
    - save_figure() routing through OutputManager
    - Common persistence diagram and group-comparison plots
    """

    def __init__(
        self,
        output_manager: OutputManager,
        style: str = "seaborn-v0_8-whitegrid",
        palette: str = "Set2",
        figure_dpi: int = 150,
    ):
        self.output_manager = output_manager
        self.style = style
        self.palette = palette
        self.figure_dpi = figure_dpi
        self._setup_style()

    # ------------------------------------------------------------------
    # Style
    # ------------------------------------------------------------------

    def _setup_style(self) -> None:
        try:
            plt.style.use(self.style)
        except OSError:
            plt.style.use("seaborn-v0_8-whitegrid")
        sns.set_palette(self.palette)
        plt.rcParams.update({
            "figure.dpi": self.figure_dpi,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
        })

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def save_figure(self, fig: plt.Figure, filename: str,
                    timestamp: bool = False) -> None:
        path = self.output_manager.get_plot_path(filename, timestamp=timestamp)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")

    # ------------------------------------------------------------------
    # Common plots
    # ------------------------------------------------------------------

    def plot_persistence_diagram(
        self,
        diagram: np.ndarray,
        dim: int,
        title: str = "",
        ax: Optional[plt.Axes] = None,
        color: str = "steelblue",
        alpha: float = 0.6,
    ) -> plt.Axes:
        """Scatter (birth, death) with diagonal line."""
        if ax is None:
            _, ax = plt.subplots(figsize=(5, 5))
        if diagram.size > 0:
            ax.scatter(diagram[:, 0], diagram[:, 1],
                       c=color, alpha=alpha, s=20, label=f"H{dim}")
            max_val = np.max(diagram) * 1.05
            ax.plot([0, max_val], [0, max_val], "k--", lw=0.8)
            ax.set_xlim(0, max_val)
            ax.set_ylim(0, max_val)
        ax.set_xlabel("Birth")
        ax.set_ylabel("Death")
        ax.set_title(title or f"Persistence Diagram H{dim}")
        return ax

    def plot_barcode(
        self,
        diagram: np.ndarray,
        ax: Optional[plt.Axes] = None,
        color: str = "steelblue",
    ) -> plt.Axes:
        """Horizontal bars for each [birth, death] interval."""
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 4))
        for i, (birth, death) in enumerate(diagram):
            ax.plot([birth, death], [i, i], lw=2, color=color, alpha=0.7)
        ax.set_xlabel("Filtration value")
        ax.set_ylabel("Bar index")
        ax.set_title("Barcode")
        return ax

    def plot_group_comparison(
        self,
        adhd_values: np.ndarray,
        control_values: np.ndarray,
        metric_name: str,
        ax: Optional[plt.Axes] = None,
        pvalue: Optional[float] = None,
    ) -> plt.Axes:
        """Boxplot + stripplot for ADHD vs control groups."""
        import pandas as pd
        if ax is None:
            _, ax = plt.subplots(figsize=(5, 4))
        df = pd.DataFrame({
            metric_name: np.concatenate([adhd_values, control_values]),
            "Group": ["ADHD"] * len(adhd_values) + ["Control"] * len(control_values),
        })
        sns.boxplot(data=df, x="Group", y=metric_name, ax=ax,
                    palette=["#e8a09a", "#9abfe8"], width=0.5)
        sns.stripplot(data=df, x="Group", y=metric_name, ax=ax,
                      color="black", alpha=0.5, size=4, jitter=True)
        title = metric_name
        if pvalue is not None:
            title += f"  (p = {pvalue:.3f})"
        ax.set_title(title)
        return ax

    def plot_heatmap(
        self,
        matrix: np.ndarray,
        title: str = "",
        ax: Optional[plt.Axes] = None,
        cmap: str = "RdBu_r",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> plt.Axes:
        """Generic heatmap."""
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax)
        ax.set_title(title)
        return ax
