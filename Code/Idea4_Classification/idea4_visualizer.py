"""
Code/Idea4_Classification/idea4_visualizer.py
Visualizations for Idea 4 — Classification with topological features.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from Core.base_visualizer import BaseVisualizer
from Core.utils import OutputManager


class Idea4Visualizer(BaseVisualizer):

    def __init__(self, output_manager: OutputManager):
        super().__init__(output_manager)

    def plot_classification_comparison(
        self,
        comparison_df: pd.DataFrame,
        metric: str = "test_accuracy_mean",
        filename: str = "classification_comparison.png",
    ) -> None:
        """Grouped bar chart: classifier × feature_set."""
        fig, ax = plt.subplots(figsize=(10, 5))
        feature_sets = comparison_df["feature_set"].unique()
        classifiers = comparison_df["classifier"].unique()
        x = np.arange(len(classifiers))
        width = 0.25
        colors = ["#4878CF", "#6ACC65", "#D65F5F"]

        for i, fs in enumerate(feature_sets):
            sub = comparison_df[comparison_df["feature_set"] == fs]
            sub = sub.set_index("classifier").reindex(classifiers)
            means = sub[metric].values
            stds = sub[f"{metric.replace('_mean', '_std')}"].values
            ax.bar(x + i * width, means, width,
                   yerr=stds, label=fs, color=colors[i % len(colors)],
                   capsize=4, alpha=0.85)

        ax.set_xticks(x + width)
        ax.set_xticklabels(classifiers, rotation=15)
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title("Classification Performance: TDA vs FC vs Combined")
        ax.legend(title="Feature set")
        ax.axhline(0.5, color="gray", ls="--", lw=0.8, label="Chance")
        fig.tight_layout()
        self.save_figure(fig, filename)

    def plot_permutation_results(
        self,
        perm_results: dict,
        title_prefix: str = "",
        filename: str = "permutation_test.png",
    ) -> None:
        """
        Histogram of null distribution with observed score marked.
        perm_results: output of PermutationTester.test_classifier()
        """
        fig, ax = plt.subplots(figsize=(7, 4))
        null = perm_results["null_distribution"]
        true_score = perm_results["true_score"]
        pvalue = perm_results["pvalue"]
        ax.hist(null, bins=30, color="#9abfe8", edgecolor="white",
                alpha=0.8, label="Null distribution")
        ax.axvline(true_score, color="#e8a09a", lw=2.5,
                   label=f"Observed = {true_score:.3f}")
        ax.set_xlabel("Accuracy")
        ax.set_ylabel("Count")
        ax.set_title(f"{title_prefix} Permutation Test (p = {pvalue:.3f})")
        ax.legend()
        fig.tight_layout()
        self.save_figure(fig, filename)

    def plot_feature_importances(
        self,
        importance_df: pd.DataFrame,
        top_n: int = 20,
        filename: str = "feature_importances.png",
    ) -> None:
        """Horizontal bar chart of top-N feature importances."""
        top = importance_df.head(top_n)
        fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.3)))
        ax.barh(top["feature"][::-1], top["importance"][::-1], color="#4878CF")
        ax.set_xlabel("Feature importance")
        ax.set_title(f"Top {top_n} Feature Importances (Random Forest)")
        fig.tight_layout()
        self.save_figure(fig, filename)

    def plot_cv_scores_heatmap(
        self,
        comparison_df: pd.DataFrame,
        metric: str = "test_accuracy_mean",
        filename: str = "cv_heatmap.png",
    ) -> None:
        """Heatmap: classifier × feature_set → metric."""
        pivot = comparison_df.pivot(
            index="classifier", columns="feature_set", values=metric
        )
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGn", ax=ax,
                    vmin=0.4, vmax=1.0, linewidths=0.5)
        ax.set_title(metric.replace("_", " ").title())
        fig.tight_layout()
        self.save_figure(fig, filename)
