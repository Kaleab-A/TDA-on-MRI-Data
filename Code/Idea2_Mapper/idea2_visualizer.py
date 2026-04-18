"""
Code/Idea2_Mapper/idea2_visualizer.py
Visualizations for Idea 2 — Mapper on brain state space.
"""

from __future__ import annotations

from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

from Core.base_visualizer import BaseVisualizer
from Core.utils import OutputManager


class Idea2Visualizer(BaseVisualizer):
    """All plots specific to Idea 2."""

    def __init__(self, output_manager: OutputManager):
        super().__init__(output_manager)

    # ------------------------------------------------------------------
    # Mapper graph
    # ------------------------------------------------------------------

    def plot_mapper_graph(
        self,
        mapper_graph: dict,
        node_color_values: np.ndarray,
        title: str = "Mapper Graph",
        filename: str = "mapper_graph.png",
    ) -> None:
        """
        Draw the Mapper graph with nodes colored by a scalar value.

        Parameters
        ----------
        node_color_values : np.ndarray, shape (n_nodes,)
            One value per node (e.g. mean lens value or ADHD fraction).
        """
        nodes = mapper_graph["nodes"]
        edges = mapper_graph["edges"]
        node_ids = list(nodes.keys())
        node_sizes = [max(50, len(nodes[n]) * 5) for n in node_ids]

        G = nx.Graph()
        G.add_nodes_from(node_ids)
        G.add_edges_from(edges)

        pos = nx.kamada_kawai_layout(G) if len(node_ids) > 1 else {node_ids[0]: (0, 0)}

        fig, ax = plt.subplots(figsize=(8, 7))
        node_color = node_color_values if node_color_values is not None else "steelblue"
        nx.draw_networkx_nodes(G, pos, nodelist=node_ids,
                               node_color=node_color, node_size=node_sizes,
                               cmap=plt.cm.RdBu_r, ax=ax, alpha=0.85)
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.4, width=1.2)

        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdBu_r,
                                    norm=plt.Normalize(
                                        vmin=node_color_values.min(),
                                        vmax=node_color_values.max()
                                    ) if hasattr(node_color_values, "min") else None)
        sm.set_array([])
        if hasattr(node_color_values, "min"):
            plt.colorbar(sm, ax=ax, shrink=0.6, label="Node color value")
        ax.set_title(title)
        ax.axis("off")
        self.save_figure(fig, filename)

    def plot_population_mapper(
        self, graph: dict, filename: str = "population_mapper.png"
    ) -> None:
        """Color nodes by ADHD fraction (red=ADHD, blue=Control)."""
        node_ids = list(graph["nodes"].keys())
        adhd_fracs = np.array([
            graph["node_adhd_fraction"].get(n, 0.0) for n in node_ids
        ])
        self.plot_mapper_graph(
            graph, node_color_values=adhd_fracs,
            title="Population Mapper (Red = ADHD, Blue = Control)",
            filename=filename,
        )

    # ------------------------------------------------------------------
    # Group topology comparison
    # ------------------------------------------------------------------

    def plot_graph_statistics_comparison(
        self,
        results: dict,
        filename: str = "mapper_group_stats.png",
    ) -> None:
        """Multi-panel boxplots for each graph statistic."""
        stat_names = list(results.keys())
        n = len(stat_names)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
        if n == 1:
            axes = [axes]
        for ax, stat in zip(axes, stat_names):
            adhd_vals = results[stat]["adhd"]
            ctrl_vals = results[stat]["ctrl"]
            pvalue = results[stat]["test"]["pvalue"]
            self.plot_group_comparison(adhd_vals, ctrl_vals, stat,
                                       ax=ax, pvalue=pvalue)
        fig.suptitle("Mapper Graph Statistics: ADHD vs Control", y=1.02)
        fig.tight_layout()
        self.save_figure(fig, filename)

    # ------------------------------------------------------------------
    # Parameter stability
    # ------------------------------------------------------------------

    def plot_parameter_stability_heatmap(
        self,
        stability_df,
        stat_name: str = "n_nodes",
        filename: str = "parameter_stability.png",
    ) -> None:
        """Heatmap: n_intervals × overlap_fraction → stat value."""
        import pandas as pd
        pivot = stability_df.pivot(
            index="n_intervals", columns="overlap_fraction", values=stat_name
        )
        fig, ax = plt.subplots(figsize=(7, 5))
        import seaborn as sns
        sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax,
                    linewidths=0.5)
        ax.set_title(f"Parameter Stability — {stat_name}")
        ax.set_xlabel("Overlap fraction")
        ax.set_ylabel("Number of intervals")
        fig.tight_layout()
        self.save_figure(fig, filename)
