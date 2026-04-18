"""
Code/Idea2_Mapper/mapper_analyzer.py
Statistical comparison of Mapper graph topology between groups.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from Core.utils import OutputManager, StatisticsHelper
from Parameters.params_idea2 import Idea2Params


class MapperAnalyzer:
    """Compares Mapper graph statistics between ADHD and control groups."""

    def __init__(self, params: Idea2Params, output_manager: OutputManager):
        self.params = params
        self.output_manager = output_manager

    # ------------------------------------------------------------------
    # Experiment 1: Group topology comparison
    # ------------------------------------------------------------------

    def topology_comparison_experiment(
        self,
        graphs_adhd: List[dict],
        graphs_ctrl: List[dict],
        stats_list: List[str] = ("n_nodes", "n_edges", "n_components",
                                  "mean_node_size"),
    ) -> dict:
        """
        Compare graph statistics between ADHD and control groups.
        Returns dict with per-statistic Mann-Whitney results and saves CSV.
        """
        rows = []
        results = {}
        for stat in stats_list:
            adhd_vals = np.array([g[stat] for g in graphs_adhd], dtype=float)
            ctrl_vals = np.array([g[stat] for g in graphs_ctrl], dtype=float)
            test = StatisticsHelper.mann_whitney_u(adhd_vals, ctrl_vals)
            test["statistic_name"] = stat
            test["adhd_mean"] = float(np.mean(adhd_vals))
            test["ctrl_mean"] = float(np.mean(ctrl_vals))
            rows.append(test)
            results[stat] = {"adhd": adhd_vals, "ctrl": ctrl_vals, "test": test}
            print(f"  {stat}: ADHD={np.mean(adhd_vals):.2f}, "
                  f"Ctrl={np.mean(ctrl_vals):.2f}, p={test['pvalue']:.4f}")
        df = pd.DataFrame(rows)
        self.output_manager.save_dataframe(df, "mapper_group_comparison.csv")
        return results

    # ------------------------------------------------------------------
    # Experiment 2: Parameter stability
    # ------------------------------------------------------------------

    def parameter_stability_experiment(
        self,
        time_series: np.ndarray,
        n_intervals_range: List[int],
        overlap_range: List[float],
        stat_name: str = "n_nodes",
    ) -> dict:
        """
        Grid search over (n_intervals, overlap_fraction) for one subject.
        Returns {(n_intervals, overlap): graph_stats_dict}.
        """
        from Code.Idea2_Mapper.mapper_builder import MapperBuilder
        builder = MapperBuilder(self.params)

        results = {}
        rows = []
        for n_int in n_intervals_range:
            for overlap in overlap_range:
                graph = builder.build_for_subject(
                    time_series,
                    n_intervals=n_int,
                    overlap_fraction=overlap,
                )
                results[(n_int, overlap)] = graph
                rows.append({
                    "n_intervals": n_int,
                    "overlap_fraction": overlap,
                    "n_nodes": graph["n_nodes"],
                    "n_edges": graph["n_edges"],
                })

        df = pd.DataFrame(rows)
        self.output_manager.save_dataframe(df, "parameter_stability.csv")
        return {"results": results, "dataframe": df}
