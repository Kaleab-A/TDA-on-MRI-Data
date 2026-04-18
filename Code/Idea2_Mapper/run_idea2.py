"""
Code/Idea2_Mapper/run_idea2.py
ENTRY POINT for Idea 2 — Mapper on brain state space.

Run from project root:
    python -m Code.Idea2_Mapper.run_idea2
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from Core.base_experiment import BaseIdeaOrchestrator
from Core.utils import SubjectRecord
from Parameters.params_idea2 import Idea2Params
from Code.Idea2_Mapper.mapper_builder import MapperBuilder
from Code.Idea2_Mapper.mapper_analyzer import MapperAnalyzer
from Code.Idea2_Mapper.idea2_visualizer import Idea2Visualizer


class Idea2Orchestrator(BaseIdeaOrchestrator):
    """Orchestrates all Idea 2 experiments."""

    def __init__(self, params: Idea2Params):
        super().__init__(params, n_subjects=params.n_subjects, idea_name="Idea2", dataset_name=params.dataset_name)
        self.mapper_builder = MapperBuilder(params)
        self.analyzer = MapperAnalyzer(params, self.output_manager)
        self.visualizer = Idea2Visualizer(self.output_manager)

    def run_all_experiments(self, records: List[SubjectRecord]) -> None:
        print("\n=== Idea 2: Mapper on Brain State Space ===")

        records = [r for r in records if r.time_series is not None]
        labels = self.loader.get_labels_array(records)
        adhd_mask = labels == 1

        # Build Mapper for each subject
        print("\n[Step 1] Building per-subject Mapper graphs...")
        all_graphs = []
        for i, rec in enumerate(records):
            print(f"  Subject {i+1}/{len(records)}: {rec.subject_id}")
            graph = self.mapper_builder.build_for_subject(rec.time_series)
            stats = self.mapper_builder.compute_graph_statistics(graph)
            graph.update(stats)   # merge stats into graph dict
            all_graphs.append(graph)

        # Plot example graph (first ADHD subject)
        adhd_indices = np.where(adhd_mask)[0]
        if len(adhd_indices) > 0:
            ex = adhd_indices[0]
            node_ids = list(all_graphs[ex]["nodes"].keys())
            lens_vals = all_graphs[ex]["lens_values"].flatten()
            # Mean lens value per node
            node_lens = np.array([
                np.mean(lens_vals[all_graphs[ex]["nodes"][n]]) for n in node_ids
            ])
            self.visualizer.plot_mapper_graph(
                all_graphs[ex], node_lens,
                title=f"Mapper — {records[ex].subject_id} (ADHD)",
                filename="example_mapper_adhd.png",
            )

        ctrl_indices = np.where(~adhd_mask)[0]
        if len(ctrl_indices) > 0:
            ex = ctrl_indices[0]
            node_ids = list(all_graphs[ex]["nodes"].keys())
            lens_vals = all_graphs[ex]["lens_values"].flatten()
            node_lens = np.array([
                np.mean(lens_vals[all_graphs[ex]["nodes"][n]]) for n in node_ids
            ])
            self.visualizer.plot_mapper_graph(
                all_graphs[ex], node_lens,
                title=f"Mapper — {records[ex].subject_id} (Control)",
                filename="example_mapper_control.png",
            )

        # Experiment 1: Group topology comparison
        if self.params.run_group_topology:
            print("\n[Exp 1] Group topology comparison...")
            graphs_adhd = [all_graphs[i] for i in range(len(records)) if adhd_mask[i]]
            graphs_ctrl = [all_graphs[i] for i in range(len(records)) if not adhd_mask[i]]
            results = self.analyzer.topology_comparison_experiment(
                graphs_adhd, graphs_ctrl)
            self.visualizer.plot_graph_statistics_comparison(results)

        # Experiment 2: Population-level Mapper
        if self.params.run_population_mapper:
            print("\n[Exp 2] Population Mapper...")
            pop_graph = self.mapper_builder.build_population_mapper(
                [r.time_series for r in records], labels)
            pop_stats = self.mapper_builder.compute_graph_statistics(pop_graph)
            print(f"  Population Mapper: {pop_stats}")
            self.visualizer.plot_population_mapper(pop_graph)

        # Experiment 3: Parameter stability
        if self.params.run_parameter_stability:
            print("\n[Exp 3] Parameter stability...")
            rep_subject = records[0]
            stability = self.analyzer.parameter_stability_experiment(
                rep_subject.time_series,
                n_intervals_range=self.params.n_intervals_range,
                overlap_range=self.params.overlap_range,
            )
            for stat in ("n_nodes", "n_edges"):
                self.visualizer.plot_parameter_stability_heatmap(
                    stability["dataframe"], stat_name=stat,
                    filename=f"stability_{stat}.png",
                )

        print("\n=== Idea 2 complete. Outputs saved to Output/Idea2/ ===")


if __name__ == "__main__":
    params = Idea2Params()
    orchestrator = Idea2Orchestrator(params)
    orchestrator.execute()
