"""
Code/Idea3_SlidingWindow/run_idea3.py
ENTRY POINT for Idea 3 — Sliding-window TDA on BOLD time series.

Run from project root:
    python -m Code.Idea3_SlidingWindow.run_idea3
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from Core.base_experiment import BaseIdeaOrchestrator
from Core.utils import SubjectRecord
from Parameters.params_idea3 import Idea3Params
from Code.Idea3_SlidingWindow.window_embedder import SlidingWindowEmbedder
from Code.Idea3_SlidingWindow.window_ph_computer import WindowPHComputer
from Code.Idea3_SlidingWindow.loop_score_analyzer import LoopScoreAnalyzer
from Code.Idea3_SlidingWindow.idea3_visualizer import Idea3Visualizer


class Idea3Orchestrator(BaseIdeaOrchestrator):

    def __init__(self, params: Idea3Params):
        super().__init__(params, n_subjects=params.n_subjects, idea_name="Idea3",
                         dataset_name=params.dataset_name)
        self.embedder = SlidingWindowEmbedder(params)
        self.ph_computer = WindowPHComputer(params)
        self.analyzer = LoopScoreAnalyzer(params, self.output_manager)
        self.visualizer = Idea3Visualizer(self.output_manager)

    def run_all_experiments(self, records: List[SubjectRecord]) -> None:
        print("\n=== Idea 3: Sliding-Window TDA on BOLD Time Series ===")

        records = [r for r in records if r.time_series is not None]
        labels = self.loader.get_labels_array(records)

        print(f"\n[Step 1] Embedding BOLD signals (window={self.params.window_length} TRs)...")
        all_windows = self.embedder.embed_all(records)
        n_valid = sum(1 for w in all_windows if len(w) > 0)
        print(f"  {n_valid}/{len(records)} subjects embedded successfully.")

        print("\n[Step 2] Computing per-window persistent homology...")
        all_diagrams = self.ph_computer.fit_transform_all_subjects(all_windows)

        print("\n[Step 3] Computing loop scores...")
        loop_scores = [
            self.analyzer.compute_loop_score(d) if d else np.array([])
            for d in all_diagrams
        ]

        if self.params.run_temporal_h1:
            print("\n[Exp 1] Temporal H1 evolution...")
            results = self.analyzer.temporal_h1_evolution_experiment(
                loop_scores, labels)
            self.visualizer.plot_temporal_h1(results)

        if self.params.run_loop_score_biomarker:
            print("\n[Exp 2] Loop score as biomarker...")
            results = self.analyzer.loop_score_as_biomarker_experiment(
                loop_scores, labels)
            self.visualizer.plot_loop_score_comparison(results)

        if self.params.run_window_size_sweep:
            print("\n[Exp 3] Window size sweep...")
            results = self.analyzer.window_size_sweep_experiment(
                records, self.params.window_lengths_sweep)
            self.visualizer.plot_window_size_sweep(results)

        if self.params.run_autocorrelation:
            print("\n[Exp 4] Autocorrelation of loop scores...")
            results = self.analyzer.autocorrelation_experiment(
                loop_scores, labels)
            self.visualizer.plot_autocorrelation(results, labels)

        print("\n=== Idea 3 complete. Outputs saved to Output/Idea3/ ===")


if __name__ == "__main__":
    params = Idea3Params()
    orchestrator = Idea3Orchestrator(params)
    orchestrator.execute()
