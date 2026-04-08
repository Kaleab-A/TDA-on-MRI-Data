"""
Code/Idea1_PH_FC/run_idea1.py
ENTRY POINT for Idea 1 — Persistent Homology on FC matrices.

Run from project root:
    python -m Code.Idea1_PH_FC.run_idea1
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np

# Ensure project root is on the path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from Core.base_experiment import BaseIdeaOrchestrator
from Core.base_loader import ADHDDataLoader
from Core.base_masker import ROIMasker
from Core.utils import SubjectRecord
from Parameters.params_idea1 import Idea1Params
from Code.Idea1_PH_FC.fc_matrix_builder import FCMatrixBuilder
from Code.Idea1_PH_FC.persistence_computer import PHFCComputer
from Code.Idea1_PH_FC.distance_analyzer import PersistenceDistanceAnalyzer
from Code.Idea1_PH_FC.idea1_visualizer import Idea1Visualizer


class Idea1Orchestrator(BaseIdeaOrchestrator):
    """Orchestrates all Idea 1 experiments."""

    def __init__(self, params: Idea1Params):
        super().__init__(params, n_subjects=params.n_subjects, idea_name="Idea1")
        self.fc_builder = FCMatrixBuilder(params)
        self.ph_computer = PHFCComputer(params)
        self.analyzer = PersistenceDistanceAnalyzer(params, self.output_manager)
        self.visualizer = Idea1Visualizer(self.output_manager)

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def run_all_experiments(self, records: List[SubjectRecord]) -> None:
        print("\n=== Idea 1: Persistent Homology on FC Matrices ===")

        # 1. Build FC + distance matrices
        print("\n[Step 1] Building FC and distance matrices...")
        records = self.fc_builder.transform(records)
        # Keep only valid records
        records = [r for r in records if r.distance_matrix is not None]

        dist_matrices = [r.distance_matrix for r in records]
        labels = self.loader.get_labels_array(records)

        # 2. Compute persistence
        print("\n[Step 2] Computing persistent homology...")
        diagrams = self.ph_computer.fit_transform(dist_matrices)

        dims = list(range(self.params.max_dimension + 1))
        diagrams_per_dim = {
            dim: self.ph_computer.get_diagram_for_dimension(diagrams, dim)
            for dim in dims
        }

        # Plot example FC + distance matrices (first subject)
        self.visualizer.plot_fc_matrix(records[0].fc_matrix, records[0].subject_id)
        self.visualizer.plot_distance_matrix(records[0].distance_matrix,
                                             records[0].subject_id)

        # Plot group-level persistence diagrams
        adhd_mask = labels == 1
        for dim in dims:
            self.visualizer.plot_group_diagrams(
                diagrams_adhd=[diagrams_per_dim[dim][i]
                                for i in range(len(records)) if adhd_mask[i]],
                diagrams_ctrl=[diagrams_per_dim[dim][i]
                                for i in range(len(records)) if not adhd_mask[i]],
                dim=dim,
            )

        # 3. Experiments
        if self.params.run_group_comparison:
            print("\n[Exp 1] Group comparison (ADHD vs Control)...")
            results = self.analyzer.group_comparison_experiment(diagrams_per_dim, labels)
            self.visualizer.plot_total_persistence_comparison(results, dims=dims)

            # Wasserstein heatmap (H1)
            if 1 in diagrams_per_dim:
                W = self.analyzer.compute_group_wasserstein_matrix(
                    diagrams_per_dim[1], labels)
                self.visualizer.plot_wasserstein_heatmap(W, labels)

        if self.params.run_h0_vs_h1:
            print("\n[Exp 2] H0 vs H1 comparison...")
            self.analyzer.h0_vs_h1_experiment(diagrams_per_dim)

        if self.params.run_subtype_analysis and 1 in diagrams_per_dim:
            print("\n[Exp 3] Subtype analysis (H1 clustering)...")
            result = self.analyzer.subtype_analysis_experiment(
                diagrams_per_dim[1], records)
            tp_vals = np.array([
                PHFCComputer.total_persistence(PHFCComputer, d)
                for d in diagrams_per_dim[1]
            ])
            self.visualizer.plot_subtype_clusters(
                tp_vals, result["cluster_labels"], labels)

        if self.params.run_atlas_scale:
            print("\n[Exp 4] Atlas scale dependence...")
            self._run_atlas_scale_experiment()

        print("\n=== Idea 1 complete. Outputs saved to Output/Idea1/ ===")

    # ------------------------------------------------------------------
    # Atlas scale experiment
    # ------------------------------------------------------------------

    def _run_atlas_scale_experiment(self) -> None:
        """Re-run PH with each atlas and compare total persistence distributions."""
        from Core.base_tda import BasePersistenceComputer as BPC
        results_per_atlas: dict = {}

        for atlas_name in self.params.atlas_names_sweep:
            print(f"  Atlas: {atlas_name}")
            try:
                records_atlas = self.load_and_mask(atlas_name=atlas_name)
                records_atlas = self.fc_builder.transform(records_atlas)
                records_atlas = [r for r in records_atlas if r.distance_matrix is not None]

                # Use same params but fresh computer
                ph = PHFCComputer(self.params)
                dist_mats = [r.distance_matrix for r in records_atlas]
                diagrams = ph.fit_transform(dist_mats)

                per_dim: dict = {}
                for dim in range(self.params.max_dimension + 1):
                    dgms = ph.get_diagram_for_dimension(diagrams, dim)
                    mean_tp = float(np.mean([BPC.total_persistence(d) for d in dgms]))
                    per_dim[dim] = mean_tp
                results_per_atlas[atlas_name] = per_dim
            except Exception as exc:
                print(f"    WARNING — atlas '{atlas_name}' failed: {exc}")

        if results_per_atlas:
            self.visualizer.plot_atlas_scale_comparison(results_per_atlas)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    params = Idea1Params()
    orchestrator = Idea1Orchestrator(params)
    orchestrator.execute()
