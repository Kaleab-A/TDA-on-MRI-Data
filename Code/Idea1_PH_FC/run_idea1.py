"""
Code/Idea1_PH_FC/run_idea1.py
ENTRY POINT for Idea 1 — Persistent Homology on FC matrices.

Run from project root:
    python -m Code.Idea1_PH_FC.run_idea1
"""

from __future__ import annotations

import pickle
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
        super().__init__(params, n_subjects=params.n_subjects, idea_name="Idea1",
                         dataset_name=params.dataset_name)
        self.fc_builder = FCMatrixBuilder(params)
        self.ph_computer = PHFCComputer(params)
        self.analyzer = PersistenceDistanceAnalyzer(params, self.output_manager)
        self.visualizer = Idea1Visualizer(self.output_manager,
                                          case_label=self.case_label)

    # ------------------------------------------------------------------
    # Caching helpers
    # ------------------------------------------------------------------

    def _diagrams_cache_path(self) -> Path:
        return self.output_manager.idea_dir / "diagrams_cache.pkl"

    def _load_diagrams_cache(self):
        """Return cached diagrams list or None if not found."""
        cache_path = self._diagrams_cache_path()
        if cache_path.exists():
            print(f"  Loading cached diagrams from {cache_path}")
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        return None

    def _save_diagrams_cache(self, diagrams) -> None:
        cache_path = self._diagrams_cache_path()
        with open(cache_path, "wb") as f:
            pickle.dump(diagrams, f)
        print(f"  Cached diagrams saved to {cache_path}")

    def _csv_exists(self, filename: str) -> bool:
        return self.output_manager.get_csv_path(filename).exists()

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

        # 2. Compute persistence (with caching)
        cached = self._load_diagrams_cache()
        if cached is not None and len(cached) == len(dist_matrices):
            print("\n[Step 2] Loaded persistent homology from cache (skipping ripser).")
            diagrams = cached
        else:
            print("\n[Step 2] Computing persistent homology...")
            diagrams = self.ph_computer.fit_transform(dist_matrices)
            self._save_diagrams_cache(diagrams)

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
            csv_name = "group_comparison_total_persistence.csv"
            if self._csv_exists(csv_name):
                print(f"\n[Exp 1] Skipping group comparison — {csv_name} already exists.")
                # Still need results dict for plots; recompute cheaply without saving
                results = self.analyzer.group_comparison_experiment(
                    diagrams_per_dim, labels)
            else:
                print("\n[Exp 1] Group comparison...")
                results = self.analyzer.group_comparison_experiment(
                    diagrams_per_dim, labels)
            self.visualizer.plot_total_persistence_comparison(results, dims=dims)

            # Wasserstein heatmap for every dimension
            for dim in dims:
                heatmap_file = f"wasserstein_heatmap_H{dim}.png"
                if self.output_manager.get_plot_path(heatmap_file).exists():
                    print(f"  Skipping Wasserstein heatmap H{dim} — already exists.")
                    continue
                if dim in diagrams_per_dim:
                    W = self.analyzer.compute_group_wasserstein_matrix(
                        diagrams_per_dim[dim], labels)
                    self.visualizer.plot_wasserstein_heatmap(W, labels, dim=dim)

        if self.params.run_h0_vs_h1:
            csv_name = "h0_h1_h2_summary.csv"
            if self._csv_exists(csv_name):
                print(f"\n[Exp 2] Skipping H0/H1/H2 summary — {csv_name} already exists.")
            else:
                print("\n[Exp 2] H0 vs H1 comparison...")
                self.analyzer.h0_vs_h1_experiment(diagrams_per_dim)

        if self.params.run_subtype_analysis:
            print("\n[Exp 3] Subtype analysis (clustering per dimension)...")
            for dim in dims:
                if dim not in diagrams_per_dim:
                    continue
                csv_name = f"subtype_clusters_H{dim}.csv"
                if self._csv_exists(csv_name):
                    print(f"  Skipping H{dim} clustering — {csv_name} already exists.")
                    # Still plot from existing data
                    tp_vals = np.array([
                        PHFCComputer.total_persistence(d)
                        for d in diagrams_per_dim[dim]
                    ])
                    import pandas as pd
                    df = pd.read_csv(self.output_manager.get_csv_path(csv_name))
                    cluster_labels = df["cluster"].values
                    self.visualizer.plot_subtype_clusters(
                        tp_vals, cluster_labels, labels, dim=dim)
                    continue
                print(f"  H{dim} clustering...")
                result = self.analyzer.subtype_analysis_experiment(
                    diagrams_per_dim[dim], records, dim=dim)
                tp_vals = np.array([
                    PHFCComputer.total_persistence(d)
                    for d in diagrams_per_dim[dim]
                ])
                self.visualizer.plot_subtype_clusters(
                    tp_vals, result["cluster_labels"], labels, dim=dim)

        if self.params.run_atlas_scale:
            print("\n[Exp 4] Atlas scale dependence...")
            self._run_atlas_scale_experiment()

        print(f"\n=== Idea 1 complete. Outputs saved to "
              f"{self.output_manager.idea_dir}/ ===")

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
