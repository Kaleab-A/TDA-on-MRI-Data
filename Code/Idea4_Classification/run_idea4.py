"""
Code/Idea4_Classification/run_idea4.py
ENTRY POINT for Idea 4 — Topological feature vectors for classification.

Run from project root:
    python -m Code.Idea4_Classification.run_idea4
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from Core.base_experiment import BaseIdeaOrchestrator
from Core.utils import SubjectRecord
from Parameters.params_idea4 import Idea4Params
from Parameters.params_idea1 import Idea1Params
from Code.Idea1_PH_FC.fc_matrix_builder import FCMatrixBuilder
from Code.Idea1_PH_FC.persistence_computer import PHFCComputer
from Code.Idea4_Classification.feature_extractor import TopologicalFeatureExtractor
from Code.Idea4_Classification.classifier_pipeline import ClassifierPipeline
from Code.Idea4_Classification.permutation_tester import PermutationTester
from Code.Idea4_Classification.idea4_visualizer import Idea4Visualizer


class Idea4Orchestrator(BaseIdeaOrchestrator):

    def __init__(self, params: Idea4Params):
        super().__init__(params, n_subjects=params.n_subjects, idea_name="Idea4",
                         dataset_name=params.dataset_name)
        # Reuse Idea1 infrastructure with matching TDA params
        self._idea1_params = Idea1Params(
            atlas_name=params.atlas_name,
            max_dimension=params.max_dimension,
            max_edge_length=params.max_edge_length,
        )
        self.fc_builder = FCMatrixBuilder(self._idea1_params)
        self.ph_computer = PHFCComputer(self._idea1_params)
        self.feature_extractor = TopologicalFeatureExtractor(params)
        self.classifier_pipeline = ClassifierPipeline(params)
        self.permutation_tester = PermutationTester(params, self.output_manager)
        self.visualizer = Idea4Visualizer(self.output_manager)

    def run_all_experiments(self, records: List[SubjectRecord]) -> None:
        print("\n=== Idea 4: Topological Feature Vectors for Classification ===")

        records = [r for r in records if r.time_series is not None]
        labels = self.loader.get_labels_array(records)

        print("\n[Step 1] Building FC and distance matrices...")
        records = self.fc_builder.transform(records)
        records = [r for r in records if r.distance_matrix is not None]
        labels = self.loader.get_labels_array(records)

        print("\n[Step 2] Computing persistent homology...")
        dist_matrices = [r.distance_matrix for r in records]
        diagrams = self.ph_computer.fit_transform(dist_matrices)

        diagrams_per_dim = {
            dim: self.ph_computer.get_diagram_for_dimension(diagrams, dim)
            for dim in self.params.homology_dimensions
        }

        print("\n[Step 3] Extracting topological features (fit on all data)...")
        self.feature_extractor.fit(diagrams_per_dim)
        X_tda = self.feature_extractor.transform_all(diagrams_per_dim)
        X_fc = self.feature_extractor.get_fc_features(records)
        X_combined = np.hstack([X_tda, X_fc])
        print(f"  TDA features: {X_tda.shape}, FC features: {X_fc.shape}, "
              f"Combined: {X_combined.shape}")

        feature_sets: dict = {}
        if self.params.run_tda_only:
            feature_sets["tda"] = X_tda
        if self.params.run_fc_only:
            feature_sets["fc"] = X_fc
        if self.params.run_combined:
            feature_sets["combined"] = X_combined

        print("\n[Exp 1] Classification comparison (TDA vs FC vs Combined)...")
        comparison_df = self.classifier_pipeline.compare_feature_sets(
            feature_sets, labels)
        if not comparison_df.empty:
            self.output_manager.save_dataframe(
                comparison_df, "classification_comparison.csv")
            self.visualizer.plot_classification_comparison(comparison_df)
            self.visualizer.plot_cv_scores_heatmap(comparison_df)

        if self.params.run_permutation_test:
            print("\n[Exp 2] Permutation tests...")
            perm_df = self.permutation_tester.run_all_permutation_tests(
                feature_sets, labels)

        if self.params.run_feature_importance and X_tda.shape[1] > 0:
            print("\n[Exp 3] Feature importances (TDA)...")
            # Build feature names: {total_H0, total_H1, ..., entropy_H0, ...}
            feature_names = []
            for dim in sorted(diagrams_per_dim.keys()):
                feature_names.append(f"total_persistence_H{dim}")
            for dim in sorted(diagrams_per_dim.keys()):
                feature_names.append(f"entropy_H{dim}")
            # Landscapes add many columns — label generically
            n_extra = X_tda.shape[1] - len(feature_names)
            feature_names += [f"landscape_feat_{i}" for i in range(max(0, n_extra))]
            feature_names = feature_names[:X_tda.shape[1]]

            importance_df = self.classifier_pipeline.get_feature_importances(
                X_tda, labels, feature_names=feature_names)
            self.output_manager.save_dataframe(
                importance_df, "feature_importances.csv")
            self.visualizer.plot_feature_importances(importance_df)

        print("\n=== Idea 4 complete. Outputs saved to Output/Idea4/ ===")


if __name__ == "__main__":
    params = Idea4Params()
    orchestrator = Idea4Orchestrator(params)
    orchestrator.execute()
