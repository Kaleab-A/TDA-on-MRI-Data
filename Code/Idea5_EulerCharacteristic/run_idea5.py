"""
Code/Idea5_EulerCharacteristic/run_idea5.py
ENTRY POINT for Idea 5 — Euler characteristic and Betti curves.

Run from project root:
    python -m Code.Idea5_EulerCharacteristic.run_idea5
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from Core.base_experiment import BaseIdeaOrchestrator
from Core.base_masker import ROIMasker
from Core.utils import SubjectRecord
from Parameters.params_idea5 import Idea5Params
from Parameters.params_idea1 import Idea1Params
from Code.Idea1_PH_FC.fc_matrix_builder import FCMatrixBuilder
from Code.Idea5_EulerCharacteristic.euler_computer import EulerCharacteristicComputer
from Code.Idea5_EulerCharacteristic.betti_curve_builder import BettiCurveBuilder
from Code.Idea5_EulerCharacteristic.fda_analyzer import FDAAnalyzer
from Code.Idea5_EulerCharacteristic.idea5_visualizer import Idea5Visualizer


class Idea5Orchestrator(BaseIdeaOrchestrator):

    def __init__(self, params: Idea5Params):
        super().__init__(params, n_subjects=params.n_subjects, idea_name="Idea5",
                         dataset_name=params.dataset_name)
        # Reuse Idea1 FC builder
        self._idea1_params = Idea1Params(atlas_name=params.atlas_name)
        self.fc_builder = FCMatrixBuilder(self._idea1_params)
        self.ec_computer = EulerCharacteristicComputer(params)
        self.betti_builder = BettiCurveBuilder(params, self.output_manager)
        self.fda_analyzer = FDAAnalyzer(params, self.output_manager)
        self.visualizer = Idea5Visualizer(self.output_manager)

    # ------------------------------------------------------------------
    # Data diagnostics
    # ------------------------------------------------------------------

    def _log_data_summary(self, records: List[SubjectRecord]) -> None:
        """Print a structured summary of all subject-level data fields."""
        case_label = self.case_label
        print("\n" + "=" * 60)
        print("DATA SUMMARY")
        print("=" * 60)

        n = len(records)
        adhd_count = sum(r.adhd_label == 1 for r in records)
        ctrl_count = sum(r.adhd_label == 0 for r in records)
        print(f"  Total subjects : {n}  ({case_label}={adhd_count}, Control={ctrl_count})")

        # Time-series shapes
        ts_shapes = [r.time_series.shape if r.time_series is not None else None
                     for r in records]
        unique_shapes = sorted(set(s for s in ts_shapes if s is not None))
        print(f"  Time-series shapes (unique): {unique_shapes}")
        missing_ts = sum(1 for s in ts_shapes if s is None)
        if missing_ts:
            print(f"  WARNING: {missing_ts} subjects missing time_series")

        # Per-subject table header
        print()
        print(f"  {'SubjectID':<12} {'Label':>6} {'Age':>6} {'Sex':>4} "
              f"{'Severity':>10} {'TS shape':>14}")
        print(f"  {'-'*12} {'-'*6} {'-'*6} {'-'*4} {'-'*10} {'-'*14}")
        for r in records:
            sev_str = f"{r.adhd_measure:.2f}" if np.isfinite(r.adhd_measure) else "NaN"
            ts_str = str(r.time_series.shape) if r.time_series is not None else "None"
            label_str = case_label if r.adhd_label == 1 else "Ctrl"
            print(f"  {r.subject_id:<12} {label_str:>6} {r.age:>6.1f} "
                  f"{r.sex:>4} {sev_str:>10} {ts_str:>14}")

        # Severity column analysis
        sev_all = np.array([r.adhd_measure for r in records])
        sev_adhd = np.array([r.adhd_measure for r in records if r.adhd_label == 1])
        sev_ctrl = np.array([r.adhd_measure for r in records if r.adhd_label == 0])
        print()
        print("  Severity (adhd_measure) breakdown:")
        print(f"    All  : valid={np.isfinite(sev_all).sum()}/{n}, "
              f"range=[{np.nanmin(sev_all):.2f}, {np.nanmax(sev_all):.2f}], "
              f"mean={np.nanmean(sev_all):.2f}")
        print(f"    {case_label} : valid={np.isfinite(sev_adhd).sum()}/{len(sev_adhd)}, "
              f"range=[{np.nanmin(sev_adhd) if np.isfinite(sev_adhd).any() else 'N/A'}, "
              f"{np.nanmax(sev_adhd) if np.isfinite(sev_adhd).any() else 'N/A'}]")
        print(f"    Ctrl : valid={np.isfinite(sev_ctrl).sum()}/{len(sev_ctrl)}")

        # Age / sex distribution
        ages = np.array([r.age for r in records])
        print()
        print(f"  Age  : mean={np.nanmean(ages):.1f}, "
              f"range=[{np.nanmin(ages):.1f}, {np.nanmax(ages):.1f}]")
        male = sum(r.sex == 1 for r in records)
        print(f"  Sex  : male={male}, female={n - male}")
        print("=" * 60)

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def run_all_experiments(self, records: List[SubjectRecord]) -> None:
        print("\n=== Idea 5: Euler Characteristic and Betti Curves ===")

        records = [r for r in records if r.time_series is not None]
        labels = self.loader.get_labels_array(records)
        severity = self.loader.get_severity_array(records)

        self._log_data_summary(records)

        print("\n[Step 1] Building FC and distance matrices...")
        records = self.fc_builder.transform(records)
        records = [r for r in records if r.distance_matrix is not None]
        labels = self.loader.get_labels_array(records)
        severity = self.loader.get_severity_array(records)

        print(f"\n[Step 2] Computing Betti curves for {len(records)} subjects "
              f"({self.params.n_epsilon_steps} ε steps)...")
        betti_data = self.ec_computer.compute_all_subjects(records)
        if not betti_data:
            print("  No valid subjects. Aborting.")
            return

        # Sync labels to betti_data (subjects that passed distance_matrix check)
        # betti_data already filtered, re-extract labels
        betti_labels = np.array([d["adhd_label"] for d in betti_data])
        betti_sev = np.array([
            severity[i] for i, r in enumerate(records)
            if any(d["subject_id"] == r.subject_id for d in betti_data)
        ])
        # Safer: rebuild severity from records in same order as betti_data
        subject_id_to_sev = {r.subject_id: sev for r, sev in zip(records, severity)}
        betti_sev = np.array([
            subject_id_to_sev.get(d["subject_id"], np.nan) for d in betti_data
        ])

        # Build summary DataFrame
        results_df = self.betti_builder.build_results_dataframe(
            betti_data, betti_labels, betti_sev)
        self.output_manager.save_dataframe(results_df, "betti_summary.csv")

        if self.params.run_group_ec_curves:
            print("\n[Exp 1] Group-averaged EC and Betti curves...")
            group_curves = self.betti_builder.group_mean_curves(
                betti_data, betti_labels)
            self.visualizer.plot_group_betti_curves(group_curves)
            self.visualizer.plot_group_ec_curves(group_curves)

            # Group-averaged FC matrices
            adhd_mask = betti_labels == 1
            adhd_fc = np.mean(
                [r.fc_matrix for r, m in zip(records, adhd_mask) if m and r.fc_matrix is not None],
                axis=0)
            ctrl_fc = np.mean(
                [r.fc_matrix for r, m in zip(records, adhd_mask) if not m and r.fc_matrix is not None],
                axis=0)
            masker = ROIMasker(atlas_name=self.params.atlas_name)
            masker.fit()
            roi_labels = masker.roi_labels_ or [str(i) for i in range(39)]
            self.visualizer.plot_group_fc_matrices(adhd_fc, ctrl_fc, roi_labels)

        if self.params.run_betti_vs_severity:
            print("\n[Exp 2] Betti area vs ADHD severity...")
            for dim in self.params.homology_dimensions:
                corr = self.betti_builder.betti_vs_severity_experiment(
                    betti_data, betti_sev, dim)
                if "correlation" in corr:
                    self.visualizer.plot_severity_scatter(
                        areas=corr["areas"],
                        severity=betti_sev,
                        labels=betti_labels,
                        dim=dim,
                        corr=corr["correlation"],
                        pvalue=corr["pvalue"],
                    )

        if self.params.run_network_analysis:
            print("\n[Exp 3] Network-specific Betti curves...")
            # Need atlas labels — re-fit masker
            masker = ROIMasker(atlas_name=self.params.atlas_name)
            masker.fit()
            atlas_labels = masker.roi_labels_ or []
            network_results = self.betti_builder.network_analysis_experiment(
                records, atlas_labels, betti_labels)
            if network_results:
                self.visualizer.plot_network_betti_curves(network_results, dim=1)

        if self.params.run_fda and self.params.run_fda_analysis:
            print("\n[Exp 4] Functional Data Analysis on β1 curves...")
            try:
                betti_1_curves = np.array([d["betti_1"] for d in betti_data])
                epsilon_vals = betti_data[0]["epsilon"]
                fd = self.fda_analyzer.curves_to_functional_data(
                    betti_1_curves, epsilon_vals)
                fd_smooth = self.fda_analyzer.smooth_functional_data(fd)
                test_results = self.fda_analyzer.functional_group_test(
                    fd, betti_labels)
                fpca_results = self.fda_analyzer.functional_pca(fd, n_components=3)
                self.visualizer.plot_fda_results(
                    fd, betti_labels, test_results, fpca_results)
            except ImportError as e:
                print(f"  skfda not available: {e}. Skipping FDA analysis.")

        print("\n=== Idea 5 complete. Outputs saved to Output/Idea5/ ===")


if __name__ == "__main__":
    params = Idea5Params()
    orchestrator = Idea5Orchestrator(params)
    orchestrator.execute()
