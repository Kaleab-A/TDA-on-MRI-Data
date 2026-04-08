"""
Code/Idea5_EulerCharacteristic/betti_curve_builder.py
Aggregates per-subject Betti curves into group-level summaries and experiments.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from Core.utils import OutputManager, StatisticsHelper, SubjectRecord
from Parameters.params_idea5 import Idea5Params


class BettiCurveBuilder:
    """
    Aggregates Betti curves and Euler characteristic curves into
    group-level summaries and scalar biomarkers.
    """

    def __init__(self, params: Idea5Params, output_manager: OutputManager):
        self.params = params
        self.output_manager = output_manager

    # ------------------------------------------------------------------
    # Group-level aggregation
    # ------------------------------------------------------------------

    def group_mean_curves(
        self, betti_data: List[dict], labels: np.ndarray
    ) -> dict:
        """
        Compute mean ± std of each Betti curve and EC curve by group.

        Returns
        -------
        dict with:
            'adhd': {'betti_0': mean, 'betti_0_std': std, ...}
            'control': {...}
            'epsilon': epsilon_values
        """
        adhd_mask = labels == 1
        result = {}
        for group_name, mask in [("adhd", adhd_mask), ("control", ~adhd_mask)]:
            group_data = [d for d, m in zip(betti_data, mask) if m]
            if not group_data:
                result[group_name] = {}
                continue
            group_stats = {}
            for key in ("betti_0", "betti_1", "betti_2", "euler"):
                arr = np.vstack([d[key] for d in group_data])
                group_stats[key] = arr.mean(axis=0)
                group_stats[f"{key}_std"] = arr.std(axis=0)
            result[group_name] = group_stats
        result["epsilon"] = betti_data[0]["epsilon"] if betti_data else np.array([])
        return result

    # ------------------------------------------------------------------
    # Experiment 1: Severity correlation
    # ------------------------------------------------------------------

    def betti_vs_severity_experiment(
        self,
        betti_data: List[dict],
        severity_scores: np.ndarray,
        dim: int,
    ) -> dict:
        """
        Correlate area under beta_{dim} curve with ADHD severity.
        Only uses ADHD subjects (label==1) since severity scores are
        NaN for controls — dsm_iv_tot is a clinical ADHD measure.
        """
        key = f"betti_{dim}"
        eps = betti_data[0]["epsilon"]
        areas = np.array([np.trapz(d[key], eps) for d in betti_data])
        adhd_labels = np.array([d["adhd_label"] for d in betti_data])

        # Filter to ADHD subjects with valid (non-NaN) severity scores
        adhd_mask = adhd_labels == 1
        valid = adhd_mask & np.isfinite(severity_scores)
        if valid.sum() < 3:
            print(f"  H{dim}: only {valid.sum()}/{adhd_mask.sum()} ADHD subjects have valid "
                  f"severity scores (others are NaN in phenotypic data) — skipping correlation.")
            return {"dim": dim, "n_valid": int(valid.sum()), "areas": areas}

        corr_result = StatisticsHelper.spearman_correlation(
            areas[valid], severity_scores[valid]
        )
        corr_result["dim"] = dim
        corr_result["areas"] = areas
        corr_result["valid_mask"] = valid
        print(f"  H{dim} area vs severity (n={valid.sum()} ADHD): "
              f"rho={corr_result['correlation']:.3f}, p={corr_result['pvalue']:.4f}")
        return corr_result

    # ------------------------------------------------------------------
    # Experiment 2: Network-specific analysis
    # ------------------------------------------------------------------

    def network_analysis_experiment(
        self,
        records: List[SubjectRecord],
        atlas_labels: List[str],
        labels: np.ndarray,
    ) -> dict:
        """
        For each network in params.network_names, extract the ROI-subset
        distance matrix and compute Betti curves.

        Returns
        -------
        {network_name: group_curves_dict}
        """
        from Code.Idea5_EulerCharacteristic.euler_computer import EulerCharacteristicComputer
        ec_computer = EulerCharacteristicComputer(self.params)
        network_results = {}

        for network_name in self.params.network_names:
            roi_indices = ec_computer.extract_network_rois(atlas_labels, network_name)
            print(f"  Network '{network_name}': {len(roi_indices)} ROIs {roi_indices}")
            if len(roi_indices) < 2:
                print(f"    Skipping — too few ROIs.")
                continue

            network_betti_data = []
            for rec, lbl in zip(records, labels):
                if rec.distance_matrix is None:
                    continue
                data = ec_computer.compute_network_betti(
                    rec.distance_matrix, roi_indices)
                data["subject_id"] = rec.subject_id
                data["adhd_label"] = int(lbl)
                network_betti_data.append(data)

            if network_betti_data:
                net_labels = np.array([d["adhd_label"] for d in network_betti_data])
                group_curves = self.group_mean_curves(network_betti_data, net_labels)
                network_results[network_name] = group_curves

        return network_results

    # ------------------------------------------------------------------
    # Summary DataFrame
    # ------------------------------------------------------------------

    def build_results_dataframe(
        self,
        betti_data: List[dict],
        labels: np.ndarray,
        severity: np.ndarray,
    ) -> pd.DataFrame:
        """
        One row per subject with area-under-curve for each Betti dimension,
        mean EC, label, and severity.
        """
        rows = []
        for d, lbl, sev in zip(betti_data, labels, severity):
            eps = d["epsilon"]
            rows.append({
                "subject_id": d.get("subject_id", ""),
                "adhd_label": int(lbl),
                "adhd_measure": float(sev),
                "area_betti_0": float(np.trapz(d["betti_0"], eps)),
                "area_betti_1": float(np.trapz(d["betti_1"], eps)),
                "area_betti_2": float(np.trapz(d["betti_2"], eps)),
                "mean_ec": float(np.mean(d["euler"])),
                "mean_betti_0": float(np.mean(d["betti_0"])),
                "mean_betti_1": float(np.mean(d["betti_1"])),
                "mean_betti_2": float(np.mean(d["betti_2"])),
            })
        return pd.DataFrame(rows)
