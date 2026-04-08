"""
Code/Idea5_EulerCharacteristic/euler_computer.py
Computes Betti number curves and Euler characteristic from distance matrices.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from ripser import ripser

from Core.utils import SubjectRecord
from Parameters.params_idea5 import Idea5Params


class EulerCharacteristicComputer:
    """
    For each subject's FC distance matrix, computes Betti numbers β0, β1, β2
    as functions of the filtration threshold ε, then derives the Euler
    characteristic χ(ε) = β0(ε) - β1(ε) + β2(ε).

    Strategy: run ripser once per subject (cheap), then sweep ε by
    counting how many bars are active at each threshold.
    """

    def __init__(self, params: Idea5Params):
        self.params = params
        self._epsilon_values = np.linspace(
            params.epsilon_min, params.epsilon_max, params.n_epsilon_steps
        )

    @property
    def epsilon_values(self) -> np.ndarray:
        return self._epsilon_values

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_betti_numbers(self, distance_matrix: np.ndarray) -> dict:
        """
        Compute Betti curves for a single distance matrix.

        Returns
        -------
        dict with:
            'betti_0', 'betti_1', 'betti_2' : np.ndarray, shape (n_epsilon_steps,)
            'euler' : np.ndarray, shape (n_epsilon_steps,)
            'epsilon' : np.ndarray, shape (n_epsilon_steps,)
        """
        result = ripser(
            distance_matrix,
            maxdim=max(self.params.homology_dimensions),
            distance_matrix=True,
            thresh=self.params.epsilon_max,
        )
        dgms = result["dgms"]

        betti = {}
        for dim in self.params.homology_dimensions:
            if dim >= len(dgms):
                betti[dim] = np.zeros(len(self._epsilon_values))
                continue
            dgm = dgms[dim].copy()
            # Replace inf deaths with epsilon_max + small delta
            inf_mask = ~np.isfinite(dgm[:, 1])
            dgm[inf_mask, 1] = self.params.epsilon_max + 1e-6

            betti_curve = np.zeros(len(self._epsilon_values))
            for i, eps in enumerate(self._epsilon_values):
                # Bar is active at ε iff birth <= ε < death
                active = np.sum((dgm[:, 0] <= eps) & (dgm[:, 1] > eps))
                betti_curve[i] = active
            betti[dim] = betti_curve

        euler = (betti.get(0, 0)
                 - betti.get(1, 0)
                 + betti.get(2, 0))

        return {
            "betti_0": betti.get(0, np.zeros(len(self._epsilon_values))),
            "betti_1": betti.get(1, np.zeros(len(self._epsilon_values))),
            "betti_2": betti.get(2, np.zeros(len(self._epsilon_values))),
            "euler": euler,
            "epsilon": self._epsilon_values.copy(),
        }

    def compute_all_subjects(
        self, records: List[SubjectRecord]
    ) -> List[dict]:
        """
        Compute Betti curves for all subjects.
        Skips records without distance_matrix.
        """
        results = []
        for i, rec in enumerate(records):
            if rec.distance_matrix is None:
                print(f"  WARNING — {rec.subject_id}: no distance_matrix, skipping.")
                continue
            print(f"  Subject {i+1}/{len(records)}: {rec.subject_id}")
            data = self.compute_betti_numbers(rec.distance_matrix)
            data["subject_id"] = rec.subject_id
            data["adhd_label"] = rec.adhd_label
            results.append(data)
        return results

    def compute_area_under_curve(
        self, betti_curve: np.ndarray, epsilon_values: np.ndarray = None
    ) -> float:
        """Area under a Betti curve via trapezoidal integration."""
        eps = epsilon_values if epsilon_values is not None else self._epsilon_values
        return float(np.trapz(betti_curve, eps))

    def extract_network_rois(
        self, atlas_labels: List[str], network_name: str
    ) -> List[int]:
        """
        Find ROI indices where the label contains network_name (case-insensitive).
        Returns list of matching indices.
        """
        name_lower = network_name.lower()
        indices = [
            i for i, label in enumerate(atlas_labels)
            if name_lower in str(label).lower()
        ]
        return indices

    def compute_network_betti(
        self,
        distance_matrix: np.ndarray,
        roi_indices: List[int],
    ) -> dict:
        """Compute Betti curves on a ROI-subset distance matrix."""
        if len(roi_indices) < 2:
            n_eps = len(self._epsilon_values)
            return {
                "betti_0": np.zeros(n_eps),
                "betti_1": np.zeros(n_eps),
                "betti_2": np.zeros(n_eps),
                "euler": np.zeros(n_eps),
                "epsilon": self._epsilon_values.copy(),
            }
        sub_D = distance_matrix[np.ix_(roi_indices, roi_indices)]
        return self.compute_betti_numbers(sub_D)
