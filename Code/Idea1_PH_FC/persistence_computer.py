"""
Code/Idea1_PH_FC/persistence_computer.py
Computes persistent homology on distance matrices using ripser.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from ripser import ripser

from Core.base_tda import BasePersistenceComputer
from Parameters.params_idea1 import Idea1Params


class PHFCComputer(BasePersistenceComputer):
    """
    Computes H0, H1, H2 persistent homology on FC-derived distance matrices.

    Uses ripser with distance_matrix=True for efficiency.
    Stores raw results after fit_transform for downstream reuse.
    """

    def __init__(self, params: Idea1Params):
        self.params = params
        self.diagrams_: Optional[List[dict]] = None   # list of ripser result dicts

    # ------------------------------------------------------------------
    # BasePersistenceComputer interface
    # ------------------------------------------------------------------

    def fit_transform(self, distance_matrices: List[np.ndarray]) -> List[dict]:
        """
        Run ripser on each subject's distance matrix.

        Parameters
        ----------
        distance_matrices : List[np.ndarray]
            Each element is (n_rois, n_rois) symmetric distance matrix.

        Returns
        -------
        List[dict]
            Each dict has key 'dgms': List[np.ndarray] indexed by dimension.
        """
        results = []
        for i, D in enumerate(distance_matrices):
            result = ripser(
                D,
                maxdim=self.params.max_dimension,
                distance_matrix=True,
                thresh=self.params.max_edge_length,
            )
            results.append(result)
            print(f"  Subject {i+1}/{len(distance_matrices)}: "
                  f"H0={len(result['dgms'][0])}, "
                  f"H1={len(result['dgms'][1])}, "
                  f"H2={len(result['dgms'][2]) if len(result['dgms']) > 2 else 'N/A'} bars")
        self.diagrams_ = results
        return results

    def get_diagram_for_dimension(
        self, diagrams: List[dict], dim: int
    ) -> List[np.ndarray]:
        """
        Extract and clean dim-specific diagrams from ripser output.

        Strips bars with infinite death value (the last H0 component).
        Returns list of (n_bars, 2) arrays.
        """
        out = []
        for result in diagrams:
            dgm = result["dgms"][dim].copy()
            dgm = self.strip_infinite_bars(dgm)
            out.append(dgm)
        return out

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def get_all_subjects_diagram(self, dim: int) -> List[np.ndarray]:
        """Use stored self.diagrams_ (must call fit_transform first)."""
        if self.diagrams_ is None:
            raise RuntimeError("Call fit_transform() first.")
        return self.get_diagram_for_dimension(self.diagrams_, dim)

    def total_persistence_per_subject(
        self, diagrams: List[np.ndarray], power: float = 1.0
    ) -> np.ndarray:
        """
        Returns (n_subjects,) array of total persistence values.
        Useful as a scalar summary for group comparison.
        """
        return np.array([self.total_persistence(d, power) for d in diagrams])
