"""
Code/Idea3_SlidingWindow/window_ph_computer.py
Computes persistent homology on each sliding window's point cloud.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from ripser import ripser

from Core.base_tda import BasePersistenceComputer
from Parameters.params_idea3 import Idea3Params


class WindowPHComputer(BasePersistenceComputer):
    """
    Computes H0 + H1 persistent homology on each window's point cloud.
    Designed to operate subject-by-subject for memory efficiency.
    """

    def __init__(self, params: Idea3Params):
        self.params = params

    # ------------------------------------------------------------------
    # BasePersistenceComputer interface
    # ------------------------------------------------------------------

    def fit_transform(self, windows: List[np.ndarray]) -> List[dict]:
        """
        Compute ripser on each window for a single subject.

        Parameters
        ----------
        windows : List[np.ndarray]
            Each element is (window_length, 1) — a 1D delay-embedded point cloud.

        Returns
        -------
        List[dict]
            Each dict is a ripser result with key 'dgms'.
        """
        results = []
        for w in windows:
            result = ripser(
                w,
                maxdim=self.params.max_dimension,
                thresh=self.params.max_edge_length,
            )
            results.append(result)
        return results

    def get_diagram_for_dimension(
        self, diagrams: List[dict], dim: int
    ) -> List[np.ndarray]:
        """Extract and clean dim-specific diagrams from ripser output."""
        out = []
        for result in diagrams:
            dgm = result["dgms"][dim].copy()
            dgm = self.strip_infinite_bars(dgm)
            out.append(dgm)
        return out

    # ------------------------------------------------------------------
    # All subjects
    # ------------------------------------------------------------------

    def fit_transform_all_subjects(
        self, all_windows: List[List[np.ndarray]]
    ) -> List[List[dict]]:
        """
        Nested computation: per subject, per window.

        Parameters
        ----------
        all_windows : List (subjects) of List (windows) of np.ndarray

        Returns
        -------
        List (subjects) of List (windows) of ripser dict
        """
        all_results = []
        for i, subject_windows in enumerate(all_windows):
            print(f"  Subject {i+1}/{len(all_windows)}: "
                  f"{len(subject_windows)} windows")
            if len(subject_windows) == 0:
                all_results.append([])
                continue
            subject_diagrams = self.fit_transform(subject_windows)
            all_results.append(subject_diagrams)
        return all_results

    # ------------------------------------------------------------------
    # Loop score
    # ------------------------------------------------------------------

    def compute_h1_lifetimes_per_window(
        self, window_diagrams: List[dict]
    ) -> np.ndarray:
        """
        For each window, compute the total H1 persistence (loop score).

        Returns
        -------
        np.ndarray, shape (n_windows,)
        """
        scores = []
        for result in window_diagrams:
            if len(result["dgms"]) < 2:
                scores.append(0.0)
                continue
            h1 = result["dgms"][1].copy()
            h1 = self.strip_infinite_bars(h1)
            scores.append(self.total_persistence(h1))
        return np.array(scores)
