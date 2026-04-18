"""
Code/Idea3_SlidingWindow/window_embedder.py
Sliding-window embedding of BOLD time series into point clouds.
"""

from __future__ import annotations

from typing import List

import numpy as np

from Core.utils import SubjectRecord
from Parameters.params_idea3 import Idea3Params


class SlidingWindowEmbedder:
    """
    Converts a 1D or multi-ROI BOLD signal into a sequence of
    point clouds via sliding window embedding.

    Each window of length w becomes a single point in R^w
    (or a (w, 1) array for ripser).
    """

    def __init__(self, params: Idea3Params):
        self.params = params

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_signal(self, time_series: np.ndarray) -> np.ndarray:
        """
        Extract 1D signal from ROI time series.

        If use_mean_bold: average across ROIs → (n_timepoints,)
        Else: use params.roi_index column → (n_timepoints,)
        """
        if self.params.use_mean_bold:
            return time_series.mean(axis=1)
        else:
            return time_series[:, self.params.roi_index]

    def embed(
        self,
        signal_1d: np.ndarray,
        window_length: int = None,
        step_size: int = None,
    ) -> List[np.ndarray]:
        """
        Sliding window embedding of a 1D signal.

        Each window is returned as shape (window_length, 1) — suitable
        for ripser as a point cloud in R^1 (where ripser computes
        H1 as loops in the delay-embedded attractor).

        Parameters
        ----------
        signal_1d : np.ndarray, shape (n_timepoints,)
        window_length : int, overrides params.window_length
        step_size : int, overrides params.step_size

        Returns
        -------
        List of np.ndarray, each shape (window_length, 1)
        """
        w = window_length or self.params.window_length
        s = step_size or self.params.step_size
        n = len(signal_1d)
        windows = []
        for start in range(0, n - w + 1, s):
            window = signal_1d[start: start + w].reshape(-1, 1)
            windows.append(window)
        return windows

    def embed_all(
        self,
        records: List[SubjectRecord],
        window_length: int = None,
        step_size: int = None,
    ) -> List[List[np.ndarray]]:
        """
        Embed all subjects.

        Returns
        -------
        List (per subject) of List (per window) of np.ndarray (window_length, 1)
        """
        result = []
        for rec in records:
            if rec.time_series is None:
                result.append([])
                continue
            signal = self.extract_signal(rec.time_series)
            windows = self.embed(signal, window_length=window_length,
                                 step_size=step_size)
            result.append(windows)
        return result

    def get_n_windows(self, n_timepoints: int,
                       window_length: int = None,
                       step_size: int = None) -> int:
        """Return the number of windows for a given signal length."""
        w = window_length or self.params.window_length
        s = step_size or self.params.step_size
        return max(0, (n_timepoints - w) // s + 1)
