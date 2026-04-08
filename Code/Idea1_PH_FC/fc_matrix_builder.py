"""
Code/Idea1_PH_FC/fc_matrix_builder.py
Converts ROI time series → FC matrix → distance matrix per subject.
"""

from __future__ import annotations

from typing import List

import numpy as np
from nilearn.connectome import ConnectivityMeasure

from Core.utils import SubjectRecord
from Parameters.params_idea1 import Idea1Params


class FCMatrixBuilder:
    """
    Builds functional connectivity (FC) matrices and derived distance matrices
    from ROI time series stored in SubjectRecords.

    Distance matrix: D = 1 - |FC|, clipped to [0, 1], diagonal = 0.
    This is a valid distance measure for Vietoris-Rips filtration.
    """

    def __init__(self, params: Idea1Params):
        self.params = params
        self._connectivity_measure: ConnectivityMeasure = ConnectivityMeasure(
            kind=params.correlation_kind
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_fc_matrix(self, time_series: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        time_series : np.ndarray, shape (n_timepoints, n_rois)

        Returns
        -------
        np.ndarray, shape (n_rois, n_rois)
        """
        fc = self._connectivity_measure.fit_transform([time_series])[0]
        return fc

    def compute_distance_matrix(self, fc_matrix: np.ndarray) -> np.ndarray:
        """
        D = 1 - |FC|, clipped to [0, 1]. Diagonal forced to 0.
        """
        D = 1.0 - np.abs(fc_matrix)
        D = np.clip(D, 0.0, 1.0)
        np.fill_diagonal(D, 0.0)
        # Symmetrise in case of floating-point asymmetry
        D = (D + D.T) / 2.0
        return D

    def transform(self, records: List[SubjectRecord]) -> List[SubjectRecord]:
        """
        Fill record.fc_matrix and record.distance_matrix for each record.
        Skips records where time_series is None.
        """
        for rec in records:
            if rec.time_series is None:
                print(f"  WARNING — {rec.subject_id}: time_series is None, skipping.")
                continue
            rec.fc_matrix = self.compute_fc_matrix(rec.time_series)
            rec.distance_matrix = self.compute_distance_matrix(rec.fc_matrix)
        return records
