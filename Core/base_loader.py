"""
Core/base_loader.py
Fetches the nilearn ADHD dataset and wraps subjects in SubjectRecord objects.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from nilearn import datasets

from Core.utils import SubjectRecord


class ADHDDataLoader:
    """
    Loads the nilearn ADHD dataset and returns a list of SubjectRecord objects.

    Usage
    -----
    loader = ADHDDataLoader(n_subjects=30)
    records = loader.fetch()
    """

    def __init__(self, n_subjects: int = 30, data_dir: Optional[str] = None):
        self.n_subjects = n_subjects
        self.data_dir = data_dir
        self._records: Optional[List[SubjectRecord]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch(self) -> List[SubjectRecord]:
        """Download (or load from cache) the ADHD dataset and build records."""
        print(f"Fetching ADHD dataset ({self.n_subjects} subjects)...")
        adhd_data = datasets.fetch_adhd(
            n_subjects=self.n_subjects,
            data_dir=self.data_dir,
        )
        fmri_filenames = adhd_data.func
        phenotypic = adhd_data.phenotypic

        # phenotypic may be a pandas DataFrame or a numpy structured array
        import pandas as pd
        if isinstance(phenotypic, pd.DataFrame):
            col_names = list(phenotypic.columns)
        else:
            col_names = list(phenotypic.dtype.names)
        print(f"  Phenotypic columns: {col_names}")

        records: List[SubjectRecord] = []
        for i, func_path in enumerate(fmri_filenames):
            if isinstance(phenotypic, pd.DataFrame):
                row = phenotypic.iloc[i]
            else:
                row = phenotypic[i]

            subject_id = self._get_field(row, col_names, ["Subject", "subject_id"], default=str(i))
            adhd_label = int(self._get_field(row, col_names, ["ADHD", "adhd", "DX", "dx"], default=0))
            age = float(self._get_field(row, col_names, ["Age", "age"], default=np.nan))
            sex = int(self._get_field(row, col_names, ["Gender", "gender", "Sex", "sex"], default=0))
            adhd_measure = float(
                self._get_field(row, col_names,
                                ["ADHD_Index", "adhd_index", "ADHD_Measure"], default=np.nan)
            )
            records.append(SubjectRecord(
                subject_id=str(subject_id),
                func_path=str(func_path),
                adhd_label=adhd_label,
                age=age,
                sex=sex,
                adhd_measure=adhd_measure,
            ))

        print(f"  Loaded {len(records)} subjects "
              f"({sum(r.is_adhd() for r in records)} ADHD, "
              f"{sum(r.is_control() for r in records)} control).")
        self._records = records
        return records

    # ------------------------------------------------------------------
    # Filtering helpers
    # ------------------------------------------------------------------

    def get_adhd_subjects(self, records: List[SubjectRecord]) -> List[SubjectRecord]:
        return [r for r in records if r.is_adhd()]

    def get_control_subjects(self, records: List[SubjectRecord]) -> List[SubjectRecord]:
        return [r for r in records if r.is_control()]

    def get_labels_array(self, records: List[SubjectRecord]) -> np.ndarray:
        return np.array([r.adhd_label for r in records], dtype=int)

    def get_severity_array(self, records: List[SubjectRecord]) -> np.ndarray:
        return np.array([r.adhd_measure for r in records], dtype=float)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_field(row, col_names: list, candidates: List[str], default):
        """Try candidate column names in order; return first match."""
        for name in candidates:
            if name in col_names:
                val = row[name]
                # pandas scalar may be a 0-d Series; unwrap it
                try:
                    return val.item()
                except AttributeError:
                    return val
        return default
