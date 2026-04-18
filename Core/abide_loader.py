"""
Core/abide_loader.py
Fetches the nilearn ABIDE PCP dataset and wraps subjects in SubjectRecord objects.

DX_GROUP coding: 1 = Autism Spectrum Disorder, 2 = Typical Control
Maps to SubjectRecord.adhd_label: 1 = case (ASD), 0 = control
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from nilearn import datasets

from Core.utils import SubjectRecord


class ABIDEDataLoader:
    """
    Loads the nilearn ABIDE PCP dataset and returns a list of SubjectRecord objects.

    The interface is identical to ADHDDataLoader so both can be used
    interchangeably in BaseIdeaOrchestrator.

    Usage
    -----
    loader = ABIDEDataLoader(n_subjects=30)
    records = loader.fetch()
    """

    def __init__(
        self,
        n_subjects: int = 30,
        data_dir: Optional[str] = None,
        pipeline: str = "cpac",
        quality_checked: bool = True,
    ):
        self.n_subjects = n_subjects
        self.data_dir = data_dir
        self.pipeline = pipeline
        self.quality_checked = quality_checked
        self._records: Optional[List[SubjectRecord]] = None

    # ------------------------------------------------------------------
    # Public API  (matches ADHDDataLoader interface)
    # ------------------------------------------------------------------

    def fetch(self) -> List[SubjectRecord]:
        """Download (or load from cache) the ABIDE dataset and build records."""
        print(f"Fetching ABIDE PCP dataset ({self.n_subjects} subjects, "
              f"pipeline='{self.pipeline}', quality_checked={self.quality_checked})...")
        abide_data = datasets.fetch_abide_pcp(
            n_subjects=self.n_subjects,
            data_dir=self.data_dir,
            pipeline=self.pipeline,
            quality_checked=self.quality_checked,
            derivatives=["func_preproc"],
        )
        func_filenames = abide_data.func_preproc
        phenotypic = abide_data.phenotypic

        import pandas as pd
        if isinstance(phenotypic, pd.DataFrame):
            col_names = list(phenotypic.columns)
            phenotypic = phenotypic.reset_index(drop=True)
        else:
            col_names = list(phenotypic.dtype.names)
        print(f"  Phenotypic columns: {col_names}")

        # Log resolved field mapping
        severity_candidates = ["ADOS_TOTAL", "SRS_RAW_TOTAL",
                               "VINELAND_ADAPTIVE_BEHAVIOR_COMPOSITE",
                               "ADOS_COMM", "ADOS_SOCIAL"]
        resolved_label_col  = next((c for c in ["DX_GROUP"] if c in col_names), None)
        resolved_sev_col    = next((c for c in severity_candidates if c in col_names), None)
        resolved_age_col    = next((c for c in ["AGE_AT_SCAN", "age"] if c in col_names), None)
        resolved_sex_col    = next((c for c in ["SEX", "sex"] if c in col_names), None)
        resolved_id_col     = next((c for c in ["SUB_ID", "subject_id", "Subject"] if c in col_names), None)
        print(f"  Field mapping  → label='{resolved_label_col}', "
              f"severity='{resolved_sev_col}', "
              f"age='{resolved_age_col}', sex='{resolved_sex_col}', "
              f"id='{resolved_id_col}'")

        # Severity column diagnostics
        if isinstance(phenotypic, pd.DataFrame):
            sev_cols = [c for c in severity_candidates if c in col_names]
            if sev_cols:
                sev_summary = phenotypic[sev_cols].describe().round(2).to_string()
                print(f"  Severity columns summary:\n{sev_summary}")
                nan_counts = phenotypic[sev_cols].isna().sum().to_dict()
                print(f"  NaN counts: {nan_counts}")

        records: List[SubjectRecord] = []
        n_pheno = len(phenotypic)
        # Guard: if more func files than phenotypic rows, truncate
        if len(func_filenames) > n_pheno:
            print(f"  Note: {len(func_filenames)} scans found for {n_pheno} subjects. "
                  f"Using first scan per subject.")
            func_filenames = func_filenames[:n_pheno]

        for i, func_path in enumerate(func_filenames):
            if isinstance(phenotypic, pd.DataFrame):
                row = phenotypic.iloc[i]
            else:
                row = phenotypic[i]

            subject_id = str(self._get_field(
                row, col_names, ["SUB_ID", "subject_id", "Subject"], default=str(i)))

            # DX_GROUP: 1=ASD, 2=Control → remap to 1=case, 0=control
            dx_raw = self._get_field(row, col_names, ["DX_GROUP"], default=2)
            try:
                dx_int = int(float(dx_raw))
            except (TypeError, ValueError):
                dx_int = 2
            adhd_label = 1 if dx_int == 1 else 0

            age = float(self._get_field(
                row, col_names, ["AGE_AT_SCAN", "age"], default=np.nan))

            sex_raw = self._get_field(row, col_names, ["SEX", "sex"], default=1)
            sex = self._parse_sex_abide(sex_raw)

            adhd_measure = float(self._get_field(
                row, col_names, severity_candidates, default=np.nan))

            records.append(SubjectRecord(
                subject_id=subject_id,
                func_path=str(func_path),
                adhd_label=adhd_label,
                age=age,
                sex=sex,
                adhd_measure=adhd_measure,
            ))

        print(f"  Loaded {len(records)} subjects "
              f"({sum(r.is_adhd() for r in records)} ASD, "
              f"{sum(r.is_control() for r in records)} control).")
        self._records = records
        return records

    # ------------------------------------------------------------------
    # Filtering helpers  (identical interface to ADHDDataLoader)
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
    def _parse_sex_abide(value) -> int:
        """
        ABIDE SEX coding: 1=male, 2=female.
        Returns: 1=male, 0=female (same convention as ADHDDataLoader).
        """
        try:
            v = int(float(value))
            return 1 if v == 1 else 0
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _get_field(row, col_names: list, candidates: List[str], default):
        """Try candidate column names in order; return first match."""
        for name in candidates:
            if name in col_names:
                val = row[name]
                try:
                    return val.item()
                except AttributeError:
                    return val
        return default
