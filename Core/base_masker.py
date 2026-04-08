"""
Core/base_masker.py
Wraps nilearn maskers + atlas fetching to extract ROI time series.

MSDL is a probabilistic (4D) atlas → uses NiftiMapsMasker.
Schaefer is a hard-label (3D) atlas → uses NiftiLabelsMasker.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker, NiftiMapsMasker

from Core.utils import SubjectRecord

# Atlases that are probabilistic (4D) and need NiftiMapsMasker
_PROBABILISTIC_ATLASES = {"msdl"}


class ROIMasker:
    """
    Fetches an atlas and extracts ROI time series for each SubjectRecord.

    Parameters
    ----------
    atlas_name : str
        One of 'msdl', 'schaefer100', 'schaefer200'.
    standardize : str or bool
        'zscore_sample' recommended for correlation stability.
    detrend : bool
    low_pass, high_pass : float
        Band-pass filter bounds (Hz).
    t_r : float
        Repetition time in seconds (2.0 for ADHD dataset).
    memory_level : int
        Nilearn caching level.
    """

    def __init__(
        self,
        atlas_name: str = "msdl",
        standardize: str = "zscore_sample",
        detrend: bool = True,
        low_pass: float = 0.1,
        high_pass: float = 0.01,
        t_r: float = 2.0,
        memory_level: int = 1,
    ):
        self.atlas_name = atlas_name
        self.standardize = standardize
        self.detrend = detrend
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.t_r = t_r
        self.memory_level = memory_level

        # Set after fit()
        self.masker_ = None
        self.roi_labels_: Optional[List[str]] = None
        self.n_rois_: Optional[int] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, example_img=None) -> "ROIMasker":
        atlas_img, roi_labels = self._fetch_atlas()
        self.roi_labels_ = roi_labels
        self.n_rois_ = len(roi_labels)

        common_kwargs = dict(
            standardize=self.standardize,
            detrend=self.detrend,
            low_pass=self.low_pass,
            high_pass=self.high_pass,
            t_r=self.t_r,
            memory_level=self.memory_level,
            verbose=0,
        )

        if self.atlas_name.lower() in _PROBABILISTIC_ATLASES:
            self.masker_ = NiftiMapsMasker(maps_img=atlas_img, **common_kwargs)
        else:
            self.masker_ = NiftiLabelsMasker(labels_img=atlas_img, **common_kwargs)

        # Passing an example image avoids the resampling-at-transform-time warning
        self.masker_.fit(example_img)
        print(f"  ROIMasker fitted: atlas='{self.atlas_name}', n_rois={self.n_rois_}")
        return self

    def transform(self, records: List[SubjectRecord]) -> List[SubjectRecord]:
        """Extract time series for each record. Sets record.time_series."""
        if self.masker_ is None:
            raise RuntimeError("Call fit() before transform().")
        valid = []
        for rec in records:
            try:
                ts = self.masker_.transform(rec.func_path)  # (n_timepoints, n_rois)
                rec.time_series = ts
                valid.append(rec)
                print(f"  {rec.subject_id}: time_series shape {ts.shape}")
            except Exception as exc:
                print(f"  WARNING — {rec.subject_id} failed masking: {exc}")
        return valid

    def fit_transform(self, records: List[SubjectRecord]) -> List[SubjectRecord]:
        # Pass first image to fit() so the masker pre-resamples (avoids warnings)
        example_img = records[0].func_path if records else None
        return self.fit(example_img=example_img).transform(records)

    # ------------------------------------------------------------------
    # Atlas fetching
    # ------------------------------------------------------------------

    def _fetch_atlas(self) -> Tuple[object, List[str]]:
        name = self.atlas_name.lower()
        if name == "msdl":
            return self._fetch_msdl()
        elif name in ("schaefer100", "schaefer200"):
            n_rois = 100 if name == "schaefer100" else 200
            return self._fetch_schaefer(n_rois)
        else:
            raise ValueError(
                f"Unknown atlas '{self.atlas_name}'. "
                f"Supported: 'msdl', 'schaefer100', 'schaefer200'"
            )

    @staticmethod
    def _fetch_msdl() -> Tuple[object, List[str]]:
        atlas = datasets.fetch_atlas_msdl()
        labels = list(atlas.labels)
        return atlas.maps, labels   # atlas.maps is 4D

    @staticmethod
    def _fetch_schaefer(n_rois: int = 100) -> Tuple[object, List[str]]:
        atlas = datasets.fetch_atlas_schaefer_2018(n_rois=n_rois)
        labels = [lbl.decode() if isinstance(lbl, bytes) else lbl
                  for lbl in atlas.labels]
        return atlas.maps, labels   # atlas.maps is 3D label image
