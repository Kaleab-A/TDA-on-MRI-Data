"""
Code/Idea4_Classification/feature_extractor.py
Extracts fixed-length feature vectors from persistence diagrams.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from Core.utils import SubjectRecord
from Parameters.params_idea4 import Idea4Params


class TopologicalFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Transforms persistence diagrams into fixed-length feature vectors.

    Implements sklearn TransformerMixin so it can be embedded inside
    a Pipeline (ensuring fit-only-on-training-data in CV loops).

    Supported feature types:
    - Total persistence per dimension
    - Persistence entropy per dimension
    - Persistence landscapes (gudhi.representations.Landscape)
    - Persistence images (gudhi.representations.PersistenceImage)
    """

    def __init__(self, params: Idea4Params):
        self.params = params
        self._landscape_transformers: Dict[int, object] = {}
        self._image_transformers: Dict[int, object] = {}
        self._fitted = False

    # ------------------------------------------------------------------
    # sklearn TransformerMixin interface
    # ------------------------------------------------------------------

    def fit(self, diagrams_per_dim: Dict[int, List[np.ndarray]], y=None):
        """
        Fit landscape and image transformers on training diagrams.

        diagrams_per_dim : {dim: List[np.ndarray (n_bars, 2)]}
        """
        from gudhi.representations import Landscape, PersistenceImage
        for dim, diagrams in diagrams_per_dim.items():
            if self.params.use_persistence_landscapes:
                ls = Landscape(
                    num_landscapes=self.params.landscape_n_layers,
                    resolution=self.params.landscape_n_bins,
                )
                ls.fit(diagrams)
                self._landscape_transformers[dim] = ls
            if self.params.use_persistence_images:
                pi = PersistenceImage(
                    bandwidth=self.params.image_sigma,
                    resolution=[self.params.image_n_bins, self.params.image_n_bins],
                )
                pi.fit(diagrams)
                self._image_transformers[dim] = pi
        self._fitted = True
        return self

    def transform(self, diagrams_per_dim: Dict[int, List[np.ndarray]], y=None) -> np.ndarray:
        """Return concatenated feature matrix (n_subjects, n_features)."""
        return self.transform_all(diagrams_per_dim)

    # ------------------------------------------------------------------
    # Individual feature extraction
    # ------------------------------------------------------------------

    def transform_landscapes(
        self, diagrams_per_dim: Dict[int, List[np.ndarray]]
    ) -> np.ndarray:
        """Returns (n_subjects, n_layers * n_bins * n_dims)."""
        parts = []
        for dim, diagrams in diagrams_per_dim.items():
            if dim in self._landscape_transformers:
                feat = self._landscape_transformers[dim].transform(diagrams)
                parts.append(feat)
        return np.hstack(parts) if parts else np.empty((0, 0))

    def transform_persistence_images(
        self, diagrams_per_dim: Dict[int, List[np.ndarray]]
    ) -> np.ndarray:
        """Returns (n_subjects, n_bins^2 * n_dims)."""
        parts = []
        for dim, diagrams in diagrams_per_dim.items():
            if dim in self._image_transformers:
                feat = self._image_transformers[dim].transform(diagrams)
                parts.append(feat.reshape(len(diagrams), -1))
        return np.hstack(parts) if parts else np.empty((0, 0))

    def transform_total_persistence(
        self,
        diagrams_per_dim: Dict[int, List[np.ndarray]],
        power: float = 1.0,
    ) -> np.ndarray:
        """Returns (n_subjects, n_dims)."""
        from Core.base_tda import BasePersistenceComputer as BPC
        cols = []
        for dim, diagrams in sorted(diagrams_per_dim.items()):
            col = np.array([BPC.total_persistence(d, power) for d in diagrams])
            cols.append(col.reshape(-1, 1))
        return np.hstack(cols) if cols else np.empty((0, 0))

    def transform_persistence_entropy(
        self, diagrams_per_dim: Dict[int, List[np.ndarray]]
    ) -> np.ndarray:
        """Returns (n_subjects, n_dims)."""
        from gudhi.representations import Entropy
        cols = []
        for dim, diagrams in sorted(diagrams_per_dim.items()):
            ent = Entropy()
            ent.fit(diagrams)
            feat = ent.transform(diagrams)
            cols.append(feat.reshape(-1, 1))
        return np.hstack(cols) if cols else np.empty((0, 0))

    def transform_all(
        self, diagrams_per_dim: Dict[int, List[np.ndarray]]
    ) -> np.ndarray:
        """Concatenate all enabled feature types horizontally."""
        parts = []
        if self.params.use_total_persistence:
            parts.append(self.transform_total_persistence(diagrams_per_dim))
        if self.params.use_persistence_entropy:
            parts.append(self.transform_persistence_entropy(diagrams_per_dim))
        if self.params.use_persistence_landscapes and self._fitted:
            parts.append(self.transform_landscapes(diagrams_per_dim))
        if self.params.use_persistence_images and self._fitted:
            parts.append(self.transform_persistence_images(diagrams_per_dim))
        valid = [p for p in parts if p.size > 0 and p.ndim == 2]
        return np.hstack(valid) if valid else np.empty((len(next(iter(diagrams_per_dim.values()))), 0))

    # ------------------------------------------------------------------
    # FC baseline features
    # ------------------------------------------------------------------

    def get_fc_features(self, records: List[SubjectRecord]) -> np.ndarray:
        """
        Upper triangle of each subject's FC matrix → flattened vector.
        Returns (n_subjects, n_rois*(n_rois-1)//2).
        """
        rows = []
        for rec in records:
            if rec.fc_matrix is None:
                raise ValueError(
                    f"Subject {rec.subject_id} has no fc_matrix. "
                    "Run FCMatrixBuilder.transform() first."
                )
            n = rec.fc_matrix.shape[0]
            idx = np.triu_indices(n, k=1)
            rows.append(rec.fc_matrix[idx])
        return np.vstack(rows)
