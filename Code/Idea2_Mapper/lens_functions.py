"""
Code/Idea2_Mapper/lens_functions.py
Lens (filter) functions that map time-point clouds to 1D values for Mapper.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from sklearn.decomposition import PCA


class BaseLens(ABC):
    """Maps (n_timepoints, n_features) → (n_timepoints, 1)."""

    @abstractmethod
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        X : np.ndarray, shape (n_timepoints, n_features)

        Returns
        -------
        np.ndarray, shape (n_timepoints, 1)  — normalized to [0, 1]
        """

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        """Scale a 1D vector to [0, 1]."""
        v_min, v_max = v.min(), v.max()
        if v_max - v_min < 1e-10:
            return np.zeros_like(v)
        return (v - v_min) / (v_max - v_min)


class PCALens(BaseLens):
    """Uses a single PCA component as the lens value."""

    def __init__(self, component: int = 0):
        self.component = component

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        pca = PCA(n_components=self.component + 1)
        scores = pca.fit_transform(X)  # (n_timepoints, n_components)
        lens = scores[:, self.component]
        return self._normalize(lens).reshape(-1, 1)


class VarianceLens(BaseLens):
    """
    Lens based on rolling variance of the mean BOLD signal.
    High values = high local temporal variability.
    """

    def __init__(self, window: int = 10):
        self.window = window

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        mean_bold = X.mean(axis=1)  # (n_timepoints,)
        n = len(mean_bold)
        variance = np.zeros(n)
        for t in range(n):
            start = max(0, t - self.window // 2)
            end = min(n, t + self.window // 2 + 1)
            variance[t] = np.var(mean_bold[start:end])
        return self._normalize(variance).reshape(-1, 1)


class TimeLens(BaseLens):
    """Lens is simply the (normalized) time index."""

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        return np.linspace(0, 1, n).reshape(-1, 1)


class LensFactory:
    """Creates lens instances from string names."""

    _registry = {
        "pca": PCALens,
        "variance": VarianceLens,
        "time": TimeLens,
    }

    @classmethod
    def create(cls, lens_name: str, **kwargs) -> BaseLens:
        """
        Parameters
        ----------
        lens_name : str
            One of 'pca', 'variance', 'time'.
        **kwargs
            Passed to the lens constructor.

        Returns
        -------
        BaseLens instance
        """
        name = lens_name.lower()
        if name not in cls._registry:
            raise ValueError(
                f"Unknown lens '{lens_name}'. "
                f"Available: {list(cls._registry.keys())}"
            )
        return cls._registry[name](**kwargs)
