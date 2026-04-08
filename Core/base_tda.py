"""
Core/base_tda.py
Abstract base classes for TDA computations — consistent interface across all Ideas.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np


class BasePersistenceComputer(ABC):
    """
    Abstract base for anything that takes point clouds / distance matrices
    and returns persistence diagrams.
    """

    @abstractmethod
    def fit_transform(self, X) -> list:
        """
        Parameters
        ----------
        X : list of np.ndarray
            Point clouds (n_pts, n_features) or distance matrices (n_pts, n_pts).

        Returns
        -------
        list
            One diagram result per input array.
        """

    @abstractmethod
    def get_diagram_for_dimension(self, diagrams: list, dim: int) -> List[np.ndarray]:
        """
        Extract dim-specific persistence diagram from raw results.

        Returns
        -------
        List[np.ndarray]
            Each element is (n_bars, 2) array of [birth, death] pairs,
            with infinite death values already removed.
        """

    @staticmethod
    def strip_infinite_bars(diagram: np.ndarray) -> np.ndarray:
        """Remove bars with infinite death value (ripser convention)."""
        if diagram.size == 0:
            return diagram
        finite_mask = np.isfinite(diagram[:, 1])
        return diagram[finite_mask]

    @staticmethod
    def compute_lifetimes(diagram: np.ndarray) -> np.ndarray:
        """Return death - birth for each finite bar."""
        if diagram.size == 0:
            return np.array([])
        return diagram[:, 1] - diagram[:, 0]

    @staticmethod
    def total_persistence(diagram: np.ndarray, power: float = 1.0) -> float:
        """Sum of bar lifetimes raised to power."""
        lifetimes = BasePersistenceComputer.compute_lifetimes(diagram)
        if lifetimes.size == 0:
            return 0.0
        return float(np.sum(np.abs(lifetimes) ** power))


class BaseTDAExperiment(ABC):
    """Abstract base for a single named experiment within an Idea."""

    def __init__(self, params, output_manager):
        self.params = params
        self.output_manager = output_manager

    @abstractmethod
    def run(self, records: list) -> dict:
        """Execute the experiment. Returns a dict of metrics/results."""

    @abstractmethod
    def save_results(self, results: dict) -> None:
        """Save CSVs and plots to Output/IdeaN/."""
