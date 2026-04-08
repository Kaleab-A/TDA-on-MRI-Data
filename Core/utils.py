"""
Core/utils.py
Shared data structures, I/O helpers, and statistical utilities.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class SubjectRecord:
    """Holds all per-subject data, progressively filled by the pipeline."""

    subject_id: str
    func_path: str
    adhd_label: int          # 1 = ADHD, 0 = control
    age: float
    sex: int                 # 0 = female, 1 = male (nilearn convention)
    adhd_measure: float      # continuous ADHD index score

    # Populated after masking
    time_series: Optional[np.ndarray] = field(default=None, repr=False)

    # Populated after FC computation
    fc_matrix: Optional[np.ndarray] = field(default=None, repr=False)
    distance_matrix: Optional[np.ndarray] = field(default=None, repr=False)

    def is_adhd(self) -> bool:
        return self.adhd_label == 1

    def is_control(self) -> bool:
        return self.adhd_label == 0


# ---------------------------------------------------------------------------
# Output management
# ---------------------------------------------------------------------------

class OutputManager:
    """Handles Output/IdeaN/ directory creation and file path generation."""

    def __init__(self, idea_name: str, base_output_dir: Path):
        self.idea_name = idea_name
        self.base_output_dir = Path(base_output_dir)
        self.idea_dir = self.base_output_dir / idea_name
        self.ensure_dirs()

    def ensure_dirs(self) -> None:
        self.idea_dir.mkdir(parents=True, exist_ok=True)

    def _timestamped(self, filename: str) -> str:
        stem, *ext = filename.rsplit(".", 1)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = f".{ext[0]}" if ext else ""
        return f"{stem}_{ts}{suffix}"

    def get_plot_path(self, filename: str, timestamp: bool = False) -> Path:
        name = self._timestamped(filename) if timestamp else filename
        return self.idea_dir / name

    def get_csv_path(self, filename: str, timestamp: bool = False) -> Path:
        name = self._timestamped(filename) if timestamp else filename
        return self.idea_dir / name

    def save_dataframe(self, df: pd.DataFrame, filename: str,
                       timestamp: bool = False) -> Path:
        path = self.get_csv_path(filename, timestamp=timestamp)
        df.to_csv(path, index=False)
        print(f"  Saved: {path}")
        return path


# ---------------------------------------------------------------------------
# Statistics helper
# ---------------------------------------------------------------------------

class StatisticsHelper:
    """Static methods for group comparison statistics."""

    @staticmethod
    def mann_whitney_u(group_a: np.ndarray, group_b: np.ndarray) -> dict:
        """Mann-Whitney U test with effect size (rank-biserial correlation)."""
        stat, pvalue = stats.mannwhitneyu(group_a, group_b, alternative="two-sided")
        n_a, n_b = len(group_a), len(group_b)
        effect_size = 1 - (2 * stat) / (n_a * n_b)  # rank-biserial r
        return {"statistic": float(stat), "pvalue": float(pvalue),
                "effect_size": float(effect_size), "n_a": n_a, "n_b": n_b}

    @staticmethod
    def permutation_test(group_a: np.ndarray, group_b: np.ndarray,
                         n_permutations: int = 1000,
                         random_seed: int = 42) -> dict:
        """Permutation test on group mean difference."""
        rng = np.random.default_rng(random_seed)
        observed = np.mean(group_a) - np.mean(group_b)
        pooled = np.concatenate([group_a, group_b])
        n_a = len(group_a)
        null = np.array([
            np.mean(perm[:n_a]) - np.mean(perm[n_a:])
            for perm in (rng.permutation(pooled) for _ in range(n_permutations))
        ])
        pvalue = float(np.mean(np.abs(null) >= np.abs(observed)))
        return {"observed": float(observed), "pvalue": pvalue,
                "null_distribution": null}

    @staticmethod
    def cohen_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
        """Cohen's d effect size."""
        n_a, n_b = len(group_a), len(group_b)
        pooled_std = np.sqrt(
            ((n_a - 1) * np.var(group_a, ddof=1) + (n_b - 1) * np.var(group_b, ddof=1))
            / (n_a + n_b - 2)
        )
        if pooled_std == 0:
            return 0.0
        return float((np.mean(group_a) - np.mean(group_b)) / pooled_std)

    @staticmethod
    def spearman_correlation(x: np.ndarray, y: np.ndarray) -> dict:
        """Spearman correlation with p-value."""
        corr, pvalue = stats.spearmanr(x, y)
        return {"correlation": float(corr), "pvalue": float(pvalue)}

    @staticmethod
    def fdr_correct(pvalues: np.ndarray, alpha: float = 0.05) -> np.ndarray:
        """Benjamini-Hochberg FDR correction. Returns corrected p-values."""
        n = len(pvalues)
        order = np.argsort(pvalues)
        corrected = np.empty(n)
        cummin = np.inf
        for i in range(n - 1, -1, -1):
            val = n / (i + 1) * pvalues[order[i]]
            cummin = min(cummin, val)
            corrected[order[i]] = min(cummin, 1.0)
        return corrected
