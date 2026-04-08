"""
Code/Idea5_EulerCharacteristic/fda_analyzer.py
Functional Data Analysis on Betti/EC curves using skfda.
"""

from __future__ import annotations

from typing import List

import numpy as np

from Parameters.params_idea5 import Idea5Params


class FDAAnalyzer:
    """
    Applies Functional Data Analysis (skfda) to Betti/EC curves.

    Provides:
    - Conversion to FDataGrid
    - B-spline basis smoothing
    - Functional group-level hypothesis test (permutation-based)
    - Functional PCA
    """

    def __init__(self, params: Idea5Params, output_manager):
        self.params = params
        self.output_manager = output_manager

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def curves_to_functional_data(
        self, curves: np.ndarray, epsilon_values: np.ndarray
    ):
        """
        Convert (n_subjects, n_epsilon_steps) array to skfda.FDataGrid.

        Parameters
        ----------
        curves : np.ndarray, shape (n_subjects, n_steps)
        epsilon_values : np.ndarray, shape (n_steps,)

        Returns
        -------
        skfda.representation.FDataGrid
        """
        from skfda import FDataGrid
        return FDataGrid(
            data_matrix=curves,
            grid_points=epsilon_values,
        )

    def smooth_functional_data(self, fd):
        """
        Smooth functional data using B-spline basis expansion.

        Returns
        -------
        skfda.representation.FDataBasis
        """
        from skfda.representation.basis import BSpline
        from skfda.preprocessing.smoothing import BasisSmoother
        basis = BSpline(
            domain_range=fd.domain_range[0],
            n_basis=self.params.fda_n_basis,
        )
        smoother = BasisSmoother(
            basis=basis,
            smoothing_parameter=self.params.fda_smoothing_parameter,
        )
        return smoother.fit_transform(fd)

    def compute_functional_mean(self, fd, labels: np.ndarray) -> dict:
        """
        Compute mean functional curve per group.

        Returns
        -------
        {'adhd': FDataGrid, 'control': FDataGrid}
        """
        adhd_mask = labels == 1
        return {
            "adhd": fd[adhd_mask].mean(),
            "control": fd[~adhd_mask].mean(),
        }

    def functional_group_test(
        self, fd, labels: np.ndarray, output_manager=None
    ) -> dict:
        """
        Permutation-based functional group test (Hotelling T² or ANOVA).

        Uses skfda.inference if available, otherwise manual permutation.
        Returns {'statistic': float, 'pvalue': float}.
        """
        try:
            from skfda.inference.anova import oneway_anova
            statistic, pvalue = oneway_anova(fd, labels,
                                             n_reps=self.params.fda_n_permutations,
                                             random_state=self.params.random_seed)
        except Exception:
            # Fallback: permutation on integrated L2 difference
            adhd_mask = labels == 1
            observed = self._integrated_l2_diff(fd, adhd_mask)
            rng = np.random.default_rng(self.params.random_seed)
            null_dist = []
            perm_labels = labels.copy()
            for _ in range(self.params.fda_n_permutations):
                rng.shuffle(perm_labels)
                perm_mask = perm_labels == 1
                null_dist.append(self._integrated_l2_diff(fd, perm_mask))
            null_arr = np.array(null_dist)
            pvalue = float(np.mean(null_arr >= observed))
            statistic = float(observed)

        print(f"  FDA group test: stat={statistic:.4f}, p={pvalue:.4f}")
        return {"statistic": float(statistic), "pvalue": float(pvalue)}

    def functional_pca(self, fd, n_components: int = 3) -> dict:
        """
        Functional PCA on the Betti/EC curves.

        Returns
        -------
        dict with 'scores' (n_subjects, n_components),
                  'explained_variance_ratio' (n_components,),
                  'components' (FDataGrid)
        """
        from skfda.preprocessing.dim_reduction import FPCA
        fpca = FPCA(n_components=n_components)
        scores = fpca.fit_transform(fd)
        evr = fpca.explained_variance_ratio_
        print(f"  FPCA: explained variance = "
              f"{[f'{v:.2%}' for v in evr]}")
        return {
            "scores": scores,
            "explained_variance_ratio": evr,
            "components": fpca.components_,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _integrated_l2_diff(fd, adhd_mask: np.ndarray) -> float:
        """Integrated squared difference between group mean curves."""
        if adhd_mask.sum() == 0 or (~adhd_mask).sum() == 0:
            return 0.0
        adhd_mean = fd[adhd_mask].mean()
        ctrl_mean = fd[~adhd_mask].mean()
        diff = adhd_mean - ctrl_mean
        return float(np.trapz(diff.data_matrix[0, :, 0] ** 2,
                               diff.grid_points[0]))
