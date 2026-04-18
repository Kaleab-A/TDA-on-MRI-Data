"""
Code/Idea4_Classification/permutation_tester.py
Permutation tests for classification significance.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, permutation_test_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from Core.utils import OutputManager
from Parameters.params_idea4 import Idea4Params


class PermutationTester:
    """
    Assesses statistical significance of classifier accuracy via
    label permutation tests (sklearn.model_selection.permutation_test_score).
    """

    def __init__(self, params: Idea4Params, output_manager: OutputManager):
        self.params = params
        self.output_manager = output_manager

    # ------------------------------------------------------------------
    # Single test
    # ------------------------------------------------------------------

    def test_classifier(
        self,
        X: np.ndarray,
        y: np.ndarray,
        classifier=None,
        n_permutations: int = None,
        scoring: str = "accuracy",
    ) -> dict:
        """
        Run permutation_test_score for one (feature_set, classifier) pair.

        Returns
        -------
        dict with 'true_score', 'pvalue', 'null_distribution'
        """
        if classifier is None:
            classifier = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    max_iter=1000, random_state=self.params.random_seed,
                    class_weight="balanced",
                )),
            ])
        n_perm = n_permutations or self.params.n_permutations
        cv = StratifiedKFold(
            n_splits=self.params.cv_folds,
            shuffle=True,
            random_state=self.params.random_seed,
        )
        true_score, perm_scores, pvalue = permutation_test_score(
            classifier, X, y,
            scoring=scoring,
            cv=cv,
            n_permutations=n_perm,
            random_state=self.params.random_seed,
            n_jobs=-1,
        )
        print(f"    True score: {true_score:.4f}, p-value: {pvalue:.4f}")
        return {
            "true_score": float(true_score),
            "pvalue": float(pvalue),
            "null_distribution": perm_scores,
        }

    # ------------------------------------------------------------------
    # All feature sets
    # ------------------------------------------------------------------

    def run_all_permutation_tests(
        self,
        feature_sets: Dict[str, np.ndarray],
        y: np.ndarray,
    ) -> pd.DataFrame:
        """
        Test each non-empty feature set with logistic regression.

        Returns
        -------
        DataFrame with columns: feature_set, true_score, pvalue
        """
        rows = []
        for fs_name, X in feature_sets.items():
            if X.shape[1] == 0:
                print(f"  Skipping permutation test for '{fs_name}' — empty.")
                continue
            print(f"  Permutation test: {fs_name}...")
            result = self.test_classifier(X, y)
            rows.append({
                "feature_set": fs_name,
                "true_score": result["true_score"],
                "pvalue": result["pvalue"],
            })
        df = pd.DataFrame(rows)
        self.output_manager.save_dataframe(df, "permutation_test_results.csv")
        return df
