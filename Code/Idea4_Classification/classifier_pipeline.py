"""
Code/Idea4_Classification/classifier_pipeline.py
Standardized cross-validation pipeline for ADHD classification.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from Parameters.params_idea4 import Idea4Params


class ClassifierPipeline:
    """
    Wraps sklearn classifiers in standardized cross-validation pipelines.

    Each classifier is embedded in Pipeline([StandardScaler, classifier])
    so that scaling (and feature extraction if needed) is always fit
    only on training data — no leakage.
    """

    def __init__(self, params: Idea4Params):
        self.params = params

    # ------------------------------------------------------------------
    # Classifier factory
    # ------------------------------------------------------------------

    def _build_classifiers(self) -> Dict[str, Pipeline]:
        classifiers = {}
        if "logistic_regression" in self.params.classifiers:
            classifiers["logistic_regression"] = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    max_iter=1000, random_state=self.params.random_seed,
                    class_weight="balanced",
                )),
            ])
        if "svm" in self.params.classifiers:
            classifiers["svm"] = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(
                    kernel=self.params.svm_kernel,
                    probability=True,
                    class_weight="balanced",
                    random_state=self.params.random_seed,
                )),
            ])
        if "random_forest" in self.params.classifiers:
            classifiers["random_forest"] = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", RandomForestClassifier(
                    n_estimators=self.params.rf_n_estimators,
                    class_weight="balanced",
                    random_state=self.params.random_seed,
                )),
            ])
        return classifiers

    # ------------------------------------------------------------------
    # Cross-validation
    # ------------------------------------------------------------------

    def run_cv(
        self, X: np.ndarray, y: np.ndarray, classifier_name: str
    ) -> dict:
        """
        Stratified k-fold CV for one classifier.

        Returns dict with fold scores for accuracy, roc_auc, f1.
        """
        clf = self._build_classifiers()[classifier_name]
        cv = StratifiedKFold(
            n_splits=self.params.cv_folds,
            shuffle=True,
            random_state=self.params.random_seed,
        )
        scoring = ["accuracy", "roc_auc", "f1"]
        results = cross_validate(clf, X, y, cv=cv, scoring=scoring,
                                 return_train_score=False)
        return results

    def run_all_classifiers(
        self, X: np.ndarray, y: np.ndarray
    ) -> pd.DataFrame:
        """
        Run CV for all classifiers in params.classifiers.

        Returns DataFrame: rows=classifier, cols=metric_mean/metric_std.
        """
        rows = []
        for clf_name in self.params.classifiers:
            results = self.run_cv(X, y, clf_name)
            row = {"classifier": clf_name}
            for metric in ("test_accuracy", "test_roc_auc", "test_f1"):
                row[f"{metric}_mean"] = float(np.mean(results[metric]))
                row[f"{metric}_std"] = float(np.std(results[metric]))
            rows.append(row)
            print(f"  {clf_name}: "
                  f"acc={row['test_accuracy_mean']:.3f}±{row['test_accuracy_std']:.3f}, "
                  f"AUC={row['test_roc_auc_mean']:.3f}±{row['test_roc_auc_std']:.3f}")
        return pd.DataFrame(rows)

    def compare_feature_sets(
        self, feature_sets: Dict[str, np.ndarray], y: np.ndarray
    ) -> pd.DataFrame:
        """
        Run all classifiers on each feature set.

        Parameters
        ----------
        feature_sets : {'tda': X_tda, 'fc': X_fc, 'combined': X_combined}

        Returns
        -------
        DataFrame: rows=(feature_set, classifier), cols=metrics
        """
        all_rows = []
        for fs_name, X in feature_sets.items():
            if X.shape[1] == 0:
                print(f"  Skipping '{fs_name}' — empty feature matrix.")
                continue
            print(f"\n  Feature set: {fs_name} ({X.shape[1]} features)")
            df = self.run_all_classifiers(X, y)
            df.insert(0, "feature_set", fs_name)
            all_rows.append(df)
        return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def get_feature_importances(
        self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None
    ) -> pd.DataFrame:
        """
        Train a Random Forest on all data and return feature importances.
        """
        rf = RandomForestClassifier(
            n_estimators=self.params.rf_n_estimators,
            class_weight="balanced",
            random_state=self.params.random_seed,
        )
        rf.fit(StandardScaler().fit_transform(X), y)
        importances = rf.feature_importances_
        names = feature_names or [f"feat_{i}" for i in range(len(importances))]
        df = pd.DataFrame({"feature": names, "importance": importances})
        df = df.sort_values("importance", ascending=False).reset_index(drop=True)
        return df
