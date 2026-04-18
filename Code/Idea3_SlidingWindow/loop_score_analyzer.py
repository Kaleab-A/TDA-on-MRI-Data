"""
Code/Idea3_SlidingWindow/loop_score_analyzer.py
Derives scalar biomarkers from temporal loop score sequences.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from scipy import stats

from Core.utils import OutputManager, StatisticsHelper, SubjectRecord
from Parameters.params_idea3 import Idea3Params


class LoopScoreAnalyzer:
    """
    Computes and analyzes H1 loop scores over time windows.

    Loop score = aggregated H1 bar lifetimes per window
    (sum / max / mean, controlled by params.loop_score_aggregation).
    """

    def __init__(self, params: Idea3Params, output_manager: OutputManager):
        self.params = params
        self.output_manager = output_manager

    # ------------------------------------------------------------------
    # Loop score computation
    # ------------------------------------------------------------------

    def compute_loop_score(self, window_diagrams: List[dict]) -> np.ndarray:
        """
        Aggregate H1 bar lifetimes per window into a scalar loop score.

        Returns
        -------
        np.ndarray, shape (n_windows,)
        """
        from Code.Idea3_SlidingWindow.window_ph_computer import WindowPHComputer
        from Core.base_tda import BasePersistenceComputer as BPC
        scores = []
        for result in window_diagrams:
            if len(result["dgms"]) < 2:
                scores.append(0.0)
                continue
            h1 = result["dgms"][1].copy()
            h1 = BPC.strip_infinite_bars(h1)
            lifetimes = BPC.compute_lifetimes(h1)
            if lifetimes.size == 0:
                scores.append(0.0)
            elif self.params.loop_score_aggregation == "sum":
                scores.append(float(np.sum(lifetimes)))
            elif self.params.loop_score_aggregation == "max":
                scores.append(float(np.max(lifetimes)))
            else:  # 'mean'
                scores.append(float(np.mean(lifetimes)))
        return np.array(scores)

    # ------------------------------------------------------------------
    # Experiment 1: Temporal H1 evolution
    # ------------------------------------------------------------------

    def temporal_h1_evolution_experiment(
        self,
        loop_scores_per_subject: List[np.ndarray],
        labels: np.ndarray,
    ) -> dict:
        """
        Compute group-level mean ± std of loop score over time.
        Subjects with different n_windows are handled by truncating to minimum.
        """
        adhd_mask = labels == 1
        adhd_scores = [s for s, m in zip(loop_scores_per_subject, adhd_mask) if m]
        ctrl_scores = [s for s, m in zip(loop_scores_per_subject, adhd_mask) if not m]

        def _group_stats(scores_list):
            if not scores_list:
                return {}, {}
            min_len = min(len(s) for s in scores_list)
            arr = np.vstack([s[:min_len] for s in scores_list])
            return arr.mean(axis=0), arr.std(axis=0)

        adhd_mean, adhd_std = _group_stats(adhd_scores)
        ctrl_mean, ctrl_std = _group_stats(ctrl_scores)

        result = {
            "adhd_mean": adhd_mean,
            "adhd_std": adhd_std,
            "ctrl_mean": ctrl_mean,
            "ctrl_std": ctrl_std,
        }
        return result

    # ------------------------------------------------------------------
    # Experiment 2: Loop score as biomarker
    # ------------------------------------------------------------------

    def loop_score_as_biomarker_experiment(
        self,
        loop_scores: List[np.ndarray],
        labels: np.ndarray,
    ) -> dict:
        """
        Compute per-subject summary statistics of loop score
        and test ADHD vs control.
        """
        adhd_mask = labels == 1
        rows = []
        for i, scores in enumerate(loop_scores):
            rows.append({
                "subject_index": i,
                "adhd_label": int(labels[i]),
                "mean_loop_score": float(np.mean(scores)) if scores.size > 0 else 0.0,
                "std_loop_score": float(np.std(scores)) if scores.size > 0 else 0.0,
                "max_loop_score": float(np.max(scores)) if scores.size > 0 else 0.0,
                "auc_loop_score": float(np.trapz(scores)) if scores.size > 0 else 0.0,
            })
        df = pd.DataFrame(rows)
        self.output_manager.save_dataframe(df, "loop_score_per_subject.csv")

        results = {}
        for metric in ("mean_loop_score", "std_loop_score",
                        "max_loop_score", "auc_loop_score"):
            adhd_vals = df.loc[df["adhd_label"] == 1, metric].values
            ctrl_vals = df.loc[df["adhd_label"] == 0, metric].values
            test = StatisticsHelper.mann_whitney_u(adhd_vals, ctrl_vals)
            test["metric"] = metric
            results[metric] = {
                "adhd": adhd_vals, "ctrl": ctrl_vals, "test": test
            }
            print(f"  {metric}: p={test['pvalue']:.4f}, d={StatisticsHelper.cohen_d(adhd_vals, ctrl_vals):.3f}")

        return {"results": results, "dataframe": df}

    # ------------------------------------------------------------------
    # Experiment 3: Window size sweep
    # ------------------------------------------------------------------

    def window_size_sweep_experiment(
        self,
        records: List[SubjectRecord],
        window_lengths: List[int],
    ) -> dict:
        """
        For each window length: re-embed, re-compute PH, aggregate loop score.
        Returns {window_length: {metric: (adhd_mean, ctrl_mean, pvalue)}}
        """
        from Code.Idea3_SlidingWindow.window_embedder import SlidingWindowEmbedder
        from Code.Idea3_SlidingWindow.window_ph_computer import WindowPHComputer

        labels = np.array([r.adhd_label for r in records])
        adhd_mask = labels == 1

        sweep_results = {}
        rows = []
        for w in window_lengths:
            print(f"  Window length = {w}...")
            embedder = SlidingWindowEmbedder(self.params)
            ph_computer = WindowPHComputer(self.params)

            all_windows = embedder.embed_all(records, window_length=w)
            all_diagrams = ph_computer.fit_transform_all_subjects(all_windows)
            loop_scores = [self.compute_loop_score(d) for d in all_diagrams]

            adhd_means = [float(np.mean(s)) for s, m in zip(loop_scores, adhd_mask) if m and s.size > 0]
            ctrl_means = [float(np.mean(s)) for s, m in zip(loop_scores, adhd_mask) if not m and s.size > 0]

            if adhd_means and ctrl_means:
                test = StatisticsHelper.mann_whitney_u(
                    np.array(adhd_means), np.array(ctrl_means))
                rows.append({"window_length": w,
                              "adhd_mean": float(np.mean(adhd_means)),
                              "ctrl_mean": float(np.mean(ctrl_means)),
                              "pvalue": test["pvalue"]})
            sweep_results[w] = loop_scores

        df = pd.DataFrame(rows)
        self.output_manager.save_dataframe(df, "window_size_sweep.csv")
        return {"sweep_results": sweep_results, "dataframe": df}

    # ------------------------------------------------------------------
    # Experiment 4: Autocorrelation
    # ------------------------------------------------------------------

    def autocorrelation_experiment(
        self,
        loop_scores: List[np.ndarray],
        labels: np.ndarray,
    ) -> dict:
        """
        Compute ACF of each subject's loop score sequence.
        Returns group-level mean ACF and KS test result.
        """
        max_lag = self.params.autocorr_max_lag

        def acf(x, max_lag):
            n = len(x)
            x = x - x.mean()
            var = np.var(x)
            if var == 0:
                return np.zeros(max_lag + 1)
            result = [np.correlate(x, x, mode="full")[n - 1 - k] / (var * n)
                      for k in range(max_lag + 1)]
            return np.array(result)

        adhd_mask = labels == 1
        adhd_acfs, ctrl_acfs = [], []
        for scores, is_adhd in zip(loop_scores, adhd_mask):
            if scores.size > max_lag + 1:
                a = acf(scores, max_lag)
                if is_adhd:
                    adhd_acfs.append(a)
                else:
                    ctrl_acfs.append(a)

        results = {"adhd_acfs": adhd_acfs, "ctrl_acfs": ctrl_acfs}
        if adhd_acfs and ctrl_acfs:
            # KS test on mean ACF value at lag-1
            adhd_lag1 = np.array([a[1] for a in adhd_acfs])
            ctrl_lag1 = np.array([a[1] for a in ctrl_acfs])
            ks_stat, pvalue = stats.ks_2samp(adhd_lag1, ctrl_lag1)
            results["ks_statistic"] = float(ks_stat)
            results["ks_pvalue"] = float(pvalue)
            print(f"  ACF lag-1 KS test: stat={ks_stat:.4f}, p={pvalue:.4f}")

        return results
