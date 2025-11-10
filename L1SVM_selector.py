#!/usr/bin/env python3
"""
Iterative L1-SVM feature discovery per (modality, polarity, condition, time_point).

Algorithm (per 4×2×3×5 setting):
  For modality in {lipid, metabolite}
    For polarity in {+,-}
      For condition in {CHIKV, DENV, ZIKV}
        For time_point in {0,5,7,14,21}
          Repeat:
            • Query samples for (condition,time_point) vs (MOCK,time_point)
            • Group filter by condition×time_point with min_prop=0.2
            • Preprocess with two_step_label_agnostic + standardize
            • 5-fold CV fit L1-regularized SVM (proximal gradient)
            • Record union of selected (non-zero) features across folds
            • Compute mean CV accuracy
            • Mask selected features from X_df
          until mean accuracy < 0.75

Outputs:
  results/selected_features.csv with columns:
    modality, polarity, condition, time_point, feature_id, iteration, mean_cv_accuracy

Requirements:
  - query_and_preprocess.py in PYTHONPATH (same folder or installed)
"""
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score

from query_and_preprocess import (
    make_dataset,
    filter_groupwise_missingness,
    make_preprocess_pipeline,
)

class L1ProxSVM(BaseEstimator, ClassifierMixin):
    """
    Linear SVM with L1 penalty via proximal gradient on the squared hinge loss:

        L(w,b) = (1/n) * sum_i (max(0, 1 - y_i * (w·x_i + b)))^2  +  λ ||w||_1

    Prox step is applied to w only (b is unregularized).

    Stopping criterion (like prox_fit):
        stop when || [w_t - w_{t-1}; b_t - b_{t-1}] ||_2 <= delta_tol

    Parameters
    ----------
    lambda_ : float, default=1e-3
        L1 regularization strength (λ).
    step_size : float or None, default=None
        Fixed step size α. If None, uses α = 1 / (2 * R^2) with R = max_i ||x_i||_2.
    max_iter : int, default=2000
        Max proximal-gradient iterations.
    delta_tol : float, default=1e-6
        Tolerance on iterate change norm.
    fit_intercept : bool, default=True
        Whether to fit an intercept b.
    record_history : bool, default=False
        If True, records 'obj' and 'delta' per iteration in self.history_.

    Notes
    -----
    • Expects y in {+1, -1}.
    • Full-batch updates.
    """

    def __init__(self,
                 lambda_: float = 1e-3,
                 step_size: float | None = None,
                 max_iter: int = 2000,
                 delta_tol: float = 1e-6,
                 fit_intercept: bool = True,
                 record_history: bool = False):
        self.lambda_ = float(lambda_)
        self.step_size = None if step_size is None else float(step_size)
        self.max_iter = int(max_iter)
        self.delta_tol = float(delta_tol)
        self.fit_intercept = bool(fit_intercept)
        self.record_history = bool(record_history)

        # learned params
        self.coef_: np.ndarray | None = None  # shape (1, p)
        self.intercept_: float = 0.0
        self.history_: dict | None = None

    # ---- internals ----
    @staticmethod
    def _soft_threshold(w: np.ndarray, t: float) -> np.ndarray:
        return np.sign(w) * np.maximum(0.0, np.abs(w) - t)

    def _squared_hinge_grads(self, X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float):
        """
        ∇ of f(w,b) = (1/n) * sum (max(0, 1 - y*(Xw+b)))^2  (smooth part).
        """
        n = X.shape[0]
        f = X @ w + (b if self.fit_intercept else 0.0)
        m = 1.0 - y * f  # margins
        active = m > 0.0
        if np.any(active):
            X_a = X[active]
            y_a = y[active]
            m_a = m[active]
            grad_w = -(2.0 / n) * (X_a.T @ (y_a * m_a))
            grad_b = float(-(2.0 / n) * np.sum(y_a * m_a)) if self.fit_intercept else 0.0
        else:
            grad_w = np.zeros_like(w)
            grad_b = 0.0
        return grad_w, grad_b

    def _objective(self, X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> float:
        f = X @ w + (b if self.fit_intercept else 0.0)
        m = 1.0 - y * f
        sq_hinge = np.maximum(0.0, m) ** 2
        return float(sq_hinge.mean() + self.lambda_ * np.abs(w).sum())

    # ---- sklearn API ----
    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        n, p = X.shape

        # init
        w = np.zeros(p, dtype=np.float32)
        b = 0.0
        lam = self.lambda_

        # step size α
        if self.step_size is None:
            R = float(np.linalg.norm(X, axis=1).max()) if n else 1.0
            alpha = 1.0 / (2.0 * (R ** 2) + 1e-12)  # safe for squared hinge
        else:
            alpha = self.step_size

        if self.record_history:
            self.history_ = {"obj": [], "delta": []}

        for _ in range(self.max_iter):
            w_old, b_old = w.copy(), b

            # gradients of smooth part
            grad_w, grad_b = self._squared_hinge_grads(X, y, w, b)

            # PGD step: gradient step then prox on w
            w_bar = w - alpha * grad_w
            b_bar = b - alpha * grad_b if self.fit_intercept else b
            w = self._soft_threshold(w_bar, alpha * lam)
            b = b_bar

            # iterate-change norm
            delta = float(np.linalg.norm(np.r_[w - w_old, b - b_old]))

            if self.record_history:
                self.history_["delta"].append(delta)
                self.history_["obj"].append(self._objective(X, y, w, b))

            if delta <= self.delta_tol:
                break

        self.coef_ = w.reshape(1, -1)
        self.intercept_ = float(b)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        return X @ self.coef_.ravel() + (self.intercept_ if self.fit_intercept else 0.0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.where(self.decision_function(X) >= 0.0, 1, -1)
# ----------------------------- Utilities ------------------------------------

def ensure_cv_splits(y: np.ndarray, desired_splits: int = 5) -> StratifiedKFold:
    # Make sure each class has at least desired_splits examples
    _, counts = np.unique(y, return_counts=True)
    n_splits = min(desired_splits, counts.min())
    n_splits = max(n_splits, 2)  # need at least 2
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)


def binary_labels_for_pair(y_condition: np.ndarray, positive_label: str) -> np.ndarray:
    return np.where(y_condition == positive_label, 1, -1)


# ----------------------------- Main loop ------------------------------------

def run_iteration_block(root: str = "parquets",
                        modalities=("lipid", "metabolite"),
                        polarities=("+", "-"),
                        conditions=("CHIKV", "DENV", "ZIKV"),
                        time_points=(0, 5, 7, 14, 21),
                        acc_threshold: float = 0.75,
                        lambda_: float = 0.1,
                        delta_tol: float = 1e-5,
                        max_iter: int = 2000) -> pd.DataFrame:
    rows: List[Dict] = []

    for modality in modalities:
        for polarity in polarities:
            for condition in conditions:
                for tp in time_points:
                    # 1) Query {condition, MOCK} at this time point
                    try:
                        X_df, y_labels, S, _ = make_dataset(
                            root=root,
                            conditions=[condition, "MOCK"],
                            times=[tp],
                            omics=[modality],
                            polarity=[polarity],
                            label_col="condition",
                        )
                    except Exception as e:
                        print(f"[SKIP] {modality} {polarity} {condition} t={tp}: query error: {e}")
                        continue

                    if X_df.empty or len(np.unique(y_labels)) < 2:
                        print(f"[SKIP] {modality} {polarity} {condition} t={tp}: insufficient data/classes")
                        continue

                    # 2) Groupwise feature filter (condition×time)
                    try:
                        X_df = filter_groupwise_missingness(
                            X_df=X_df, samples_df=S,
                            group_cols=("condition", "time_point"),
                            min_prop=0.2, min_group_n=1, require_all_groups=False,
                        )
                    except Exception as e:
                        print(f"[WARN] group filter failed; continuing unfiltered: {e}")

                    if X_df.shape[1] == 0:
                        print(f"[SKIP] {modality} {polarity} {condition} t={tp}: no features after filter")
                        continue

                    y_bin = binary_labels_for_pair(y_labels, positive_label=condition)

                    # 3) Iteratively select & mask features
                    iteration = 1
                    remaining_cols = X_df.columns.tolist()

                    while True:
                        if len(remaining_cols) == 0:
                            break
                        X_iter = X_df[remaining_cols]

                        # Preprocess: two-step label-agnostic + std scale
                        pre = make_preprocess_pipeline(
                            imputer="two_step_label_agnostic", scale="standard"
                        )
                        clf = L1ProxSVM(lambda_=lambda_, max_iter=max_iter, delta_tol=delta_tol)

                        # Build a pipeline manually: fit preprocessor inside CV for leak-free
                        # We'll use cross_validate with return_estimator to collect coefs
                        from sklearn.pipeline import Pipeline as SkPipeline
                        pipe = SkPipeline([
                            ("prep", pre),
                            ("clf", clf),
                        ])

                        cv = ensure_cv_splits(y_bin, desired_splits=6)
                        cvres = cross_validate(
                            pipe,
                            X_iter.values, y_bin,
                            scoring="accuracy",
                            cv=cv,
                            return_estimator=True,
                            n_jobs=None,
                        )
                        mean_acc = float(np.mean(cvres["test_score"]))

                        # Collect selected (non-zero) features across folds
                        selected = set()
                        for est in cvres["estimator"]:
                            # Access inner classifier coef_ (after preprocessing)
                            w = est.named_steps["clf"].coef_.ravel()
                            nz = np.where(np.abs(w) > 1e-8)[0]
                            # Map nz indices back to column names (preprocessing preserves order)
                            for j in nz:
                                selected.add(remaining_cols[j])

                        if not selected:
                            print(f"[{modality} {polarity} {condition} t={tp}] iter {iteration}: no selected features; stopping.")
                            break

                        if mean_acc > acc_threshold:
                            # Record rows (one per feature)
                            for fid in sorted(selected):
                                rows.append({
                                    "modality": modality,
                                    "polarity": polarity,
                                    "condition": condition,
                                    "time_point": tp,
                                    "feature_id": fid,
                                    "iteration": iteration,
                                    "mean_cv_accuracy": mean_acc,
                                })

                        print(f"[{modality} {polarity} {condition} t={tp}] iter {iteration}: k={len(selected)} features, acc={mean_acc:.3f}")

                        # Stop if below threshold
                        if mean_acc < acc_threshold:
                            break

                        # Mask selected features for next iteration
                        remaining_cols = [c for c in remaining_cols if c not in selected]
                        iteration += 1

    # Build results DataFrame
    if rows:
        res = pd.DataFrame(rows)
        res.sort_values(["modality", "polarity", "condition", "time_point", "iteration", "feature_id"], inplace=True)
    else:
        res = pd.DataFrame(columns=[
            "modality", "polarity", "condition", "time_point", "feature_id", "iteration", "mean_cv_accuracy"
        ])
    return res


if __name__ == "__main__":

    os.makedirs(os.path.dirname("results/selected_features.csv"), exist_ok=True)
    res = run_iteration_block(root="parquets",
                              acc_threshold=0.7,
                              lambda_= 0.25,
                              delta_tol=1e-7,
                              max_iter=20000)
    res.to_csv("results/selected_features.csv", index=False)
    print(f"Saved: results/selected_features.csv ({len(res)} rows)")
