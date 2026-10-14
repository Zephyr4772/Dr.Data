"""Metrics, splitting strategies, the candidate model zoo and a cross-validation loop that records everything."""
from __future__ import annotations

import pickle
import time

import numpy as np
from sklearn import metrics as M
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold, TimeSeriesSplit
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .core import METRIC_INFO, SEED

HIGHER = {k: v[1] for k, v in METRIC_INFO.items()}


# ----------------------------------------------------------------------------- metrics
def _fbeta(y, pred, beta, pos, k):
    if k == 2:
        return M.fbeta_score(y, pred, beta=beta, pos_label=pos, zero_division=0)
    return M.fbeta_score(y, pred, beta=beta, average="macro", zero_division=0)


def score(metric, y, pred=None, proba=None, pos=None, k=2):
    try:
        if metric == "balanced_accuracy":
            return M.balanced_accuracy_score(y, pred)
        if metric == "accuracy":
            return M.accuracy_score(y, pred)
        if metric == "f1":
            return M.f1_score(y, pred, average="macro", zero_division=0)
        if metric == "f2":
            return _fbeta(y, pred, 2, pos, k)
        if metric == "f05":
            return _fbeta(y, pred, .5, pos, k)
        if metric == "recall":
            return M.recall_score(y, pred, pos_label=pos, zero_division=0) if k == 2 else M.recall_score(y, pred, average="macro", zero_division=0)
        if metric == "precision":
            return M.precision_score(y, pred, pos_label=pos, zero_division=0) if k == 2 else M.precision_score(y, pred, average="macro", zero_division=0)
        if metric == "roc_auc":
            if proba is None:
                return np.nan
            if k == 2:
                return M.roc_auc_score(y == pos, proba[:, pos])
            return M.roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=list(range(k)))
        if metric == "pr_auc":
            if proba is None or k != 2:
                return np.nan
            return M.average_precision_score(y == pos, proba[:, pos])
        if metric == "mae":
            return M.mean_absolute_error(y, pred)
        if metric == "rmse":
            return float(np.sqrt(M.mean_squared_error(y, pred)))
        if metric == "mape":
            return float(np.mean(np.abs(y - pred) / np.maximum(np.abs(y), 1e-9)))
        if metric == "r2":
            return M.r2_score(y, pred)
    except ValueError:
        return np.nan
    raise KeyError(metric)


def metric_set(task, k):
    if task == "regression":
        return ["mae", "rmse", "r2", "mape"]
    s = ["balanced_accuracy", "accuracy", "f1", "roc_auc"]
    if k == 2:
        s += ["recall", "precision", "f2", "f05", "pr_auc"]
    return s


def baseline_score(metric, y, task, pos=None, k=2):
    """Best score achievable WITHOUT looking at any feature (constant / random-by-prior predictions)."""
    y = np.asarray(y)
    if task == "regression":
        cands = [np.full_like(y, y.mean(), dtype=float), np.full_like(y, np.median(y), dtype=float)]
        vals = [score(metric, y, c) for c in cands]
        return max(vals) if HIGHER[metric] else min(vals)
    if metric == "roc_auc":
        return 0.5
    if metric == "pr_auc":
        return float(np.mean(y == pos))
    counts = np.bincount(y, minlength=k)
    rng = np.random.default_rng(SEED)
    cands = [np.full_like(y, counts.argmax())]
    if k == 2:
        cands.append(np.full_like(y, pos))
    cands += [rng.choice(k, len(y), p=counts / counts.sum()) for _ in range(3)]
    return max(score(metric, y, c, pos=pos, k=k) for c in cands)


def skill(metric, value, base):
    """0 = no better than the feature-free baseline, 1 = perfect."""
    if value is None or not np.isfinite(value):
        return np.nan
    if HIGHER[metric]:
        top = 1.0
        return (value - base) / (top - base) if top > base else 0.0
    return 1 - value / base if base > 0 else 0.0


# ----------------------------------------------------------------------------- splits
def splitter(kind, task, n_splits=5, y=None, groups=None):
    """kind: random | time | group. Returns a list of (train_idx, test_idx)."""
    n = len(y)
    if kind == "time":
        return list(TimeSeriesSplit(n_splits=n_splits).split(np.zeros(n)))
    if kind == "group":
        if task == "classification":
            try:
                return list(StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED).split(np.zeros(n), y, groups))
            except ValueError:
                pass
        return list(GroupKFold(n_splits=n_splits).split(np.zeros(n), y, groups))
    if task == "classification":
        k = int(min(n_splits, max(2, np.bincount(y).min())))
        return list(StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED).split(np.zeros(n), y))
    return list(KFold(n_splits=n_splits, shuffle=True, random_state=SEED).split(np.zeros(n)))


# ----------------------------------------------------------------------------- model zoo
# interpretability: 3 = a human can read the whole model, 2 = explainable with tools, 1 = black box
def zoo(task, balanced=False, skewed_target=False, fast=False):
    cw = "balanced" if balanced else None
    if task == "classification":
        z = {
            "Logistic regression": dict(model=make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0, class_weight=cw)),
                                        family="linear", interp=3,
                                        about="Weighted sum of features. Fast, stable, coefficients are readable; misses interactions and non-linear effects."),
            "Decision tree": dict(model=DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, class_weight=cw, random_state=SEED),
                                  family="tree", interp=3,
                                  about="A flowchart of if/else rules (depth 5). Fully readable, but high variance and usually less accurate."),
            "k-nearest neighbours": dict(model=make_pipeline(StandardScaler(), KNeighborsClassifier(15)),
                                         family="instance", interp=2,
                                         about="Predicts from the 15 most similar training rows. No training, slow at prediction, suffers in high dimensions."),
            "Random forest": dict(model=RandomForestClassifier(200 if not fast else 80, min_samples_leaf=2, class_weight=cw, n_jobs=-1, random_state=SEED),
                                  family="ensemble", interp=2,
                                  about="Average of many decorrelated trees. Robust default with little tuning; large model, slower inference."),
            "Gradient boosting": dict(model=HistGradientBoostingClassifier(max_iter=300, learning_rate=.08, early_stopping=True,
                                                                           class_weight=cw, random_state=SEED),
                                      family="boosting", interp=2,
                                      about="Trees added sequentially to fix previous errors. Usually the strongest model on tabular data."),
            "Neural network (MLP)": dict(model=make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True, max_iter=300,
                                                                                            random_state=SEED)),
                                         family="neural", interp=1,
                                         about="Two hidden layers (64, 32). Flexible, but data-hungry on tabular data and needs scaling + tuning."),
        }
    else:
        z = {
            "Ridge regression": dict(model=make_pipeline(StandardScaler(), Ridge(1.0)), family="linear", interp=3,
                                     about="Regularised linear model. Fast and readable; cannot capture non-linear effects or interactions."),
            "Decision tree": dict(model=DecisionTreeRegressor(max_depth=6, min_samples_leaf=10, random_state=SEED), family="tree", interp=3,
                                  about="Piecewise-constant rules (depth 6). Readable, coarse predictions, high variance."),
            "k-nearest neighbours": dict(model=make_pipeline(StandardScaler(), KNeighborsRegressor(15)), family="instance", interp=2,
                                         about="Averages the 15 most similar rows. Simple, slow at prediction, poor in high dimensions."),
            "Random forest": dict(model=RandomForestRegressor(200 if not fast else 80, min_samples_leaf=2, n_jobs=-1, random_state=SEED),
                                  family="ensemble", interp=2,
                                  about="Average of many trees. Robust default; large model, cannot extrapolate beyond seen target values."),
            "Gradient boosting": dict(model=HistGradientBoostingRegressor(max_iter=300, learning_rate=.08, early_stopping=True, random_state=SEED),
                                      family="boosting", interp=2,
                                      about="Sequentially fitted trees. Usually the strongest tabular regressor."),
            "Neural network (MLP)": dict(model=TransformedTargetRegressor(
                                             make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 32), early_stopping=True,
                                                                                          max_iter=400, random_state=SEED)),
                                             transformer=StandardScaler()),
                                         family="neural", interp=1,
                                         about="Two hidden layers (64, 32). Needs scaled inputs and targets; data-hungry."),
        }
        if skewed_target:
            z["Gradient boosting (log target)"] = dict(
                model=TransformedTargetRegressor(HistGradientBoostingRegressor(max_iter=300, learning_rate=.08, early_stopping=True,
                                                                               random_state=SEED),
                                                 func=np.log1p, inverse_func=np.expm1),
                family="boosting", interp=2,
                about="Same booster trained on log(1+target): tames a long right tail so large values don't dominate the loss.")
    return z


def dummy(task):
    return DummyClassifier(strategy="prior") if task == "classification" else DummyRegressor(strategy="mean")


# ----------------------------------------------------------------------------- CV loop
def cv_eval(model, X, y, splits, task, metrics, pos=None, k=2, keep_oof=True):
    """Fit/score a model over the given splits. X: numpy array. Returns per-metric fold scores + timings + OOF."""
    n = len(y)
    oof_pred = np.full(n, np.nan) if task == "regression" else np.full(n, -1)
    oof_proba = np.full((n, k), np.nan) if task == "classification" else None
    folds = {m: [] for m in metrics}
    fit_t, pred_t, size = [], [], 0
    for tr, te in splits:
        m = clone(model)
        t0 = time.perf_counter()
        m.fit(X[tr], y[tr])
        fit_t.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        pred = m.predict(X[te])
        pred_t.append((time.perf_counter() - t0) / max(1, len(te)))
        proba = None
        if task == "classification" and hasattr(m, "predict_proba"):
            pr = m.predict_proba(X[te])
            proba = np.zeros((len(te), k))
            cls = m.classes_ if hasattr(m, "classes_") else m[-1].classes_
            proba[:, cls] = pr
            oof_proba[te] = proba
        oof_pred[te] = pred
        for met in metrics:
            folds[met].append(score(met, y[te], pred, proba, pos, k))
    try:
        size = len(pickle.dumps(m))
    except Exception:
        size = 0
    tested = np.unique(np.concatenate([te for _, te in splits]))
    out = {"folds": {mm: [float(v) for v in vals] for mm, vals in folds.items()},
           "mean": {mm: float(np.nanmean(v)) if np.isfinite(v).any() else None for mm, v in folds.items()},
           "std": {mm: float(np.nanstd(v)) if np.isfinite(v).any() else None for mm, v in folds.items()},
           "fit_time": float(np.mean(fit_t)), "train_rows": int(np.mean([len(tr) for tr, _ in splits])),
           "latency_ms": float(np.mean(pred_t) * 1000), "size_kb": size / 1024, "tested": tested}
    if keep_oof:
        out["oof_pred"], out["oof_proba"] = oof_pred, oof_proba
    return out
