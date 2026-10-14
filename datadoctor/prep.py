"""Turning the raw table + use case into an analysis-ready frame and numeric feature matrices."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .core import MAX_MODEL_ROWS, SEED, GOALS, UseCase


def norm_label(v: str) -> str:
    return re.sub(r"[^0-9a-z]", "", str(v).lower())


def category_variants(s: pd.Series):
    """Groups of spellings that collapse to the same normalised value, e.g. 'PayPal' / 'paypal' / ' PayPal '."""
    vc = s.dropna().astype(str).value_counts()
    if len(vc) > 5000:
        return {}
    groups = {}
    for val, cnt in vc.items():
        groups.setdefault(norm_label(val), []).append((val, int(cnt)))
    return {k: v for k, v in groups.items() if len(v) > 1 and k}


@dataclass
class Prepared:
    df: pd.DataFrame                 # analysis frame (target non-missing, variants merged, exact dups dropped)
    uc: UseCase
    task: str
    y: np.ndarray
    classes: list = field(default_factory=list)
    pos: int | None = None           # index of the positive class (binary)
    t: pd.Series | None = None       # parsed time column
    g: pd.Series | None = None       # entity ids
    features: list = field(default_factory=list)   # raw columns usable as features
    excluded: dict = field(default_factory=dict)   # column -> reason
    repairs: list = field(default_factory=list)    # what we changed for the analysis
    n_raw: int = 0
    metric: str = ""
    goal: str = ""


def resolve_task(s: pd.Series, requested: str, kind: str):
    if requested in ("classification", "regression"):
        return requested, ""
    num = pd.to_numeric(s, errors="coerce")
    if kind in ("categorical", "boolean") or num.notna().mean() < .95:
        return "classification", "target is categorical"
    nun = s.nunique()
    if nun <= 10 and np.all(np.mod(num.dropna(), 1) == 0):
        return "classification", f"target has only {nun} distinct integer values; treated as classes (pick regression if it is an ordinal score)"
    return "regression", "target is a continuous number"


def prepare(df: pd.DataFrame, prof: dict, uc: UseCase) -> Prepared:
    cols = {c.lower().strip(): c for c in df.columns}
    if uc.target not in df.columns:
        if uc.target.lower().strip() in cols:
            uc.target = cols[uc.target.lower().strip()]
        else:
            raise ValueError(f"Target column '{uc.target}' not found")
    tgt = uc.target
    repairs = []
    n_raw = len(df)
    df = df.copy()
    task, why = resolve_task(df[tgt], uc.task, prof[tgt]["kind"])
    if why:
        repairs.append(f"Task inferred as {task}: {why}.")

    # merge spelling variants in categoricals (incl. target) so they don't split categories
    for c, p in prof.items():
        if p["kind"] in ("categorical", "boolean") and c in df.columns:
            var = category_variants(df[c])
            if var:
                mapping = {}
                for grp in var.values():
                    canon = max(grp, key=lambda t: t[1])[0].strip()
                    for v, _ in grp:
                        mapping[v] = canon
                df[c] = df[c].map(lambda v: mapping.get(v, v) if isinstance(v, str) else v)
    n_missing_t = int(df[tgt].isna().sum())
    df = df[df[tgt].notna()]
    if n_missing_t:
        repairs.append(f"Dropped {n_missing_t:,} rows with a missing target.")
    nd = int(df.duplicated().sum())
    if nd:
        df = df.drop_duplicates()
        repairs.append(f"Dropped {nd:,} exact duplicate rows so copies cannot sit on both sides of a split.")
    df = df.reset_index(drop=True)

    classes, pos = [], None
    if task == "classification":
        codes, uniq = pd.factorize(df[tgt].astype(str), sort=True)
        classes = [str(u) for u in uniq]
        y = codes
        if len(classes) == 2:
            if uc.positive_class is not None and str(uc.positive_class) in classes:
                pos = classes.index(str(uc.positive_class))
            else:
                counts = np.bincount(y)
                yes = [i for i, c in enumerate(classes) if c.lower() in ("1", "yes", "true", "y", "positive", "malignant")]
                pos = yes[0] if yes else int(np.argmin(counts))
                uc.positive_class = classes[pos]
    else:
        y = pd.to_numeric(df[tgt], errors="coerce").to_numpy(float)
        ok = np.isfinite(y)
        if not ok.all():
            repairs.append(f"Dropped {(~ok).sum()} rows whose target is not a number.")
            df, y = df[ok].reset_index(drop=True), y[ok]

    goal = uc.goal if uc.goal in GOALS[task] else ("balanced" if task == "classification" else "mae")
    metric = GOALS[task][goal][1]
    if task == "classification" and metric == "roc_auc" and len(classes) > 2:
        metric = "roc_auc"   # handled as one-vs-rest
    if task == "regression" and metric == "mape" and (np.abs(y) < 1e-9).mean() > .01:
        repairs.append("Target has zeros, so MAPE is undefined; MAE is used as the primary metric instead.")
        metric, goal = "mae", "mae"

    t = None
    if uc.time_column and uc.time_column in df.columns:
        p = prof.get(uc.time_column, {})
        if p.get("kind") == "datetime":
            t = pd.to_datetime(df[uc.time_column], errors="coerce", dayfirst=p.get("dayfirst", False), format="mixed")
        else:
            t = pd.to_numeric(df[uc.time_column], errors="coerce")
        if t.notna().mean() < .8:
            repairs.append(f"Time column '{uc.time_column}' could not be parsed for most rows; temporal checks skipped.")
            t = None
    g = df[uc.entity_column].astype(str) if uc.entity_column and uc.entity_column in df.columns else None

    excluded = {}
    for c, p in prof.items():
        if c == tgt or c not in df.columns:
            continue
        if c in uc.ignore:
            excluded[c] = "ignored by you"
        elif c in uc.unavailable:
            excluded[c] = "not available at prediction time (your declaration)"
        elif c == uc.entity_column:
            excluded[c] = "entity identifier (used for grouping, not as a feature)"
        elif p["kind"] == "identifier":
            excluded[c] = "identifier: unique per row, carries no generalisable information"
        elif p["kind"] == "constant":
            excluded[c] = "constant"
        elif p["pii"]:
            excluded[c] = "personal data (PII)"
        elif p["kind"] == "categorical" and p["unique"] > 0.5 * len(df):
            excluded[c] = "near-unique text values"
    feats = [c for c in df.columns if c != tgt and c not in excluded]
    return Prepared(df=df, uc=uc, task=task, y=np.asarray(y), classes=classes, pos=pos, t=t, g=g,
                    features=feats, excluded=excluded, repairs=repairs, n_raw=n_raw, metric=metric, goal=goal)


def encode(df: pd.DataFrame, cols: list, prof: dict, max_levels=20):
    """Numeric matrix + mapping encoded-column -> raw column. Unsupervised encodings only (no target info)."""
    parts, origin = [], {}

    def add(name, values, raw):
        parts.append(pd.Series(np.asarray(values, dtype=float), name=name))
        origin[name] = raw

    for c in cols:
        p, s = prof[c], df[c]
        k = p["kind"]
        if k == "numeric":
            x = pd.to_numeric(s.astype(str).str.replace(r"[,\s$€£₹%]", "", regex=True), errors="coerce") if p.get("coerced_numeric") else s.astype(float)
            miss = x.isna()
            add(c, x.fillna(x.median() if miss.mean() < 1 else 0), c)
            if miss.mean() > .01:
                add(f"{c}__missing", miss.astype(float), c)
        elif k == "boolean":
            codes, _ = pd.factorize(s.astype(str).where(s.notna(), "__nan__"), sort=True)
            add(c, codes, c)
        elif k == "categorical":
            v = s.astype(str).where(s.notna(), "(missing)")
            if s.nunique() <= 25:
                top = v.value_counts().index[:max_levels]
                v = v.where(v.isin(top), "(other)")
                for lev in sorted(v.unique()):
                    add(f"{c}={lev}", (v == lev).astype(float), c)
            else:   # high cardinality: frequency encoding
                add(f"{c}__freq", v.map(v.value_counts(normalize=True)), c)
        elif k == "datetime":
            d = pd.to_datetime(s, errors="coerce", dayfirst=p.get("dayfirst", False), format="mixed")
            base = d.min()
            add(f"{c}__days", (d - base).dt.days.fillna(0), c)
            add(f"{c}__month", d.dt.month.fillna(0), c)
            add(f"{c}__weekday", d.dt.dayofweek.fillna(0), c)
        elif k == "time":
            tt = pd.to_datetime(s.astype(str), errors="coerce", format="mixed")
            add(f"{c}__hour", (tt.dt.hour + tt.dt.minute / 60).fillna(12), c)
        elif k == "text":
            v = s.fillna("").astype(str)
            add(f"{c}__len", v.str.len(), c)
            add(f"{c}__words", v.str.split().str.len(), c)
    if not parts:
        return pd.DataFrame(index=df.index), {}
    X = pd.concat(parts, axis=1)
    X.index = df.index
    keep = X.columns[X.nunique() > 1]
    return X[keep], {k: origin[k] for k in keep}


def sample_idx(P: Prepared, n=MAX_MODEL_ROWS):
    """Indices of a representative subsample for the expensive model-based checks."""
    N = len(P.df)
    if N <= n:
        idx = np.arange(N)
    else:
        rng = np.random.default_rng(SEED)
        if P.task == "classification":
            # stratified, but keep every row of very rare classes
            idx = []
            frac = n / N
            for k in np.unique(P.y):
                ki = np.where(P.y == k)[0]
                take = max(min(len(ki), 30), int(round(len(ki) * frac)))
                idx.append(rng.choice(ki, take, replace=False))
            idx = np.concatenate(idx)
        else:
            idx = rng.choice(N, n, replace=False)
    if P.t is not None:
        idx = idx[np.argsort(P.t.iloc[idx].to_numpy(), kind="stable")]
    else:
        idx = np.sort(idx)
    return idx
