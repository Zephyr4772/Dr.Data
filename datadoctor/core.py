"""Shared data structures: the use case, findings, and the analysis context passed between stages."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

SEED = 42
MAX_MODEL_ROWS = 5000      # model-based checks run on a (stratified) sample of at most this many rows

SEVERITIES = ["critical", "major", "minor", "info"]
SEV_PENALTY = {"critical": 45, "major": 20, "minor": 7, "info": 0}

# Fitness dimensions: key -> (label, weight in overall score, one-line question it answers)
DIMENSIONS = {
    "volume":   ("Volume",              0.10, "Is there enough data for this task?"),
    "quality":  ("Data hygiene",        0.15, "Are the values clean, consistent and plausible?"),
    "labels":   ("Target quality",      0.15, "Is the target well-defined, balanced and trustworthy?"),
    "leakage":  ("Leakage safety",      0.20, "Could a model cheat using information it won't have in production?"),
    "signal":   ("Predictive signal",   0.20, "Do the features actually carry information about the target?"),
    "validity": ("Validation realism",  0.10, "Will offline scores hold up on future / unseen data?"),
    "fairness": ("Coverage & subgroups", 0.10, "Does the model work equally well across the groups that matter?"),
}

GOALS = {
    "classification": {
        "balanced":  ("Balanced performance across classes", "balanced_accuracy"),
        "recall":    ("Catch as many positives as possible (missing one is costly)", "f2"),
        "precision": ("Avoid false alarms (acting on a wrong positive is costly)", "f05"),
        "ranking":   ("Rank / score cases by risk (probabilities matter)", "roc_auc"),
    },
    "regression": {
        "mae":  ("Small typical error (every unit of error costs the same)", "mae"),
        "rmse": ("Avoid large misses (big errors are disproportionately bad)", "rmse"),
        "mape": ("Small relative (%) error", "mape"),
    },
}

METRIC_INFO = {   # name -> (label, higher_is_better)
    "balanced_accuracy": ("Balanced accuracy", True),
    "f2": ("F2 score (recall-weighted)", True),
    "f05": ("F0.5 score (precision-weighted)", True),
    "roc_auc": ("ROC-AUC", True),
    "accuracy": ("Accuracy", True),
    "f1": ("F1 (macro)", True),
    "recall": ("Recall", True),
    "precision": ("Precision", True),
    "pr_auc": ("PR-AUC", True),
    "mae": ("MAE", False),
    "rmse": ("RMSE", False),
    "mape": ("MAPE", False),
    "r2": ("R²", True),
}


@dataclass
class UseCase:
    target: str
    task: str = "auto"                  # auto | classification | regression
    title: str = ""
    goal: str = ""                      # see GOALS; empty -> default for the task
    positive_class: str | None = None   # binary classification: the class you care about
    interpretability: str = "nice"      # not_needed | nice | required
    deployment: str = "server"          # server | batch | edge
    time_column: str | None = None      # rows ordered in time; "" or None = none
    entity_column: str | None = None    # rows belonging to the same customer / patient / device
    unavailable: list[str] = field(default_factory=list)   # not known at prediction time
    group_columns: list[str] = field(default_factory=list) # subgroups to check performance on
    ignore: list[str] = field(default_factory=list)        # columns to exclude entirely

    @classmethod
    def from_dict(cls, d: dict) -> "UseCase":
        keys = cls.__dataclass_fields__.keys()
        d = {k: v for k, v in d.items() if k in keys}
        for k in ("time_column", "entity_column", "positive_class"):
            if d.get(k) in ("", "none", "None"):
                d[k] = None
        return cls(**d)

    def to_dict(self):
        return asdict(self)


@dataclass
class Finding:
    id: str
    severity: str           # critical | major | minor | info
    dimension: str          # key of DIMENSIONS
    title: str              # one line, plain language
    evidence: str           # what was measured
    impact: str             # why it matters for THIS use case
    fix: str                # what to do
    columns: list = field(default_factory=list)
    chart: dict | None = None
    auto_handled: str = ""  # what the tool already did about it for the downstream analysis


def jsonable(o: Any):
    """Recursively convert numpy / pandas objects into plain JSON types."""
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if not np.isfinite(f) else round(f, 6)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (pd.Series, pd.Index)):
        return jsonable(o.tolist())
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    if hasattr(o, "__dataclass_fields__"):
        return jsonable(asdict(o))
    return o


def fmt(x, d=2):
    if x is None:
        return "n/a"
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    return f"{x:.{d}f}"


def pct(x, d=0):
    return f"{100 * x:.{d}f}%"
