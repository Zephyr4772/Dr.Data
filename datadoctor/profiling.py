"""Loading a CSV robustly and working out what each column actually is."""
from __future__ import annotations

import io
import re
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

TIME_RE = re.compile(r"^\s*\d{1,2}:\d{2}(:\d{2})?\s*$")
ID_NAME_RE = re.compile(r"(^id$|_id$|^id_|\bid\b|ID$|Id$|uuid|guid|^key$|_key$|number$|_no$)")
PII_NAME_RE = re.compile(r"(^name$|^full.?name|first.?name|last.?name|sur.?name|(customer|client|patient|user|employee|person|member|contact|student)_?.?name"
                         r"|email|e-mail|phone|mobile|address|ssn|passport|aadhaar|pan.?no|dob|birth.?date|date.?of.?birth)", re.I)
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[a-z]{2,}$", re.I)
PHONE_RE = re.compile(r"^\+?[\d\s\-()]{8,}$")


def load_csv(raw: bytes, name="dataset.csv"):
    """Decode bytes (UTF-8, falling back to cp1252) and sniff the delimiter. Returns (df, load_notes)."""
    notes = {}
    try:
        text = raw.decode("utf-8-sig")
        notes["encoding"] = "utf-8"
    except UnicodeDecodeError:
        text = raw.decode("cp1252", errors="replace")
        notes["encoding"] = "cp1252"
    sample = text[:20000]
    delims = {d: sample.count(d) for d in [",", ";", "\t", "|"]}
    sep = max(delims, key=delims.get)
    df = pd.read_csv(io.StringIO(text), sep=sep, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed: \d+$") | df.notna().any().values]
    notes["delimiter"] = sep
    notes["rows"], notes["cols"] = df.shape
    return df, notes


def _try_datetime(s: pd.Series):
    """Return (parsed_series, dayfirst) if the string column is mostly dates, else (None, None)."""
    v = s.dropna().astype(str)
    if len(v) == 0 or not v.str.contains(r"\d").mean() > .9:
        return None, None
    if not v.str.contains(r"[-/.]|\d{4}").mean() > .9 or v.str.len().median() < 6:
        return None, None
    smp = v.sample(min(500, len(v)), random_state=0)
    # day-first if any leading component > 12 (e.g. 25-12-2024)
    first = smp.str.extract(r"^(\d{1,2})[-/.]")[0].dropna().astype(int)
    dayfirst = bool(len(first) and (first > 12).any())
    parsed = pd.to_datetime(smp, errors="coerce", dayfirst=dayfirst, format="mixed")
    if parsed.notna().mean() < .9:
        return None, None
    full = pd.to_datetime(s, errors="coerce", dayfirst=dayfirst, format="mixed")
    return full, dayfirst


def _try_numeric(s: pd.Series):
    v = s.dropna().astype(str).str.replace(r"[,\s$€£₹%]", "", regex=True)
    if len(v) == 0:
        return None
    num = pd.to_numeric(v, errors="coerce")
    if num.notna().mean() >= .95:
        return pd.to_numeric(s.astype(str).str.replace(r"[,\s$€£₹%]", "", regex=True), errors="coerce")
    return None


def profile_columns(df: pd.DataFrame):
    """Infer a semantic type for every column plus summary stats used by the UI and by later checks.

    Types: numeric, categorical, boolean, datetime, time, text, identifier, constant
    """
    n = len(df)
    out = {}
    for c in df.columns:
        s = df[c]
        nun = int(s.nunique(dropna=True))
        info = {"name": c, "dtype": str(s.dtype), "missing": float(s.isna().mean()), "unique": nun,
                "unique_ratio": nun / max(1, s.notna().sum()), "notes": [], "pii": False}
        kind = None
        if nun <= 1:
            kind = "constant"
        elif pd.api.types.is_bool_dtype(s) or nun == 2:
            kind = "boolean" if nun == 2 else None
        if kind is None and pd.api.types.is_numeric_dtype(s):
            is_int = bool(np.all(np.mod(s.dropna(), 1) == 0))
            if ID_NAME_RE.search(c) and is_int and info["unique_ratio"] > .5:
                kind = "identifier"
            elif is_int and info["unique_ratio"] > .98 and n > 50 and s.dropna().is_monotonic_increasing:
                kind = "identifier"
                info["notes"].append("monotonic integer that never repeats: looks like a row index")
            else:
                kind = "numeric"
        if kind is None:
            v = s.dropna().astype(str)
            if v.head(200).str.match(TIME_RE).mean() > .9:
                kind = "time"
            else:
                parsed, dayfirst = _try_datetime(s)
                if parsed is not None:
                    kind = "datetime"
                    info["dayfirst"] = dayfirst
                    info["min"], info["max"] = str(parsed.min().date()), str(parsed.max().date())
                else:
                    num = _try_numeric(s)
                    if num is not None:
                        kind = "numeric"
                        info["notes"].append("numbers stored as text (symbols/commas stripped)")
                        info["coerced_numeric"] = True
                    elif ID_NAME_RE.search(c) and info["unique_ratio"] > .5:
                        kind = "identifier"
                    elif info["unique_ratio"] > .9 and n > 50 and v.str.len().median() < 25 and not v.str.contains(" ").mean() > .5:
                        kind = "identifier"
                    elif v.str.len().median() > 40 and v.str.contains(" ").mean() > .8 and info["unique_ratio"] > .3:
                        kind = "text"
                    else:
                        kind = "categorical"
        info["kind"] = kind
        # entity-like: an identifier / high-cardinality column whose values repeat (same customer, patient...)
        rows_per = s.notna().sum() / max(nun, 1)
        info["rows_per_value"] = float(rows_per)
        info["entity_like"] = bool(kind in ("identifier", "categorical") and nun >= 30 and 1.3 <= rows_per <= n / 10
                                   and (ID_NAME_RE.search(c) or kind == "identifier"))
        # PII
        if kind in ("categorical", "text", "identifier") and PII_NAME_RE.search(c):
            info["pii"] = True
        elif kind in ("categorical", "identifier", "text"):
            v = s.dropna().astype(str).head(300)
            if len(v) and (v.str.match(EMAIL_RE).mean() > .8 or v.str.match(PHONE_RE).mean() > .8):
                info["pii"] = True
        # stats + a tiny histogram for the UI sparkline
        if kind == "numeric":
            x = pd.to_numeric(s, errors="coerce") if info.get("coerced_numeric") else s
            x = x.astype(float).dropna()
            if len(x):
                info.update(mean=float(x.mean()), std=float(x.std()), min=float(x.min()), max=float(x.max()),
                            median=float(x.median()), skew=float(x.skew()) if len(x) > 2 else 0.0)
                lo, hi = np.percentile(x, [0.5, 99.5])
                if hi > lo:
                    h, e = np.histogram(x.clip(lo, hi), bins=24)
                    info["hist"] = {"counts": h.tolist(), "edges": [float(e[0]), float(e[-1])]}
        elif kind in ("categorical", "boolean", "constant"):
            vc = s.astype(str).where(s.notna(), None).value_counts().head(8)
            info["top"] = [[str(k), int(v)] for k, v in vc.items()]
        out[c] = info
    return out


def suggest_usecase(df: pd.DataFrame, prof: dict):
    """Pre-fill hints for the use-case form."""
    n = len(df)
    targets = []
    for c, p in prof.items():
        if p["kind"] in ("identifier", "text", "constant", "time", "datetime") or p["pii"]:
            continue
        if p["kind"] in ("boolean", "categorical") and 2 <= p["unique"] <= 20:
            targets.append({"column": c, "task": "classification", "why": f"{p['unique']} classes"})
        elif p["kind"] == "numeric" and p["unique"] > 20:
            targets.append({"column": c, "task": "regression", "why": "continuous numeric"})
        elif p["kind"] == "numeric" and 2 <= p["unique"] <= 10:
            targets.append({"column": c, "task": "classification", "why": f"{p['unique']} discrete values (ordinal?)"})
    # prefer typical target names
    strong = re.compile(r"(target|label|^class$|flag|churn|fraud|default|outcome|survived|returned|diagnosis|^y$)", re.I)
    weak = re.compile(r"(price|profit|revenue|sales|amount|score|rating|cost)", re.I)
    targets.sort(key=lambda t: (not strong.search(t["column"]), not weak.search(t["column"]), t["task"] != "classification"))
    times = [c for c, p in prof.items() if p["kind"] == "datetime"]
    entities = sorted([c for c, p in prof.items() if p["entity_like"]], key=lambda c: -prof[c]["unique"])
    groups = [c for c, p in prof.items() if p["kind"] in ("categorical", "boolean") and 2 <= p["unique"] <= 12
              and not p["pii"] and re.search(r"(gender|sex|age|region|country|segment|race|ethnic|channel|group|city|state)", c, re.I)]
    ignore = [c for c, p in prof.items() if p["kind"] == "identifier" and not p["entity_like"]]
    return {"targets": targets[:12], "time_columns": times, "entity_columns": entities,
            "group_columns": groups[:4], "ignore": ignore, "n_rows": n}
