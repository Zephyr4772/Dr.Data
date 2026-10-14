"""Data-level checks: hygiene, target quality and leakage. Each check explains WHY the issue matters for the use case."""
from __future__ import annotations

import re
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .core import SEED, Finding, fmt, pct
from .evaluation import baseline_score, score, skill, splitter
from .prep import category_variants, encode, sample_idx

NONNEG_HINT = re.compile(r"(qty|quant|price|amount|age|count|days|units|cost|sales|weight|height|stock|inventor|revenue|income|salary|duration|distance|area|size)", re.I)
PCT_HINT = re.compile(r"(percent|pct|discount|_rate$|ratio_pct)", re.I)
POST_HINT = re.compile(r"(final|outcome|result|status|resolved|closed|approved|refund|returned|cancel|paid_off|settle|after|post_|_post|actual)", re.I)


class Ctx:
    """Everything the stages share."""
    def __init__(self, raw, prof, P, notes, progress=None):
        self.raw, self.prof, self.P, self.notes = raw, prof, P, notes
        self.findings: list[Finding] = []
        self.R: dict = {}
        self.progress = progress or (lambda *a, **k: None)

    def add(self, **kw):
        kw.setdefault("id", f"f{len(self.findings) + 1}")
        self.findings.append(Finding(**kw))

    @property
    def tgt(self):
        return self.P.uc.target

    def target_words(self):
        P = self.P
        if P.task == "classification" and len(P.classes) == 2:
            return f"'{P.uc.target}' = {P.classes[P.pos]}"
        return f"'{P.uc.target}'"


# ============================================================================= hygiene
def hygiene(ctx: Ctx):
    raw, prof, P = ctx.raw, ctx.prof, ctx.P
    n = len(raw)
    tgt = ctx.tgt

    # --- encoding damage
    bad = {}
    for c, p in prof.items():
        if p["kind"] in ("categorical", "text"):
            v = raw[c].dropna().astype(str)
            m = v.str.contains("�|Ã.|â€", regex=True)
            if m.any():
                bad[c] = (int(m.sum()), v[m].iloc[0])
    if bad:
        ex = "; ".join(f"'{c}': {k} values, e.g. \"{e}\"" for c, (k, e) in list(bad.items())[:3])
        ctx.add(severity="minor", dimension="quality", title="Text encoding is damaged in some values",
                evidence=f"Replacement / mojibake characters found: {ex}.",
                impact="Corrupted strings become separate categories, so 'Île-de-France' and its broken copy are treated as different places.",
                fix="Re-export the file as UTF-8 from the source system, or repair with a mapping before training.",
                columns=list(bad))

    # --- missing values
    miss = raw.drop(columns=[tgt]).isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0]
    ctx.R["missing"] = miss.round(4).to_dict()
    for c, m in miss.items():
        if m < .02:
            continue
        sev = "major" if m > .4 else "minor"
        kind = prof[c]["kind"]
        informative = _missing_informative(ctx, c)
        structural = kind == "categorical" and m > .2
        title = f"'{c}' is {pct(m)} empty"
        ev = f"{int(raw[c].isna().sum()):,} of {n:,} rows have no value."
        if informative:
            ev += " " + informative["text"]
        impact = ("Most models cannot take missing values directly; how you fill them changes what the model learns. "
                  if not informative else
                  "The absence itself predicts the target, so it is information, not noise. Filling it with a median/mode erases that signal. ")
        if structural:
            fix = f"This looks like a structural blank (e.g. 'no {c.replace('_', ' ').lower()}'). Encode it as an explicit category such as 'None' rather than imputing."
        elif m > .4:
            fix = "Keep it only if it is predictive (see below); otherwise drop. If kept, impute inside the CV pipeline and add a missing-indicator column."
        else:
            fix = "Impute inside the training pipeline (median / most-frequent, fit on training folds only) and add a missing-indicator flag."
        ctx.add(severity=sev, dimension="quality", title=title, evidence=ev, impact=impact, fix=fix, columns=[c],
                chart=informative.get("chart") if informative else None,
                auto_handled="Median/'(missing)' fill plus a missing-indicator was used for the analysis.")

    # --- duplicates
    dup = raw.duplicated().sum()
    ctx.R["duplicates"] = int(dup)
    if dup:
        r = dup / n
        ctx.add(severity="major" if r > .01 else "minor", dimension="quality",
                title=f"{dup:,} exact duplicate rows ({pct(r, 1)})",
                evidence=f"{dup:,} rows are byte-for-byte copies of another row.",
                impact="With a random split, a row and its copy can land in train and test. The model is then graded on rows it has memorised, inflating every score.",
                fix="Deduplicate before splitting. If duplicates are genuine repeat events, keep them but split by entity/time.",
                auto_handled="Exact duplicates were dropped for all model-based checks.")
    for c, p in prof.items():
        if p["kind"] == "identifier" and .9 < p["unique_ratio"] < 1 and raw[c].notna().all():
            rep = int(raw[c].duplicated().sum())
            same = int(raw[raw[c].duplicated(keep=False)].duplicated().sum())
            if same == rep and dup:
                last = ctx.findings[-1]
                last.evidence += f" They also explain all {rep} repeated values of '{c}', which should be unique: the same records were loaded twice."
                last.columns = [c]
                continue
            ctx.add(severity="minor", dimension="quality", title=f"Identifier '{c}' is not unique ({rep:,} repeats)",
                    evidence=f"{rep:,} values of '{c}' occur more than once; only {same:,} of those rows are full duplicates.",
                    impact="An ID that should be unique but repeats with different content means either the key is wrong or records were updated without versioning.",
                    fix=f"Confirm the grain of the table (one row per what?) and decide which version of each record is authoritative.",
                    columns=[c])
    feats = [c for c in P.features if c in P.df.columns]
    if feats:
        fd = P.df.duplicated(subset=feats, keep=False)
        if fd.any():
            conflict = P.df[fd].groupby(feats, dropna=False)[tgt].nunique()
            nc = int((conflict > 1).sum())
            if nc:
                ctx.add(severity="major" if nc > .01 * len(P.df) else "minor", dimension="labels",
                        title=f"{nc:,} groups of identical inputs carry different targets",
                        evidence=f"Rows with exactly the same feature values disagree on '{tgt}' in {nc:,} cases.",
                        impact="No model can be right on both rows. This sets a hard ceiling on accuracy and hints at label errors or missing features.",
                        fix="Inspect these rows: either labels are wrong, or a feature that distinguishes them is missing from the data.")

    # --- constant / near-constant
    for c, p in prof.items():
        if c == tgt:
            continue
        if p["kind"] == "constant":
            ctx.add(severity="minor", dimension="quality", title=f"'{c}' has a single value", evidence="Every row is identical.",
                    impact="Zero information; it only adds noise to the feature list.", fix="Drop it.", columns=[c],
                    auto_handled="Excluded.")
        elif p.get("top") and p["top"][0][1] / n > .99:
            ctx.add(severity="info", dimension="quality", title=f"'{c}' is almost constant ({pct(p['top'][0][1] / n, 1)} = '{p['top'][0][0]}')",
                    evidence="Fewer than 1% of rows differ.", impact="Nearly no information; rare levels may cause overfitting.",
                    fix="Drop it or merge the rare values.", columns=[c])

    # --- inconsistent category spellings
    var_cols = {}
    for c, p in prof.items():
        if p["kind"] in ("categorical", "boolean"):
            v = category_variants(raw[c])
            if v:
                var_cols[c] = v
    ctx.R["variants"] = {c: [[x for x, _ in g] for g in v.values()] for c, v in var_cols.items()}
    if var_cols:
        examples, affected = [], 0
        for c, v in var_cols.items():
            g = max(v.values(), key=lambda gr: sum(cnt for _, cnt in gr))
            examples.append(f"{c}: " + " / ".join(f"'{x}' ({k})" for x, k in g[:3]))
            affected += sum(sum(cnt for _, cnt in gr) - max(cnt for _, cnt in gr) for gr in v.values())
        hit_target = tgt in var_cols
        ctx.add(severity="major" if hit_target or len(var_cols) > 3 else "minor", dimension="quality",
                title=f"Same category spelled differently in {len(var_cols)} column(s)",
                evidence=f"{affected:,} values use a non-standard spelling. " + "; ".join(examples[:4]) + ".",
                impact=("One real category is split into several, each with fewer rows. One-hot encoding creates useless extra columns "
                        "and group statistics (return rate per payment method, etc.) are wrong." +
                        (" The TARGET itself is affected, so some classes are fragmented." if hit_target else "")),
                fix="Normalise case and whitespace and map known aliases (e.g. 'DebitCard' to 'Debit Card') before any encoding.",
                columns=list(var_cols), auto_handled="Variants were merged into their most frequent spelling for the analysis.")

    # --- impossible / implausible values
    for c, p in prof.items():
        if p["kind"] != "numeric" or c == tgt:
            continue
        x = pd.to_numeric(raw[c], errors="coerce").dropna()
        if not len(x):
            continue
        probs = []
        neg = (x < 0).sum()
        if neg and (x >= 0).mean() > .97 and NONNEG_HINT.search(c):
            probs.append(f"{neg} negative value(s) (min {fmt(x.min())})")
        if PCT_HINT.search(c) and ((x > 100).sum() or (x < 0).sum()):
            probs.append(f"{int((x > 100).sum() + (x < 0).sum())} value(s) outside 0-100 (max {fmt(x.max())})")
        if re.search(r"(^|_)age($|_)", c, re.I):
            k = int(((x < 0) | (x > 110)).sum())
            low = int(((x >= 0) & (x < 12)).sum())
            if k:
                probs.append(f"{k} impossible age(s) (max {fmt(x.max(), 0)})")
            if low and x.median() > 25:
                probs.append(f"{low} implausibly young value(s) (min {fmt(x.min(), 0)})")
        if re.search(r"year", c, re.I) and ((x < 1900) | (x > 2100)).any():
            probs.append("years outside 1900-2100")
        if probs:
            ctx.add(severity="major", dimension="quality", title=f"'{c}' contains impossible values",
                    evidence="; ".join(probs) + ".",
                    impact="These are almost certainly entry or pipeline errors. A model will treat them as real behaviour, and a few extreme ones can bend linear models and scaling.",
                    fix="Trace them back to the source. Fix if recoverable; otherwise set to missing (not to 0) and let the imputer handle them.",
                    columns=[c])

    # --- outliers (robust z-score)
    rows = []
    for c, p in prof.items():
        if p["kind"] != "numeric" or c == tgt or p["unique"] < 10:
            continue
        x = pd.to_numeric(raw[c], errors="coerce").dropna().astype(float)
        med = x.median()
        mad = (x - med).abs().median()
        if mad == 0:
            continue
        z = .6745 * (x - med) / mad
        share = float((z.abs() > 5).mean())
        if share > .005:
            rows.append((c, share, float(x.skew()), float(z.abs().max())))
    ctx.R["outliers"] = {c: s for c, s, _, _ in rows}
    if rows:
        rows.sort(key=lambda r: -r[1])
        tail = [r[0] for r in rows if abs(r[2]) > 2]
        glitch = [r[0] for r in rows if abs(r[2]) <= 2]
        ev = "; ".join(f"'{c}': {pct(s, 1)} of rows beyond 5 robust SDs (skew {sk:.1f})" for c, s, sk, _ in rows[:5])
        imp = []
        if tail:
            imp.append(f"{', '.join(tail[:4])} {'is' if len(tail) == 1 else 'are'} heavy-tailed: the extremes are part of the distribution "
                       "(a few large orders), not errors.")
        if glitch:
            imp.append(f"{', '.join(glitch[:4])} {'has' if len(glitch) == 1 else 'have'} isolated extreme points in an otherwise compact distribution: possible glitches.")
        imp.append("Linear models, k-NN and neural nets are pulled toward extremes; tree models are largely unaffected.")
        ctx.add(severity="minor", dimension="quality", title=f"Extreme values in {len(rows)} numeric column(s)", evidence=ev + ".",
                impact=" ".join(imp),
                fix="For heavy tails: log-transform or use tree models. For glitches: verify and cap (winsorise at the 1st/99th percentile). Never delete rows blindly; extremes are often the cases the use case cares about.",
                columns=[r[0] for r in rows],
                chart={"type": "bars", "labels": [r[0] for r in rows[:8]], "values": [r[1] * 100 for r in rows[:8]], "unit": "% rows"})

    # --- redundancy: 1:1 categorical encodings, |r|>.95 numeric pairs, numeric = date part
    cats = [c for c, p in prof.items() if p["kind"] in ("categorical", "identifier") and 2 < p["unique"] <= 500 and c != tgt]
    clean = {c: raw[c].fillna("").astype(str).str.lower().str.replace(r"[^0-9a-z]", "", regex=True) for c in cats}
    pairs = []
    for a, b in combinations(cats, 2):
        if abs(prof[a]["unique"] - prof[b]["unique"]) > 3:
            continue
        ab = pd.DataFrame({"a": clean[a], "b": clean[b]})
        if (ab.groupby("a")["b"].nunique() <= 1).mean() > .99 and (ab.groupby("b")["a"].nunique() <= 1).mean() > .99:
            pairs.append((a, b))
    num = [c for c, p in prof.items() if p["kind"] == "numeric" and c != tgt]
    if len(num) > 1:
        cm = raw[num].apply(pd.to_numeric, errors="coerce").corr().abs()
        for i, a in enumerate(num):
            for b in num[i + 1:]:
                if cm.loc[a, b] > .95:
                    pairs.append((a, b, float(cm.loc[a, b])))
    for c, p in prof.items():
        if p["kind"] == "datetime":
            d = pd.to_datetime(raw[c], errors="coerce", dayfirst=p.get("dayfirst", False), format="mixed")
            for nc in num:
                x = pd.to_numeric(raw[nc], errors="coerce")
                for part in ("year", "month"):
                    if (x == getattr(d.dt, part)).mean() > .99:
                        pairs.append((nc, f"{part} of {c}"))
    ctx.R["redundant"] = [list(p) for p in pairs]
    if pairs:
        txt = "; ".join(f"'{p[0]}' = '{p[1]}'" + (f" (r={p[2]:.2f})" if len(p) == 3 else "") for p in pairs[:6])
        ctx.add(severity="minor", dimension="quality", title=f"{len(pairs)} redundant column pair(s)",
                evidence=f"These columns encode the same information: {txt}.",
                impact="Duplicate information doubles the feature count without adding signal, splits feature importance between copies "
                       "(each looks half as important) and makes linear coefficients unstable.",
                fix="Keep one column from each pair (prefer the human-readable one for interpretability).",
                columns=sorted({x for p in pairs for x in p[:2] if x in raw.columns}))

    # --- derived numeric columns (one column is an exact formula of others) and rows that break the formula
    derived = []
    numc = [c for c, p in prof.items() if p["kind"] == "numeric" and c != tgt and p["unique"] > 20][:15]
    if len(numc) >= 3:
        Xn = raw[numc].apply(pd.to_numeric, errors="coerce")
        seen = set()
        for c in numc:
            for combo, formula, share, bad_rows in find_identities(Xn[c], Xn.drop(columns=[c]), c, max_cands=8, min_share=.95)[:1]:
                key = frozenset((c,) + tuple(combo))
                if len(combo) == 2 and key not in seen:
                    seen.add(key)
                    derived.append((c, combo, formula, share, bad_rows))
    ctx.R["derived"] = [{"column": d[0], "formula": d[2], "share": d[3], "violations": len(d[4])} for d in derived]
    for c, combo, formula, share, bad_rows in derived[:3]:
        nbad = len(bad_rows)
        ctx.add(severity="major" if nbad > 0 else "minor", dimension="quality",
                title=f"'{c}' is calculated from other columns" + (f", and {nbad:,} rows break the formula" if nbad else ""),
                evidence=f"{formula} holds for {pct(share, 1)} of rows." + (f" Violating rows include {', '.join(map(str, bad_rows[:6]))}." if nbad else ""),
                impact=("A column that is a fixed formula of others adds no information, and the rows where the formula breaks are "
                        "internal contradictions: at least one of the numbers on those rows is wrong, and you can't tell which."),
                fix=(f"Recompute '{c}' from its inputs (or drop it), and audit the {nbad:,} inconsistent rows at the source." if nbad
                     else f"Drop '{c}' or its inputs; keep the most interpretable version."),
                columns=[c, *combo])

    # --- identifiers
    ids = [c for c, p in prof.items() if p["kind"] == "identifier" and c != P.uc.entity_column]
    if ids:
        ctx.add(severity="info", dimension="leakage", title=f"{len(ids)} identifier column(s) excluded: {', '.join(ids[:4])}",
                evidence="Values are (nearly) unique per row or follow a row-index pattern.",
                impact="IDs carry no generalisable information. Worse, if IDs were assigned in time order or by outcome, a model can use them as a shortcut that disappears in production.",
                fix="Keep IDs out of the feature set; use them only for joining and for grouped splits.", columns=ids,
                auto_handled="Excluded from features.")

    # --- personal data
    pii = [c for c, p in prof.items() if p["pii"]]
    if pii:
        ctx.add(severity="minor", dimension="quality", title=f"Personal data present: {', '.join(pii)}",
                evidence="Column names / value patterns look like names, e-mails, phone numbers or addresses.",
                impact="Personal identifiers almost never generalise (a model that learns 'Amelia King returns items' is memorising), and storing them in a training set creates privacy and compliance risk.",
                fix="Drop them from the training data, or replace with aggregated behavioural features (e.g. customer's past order count).",
                columns=pii, auto_handled="Excluded from features.")


def _term(w, c):
    sign = "+" if w >= 0 else "-"
    mag = "" if abs(abs(w) - 1) < 1e-3 else f"{abs(w):g}·"
    return f"{sign} {mag}{c}"


def find_identities(y: pd.Series, Xn: pd.DataFrame, yname: str, max_cands=10, min_share=.9):
    """Search for exact relations y = a*x1 (+ b*x2) + c or y = k * x1^a * x2^b that hold for most rows.

    Robust to a few corrupted rows: a relation counts if >= min_share of rows satisfy it to rounding precision.
    Returns [(columns, formula_text, share_of_rows_satisfying, violating_row_labels)].
    """
    Xn = Xn.loc[:, Xn.notna().mean() > .95]
    if Xn.shape[1] == 0:
        return []
    ok = Xn.notna().all(1) & y.notna()
    Xn, y = Xn[ok], y[ok]
    if len(y) < 30:
        return []
    smp = np.random.default_rng(SEED).choice(len(y), min(2000, len(y)), replace=False)
    cor = Xn.iloc[smp].corrwith(y.iloc[smp]).abs().sort_values(ascending=False)
    top = [c for c in cor.index[:max_cands] if np.isfinite(cor[c])]
    yv = y.to_numpy(float)
    tol = max(1e-6 * np.abs(yv).max(), .011)       # cents-level rounding
    found = []
    for k_ in (1, 2):
        for combo in combinations(top, k_):
            A = Xn[list(combo)].to_numpy(float)
            lr = LinearRegression().fit(A[smp], yv[smp])
            res = np.abs(lr.predict(A[smp]) - yv[smp])
            inl = res < tol + 1e-3 * np.std(yv)
            if inl.mean() < .5:
                continue
            lr = LinearRegression().fit(A[smp][inl], yv[smp][inl])        # refit on inliers only
            good = np.abs(lr.predict(A) - yv) < tol + 1e-4 * np.std(yv)
            if good.mean() >= min_share:
                terms = " ".join(_term(w, c) for w, c in zip(np.round(lr.coef_, 3), combo)).lstrip("+ ")
                b = lr.intercept_
                if abs(b) > tol:
                    terms += f" {'+' if b >= 0 else '-'} {abs(b):.3g}"
                found.append((combo, f"{yname} = {terms}", float(good.mean()), list(y.index[~good])))
        if found:
            return found
    pos = (Xn > 0).all(1).to_numpy() & (yv > 0)
    if pos.mean() > .9:
        for combo in combinations(top, 2):
            A = np.log(Xn[list(combo)].to_numpy(float)[pos]); ly = np.log(yv[pos])
            lr = LinearRegression().fit(A, ly)
            good = np.abs(lr.predict(A) - ly) < 1e-3
            if good.mean() >= min_share:
                found.append((combo, f"{yname} = {np.exp(lr.intercept_):.3g} × {combo[0]}^{lr.coef_[0]:.2f} × {combo[1]}^{lr.coef_[1]:.2f}",
                              float(good.mean() * pos.mean()), list(y.index[pos][~good])))
                return found
    return found


def _missing_informative(ctx: Ctx, c):
    P = ctx.P
    if c not in P.df.columns:
        return None
    m = P.df[c].isna().to_numpy()
    if m.mean() < .02 or m.mean() > .98 or m.sum() < 20:
        return None
    y = P.y
    if P.task == "classification":
        tab = np.array([[np.sum((m == a) & (y == k)) for k in range(len(P.classes))] for a in (0, 1)])
        tab = tab[:, tab.sum(0) > 0]
        if tab.shape[1] < 2:
            return None
        chi2, p, _, _ = stats.chi2_contingency(tab)
        if P.pos is not None:
            r1, r0 = (y[m] == P.pos).mean(), (y[~m] == P.pos).mean()
            eff = abs(r1 - r0) / max(r0, 1e-9)
            txt = f"Rows where it is missing have {ctx.target_words()} in {pct(r1, 1)} of cases vs {pct(r0, 1)} when present (chi-square p={p:.1g})."
            chart = {"type": "bars", "labels": ["value missing", "value present"], "values": [r1 * 100, r0 * 100], "unit": f"% {P.classes[P.pos]}"}
        else:
            eff, txt, chart = .3, f"The class mix differs between rows with and without a value (chi-square p={p:.1g}).", None
    else:
        a, b = y[m], y[~m]
        _, p = stats.mannwhitneyu(a, b)
        eff = abs(np.median(a) - np.median(b)) / (np.std(y) + 1e-9)
        txt = f"Median {ctx.tgt} is {fmt(np.median(a))} when missing vs {fmt(np.median(b))} when present (Mann-Whitney p={p:.1g})."
        chart = {"type": "bars", "labels": ["value missing", "value present"], "values": [float(np.median(a)), float(np.median(b))], "unit": f"median {ctx.tgt}"}
    if p < .001 and eff > .1:
        return {"text": txt, "p": p, "chart": chart}
    return None


# ============================================================================= target
def target_checks(ctx: Ctx):
    P = ctx.P
    y = P.y
    tgt = ctx.tgt
    miss = ctx.raw[tgt].isna().mean()
    if miss > 0:
        ctx.add(severity="minor" if miss < .1 else "major", dimension="labels", title=f"{pct(miss, 1)} of rows have no target value",
                evidence=f"{int(ctx.raw[tgt].isna().sum()):,} rows cannot be used for supervised training.",
                impact="Fewer usable rows; if the missing labels are not random (e.g. only resolved cases get labelled) the training set is biased.",
                fix="Check why labels are missing. If they are the most recent rows, they are your future scoring set, not training data.",
                auto_handled="Dropped.")
    if P.task == "classification":
        counts = np.bincount(y, minlength=len(P.classes))
        ctx.R["target"] = {"type": "classes", "labels": P.classes, "counts": counts.tolist(), "positive": P.uc.positive_class}
        k = len(P.classes)
        ratio = counts.max() / max(counts.min(), 1)
        maj = counts.max() / counts.sum()
        if k > 20:
            ctx.add(severity="major", dimension="labels", title=f"{k} target classes",
                    evidence=f"The target has {k} distinct values; the smallest has {counts.min()} rows.",
                    impact="Many classes with few examples each are hard to learn and to evaluate; most errors will be between similar classes.",
                    fix="Group rare classes into 'other', use a hierarchical target, or reconsider whether this is really a regression problem.")
        rare = [P.classes[i] for i in range(k) if counts[i] < 30]
        if rare:
            ctx.add(severity="major", dimension="labels", title=f"{len(rare)} class(es) have fewer than 30 rows",
                    evidence=f"Rare classes: {', '.join(rare[:6])} (min {counts.min()} rows).",
                    impact="A model sees too few examples to learn these classes, and their test-set metrics are statistically meaningless (one error swings recall by several points).",
                    fix="Collect more examples, merge with a related class, or treat them as a separate anomaly-detection problem.")
        if ratio > 3:
            sev = "major" if ratio > 10 else "minor"
            pc = P.classes[P.pos] if P.pos is not None else P.classes[int(counts.argmin())]
            goal_txt = {"recall": "Because your goal is to catch positives, an imbalanced model that rarely predicts the minority class is the worst failure mode.",
                        "precision": "With few positives, even a small false-positive rate produces more false alarms than true hits.",
                        "ranking": "ROC-AUC can look good under heavy imbalance while precision at the top is poor; check PR-AUC too.",
                        "balanced": "Plain accuracy is meaningless here; balanced accuracy weights every class equally."}.get(P.goal, "")
            ctx.add(severity=sev, dimension="labels", title=f"Class imbalance {ratio:.1f} : 1",
                    evidence=f"The majority class is {pct(maj, 1)} of rows. A model that always predicts it scores {pct(maj, 1)} accuracy while being useless for '{pc}'.",
                    impact="Models trained naively drift toward the majority class. " + goal_txt,
                    fix="Use stratified splits, class weights (or resampling inside the CV loop only), report recall/precision/PR-AUC for the minority class, and tune the decision threshold instead of using 0.5.",
                    auto_handled="Class-weighted models are included in the bake-off.",
                    chart={"type": "bars", "labels": P.classes[:10], "values": counts[:10].tolist(), "unit": "rows"})
    else:
        sk = float(stats.skew(y))
        zeros = float((y == 0).mean())
        h, e = np.histogram(np.clip(y, *np.percentile(y, [.5, 99.5])), bins=30)
        ctx.R["target"] = {"type": "numeric", "hist": h.tolist(), "edges": [float(e[0]), float(e[-1])], "skew": sk,
                           "mean": float(y.mean()), "median": float(np.median(y)), "std": float(y.std()),
                           "min": float(y.min()), "max": float(y.max()), "zeros": zeros}
        if abs(sk) > 2:
            pos = (y > 0).all() or ((y >= 0).all())
            ctx.add(severity="minor", dimension="labels", title=f"Target is strongly skewed (skew {sk:.1f})",
                    evidence=f"Mean {fmt(y.mean())} vs median {fmt(np.median(y))}; the top 1% of rows exceed {fmt(np.percentile(y, 99))}.",
                    impact=("A few very large values dominate squared-error training: the model spends its capacity on the tail and is "
                            "systematically off for typical rows. RMSE will be driven by a handful of rows."),
                    fix=("Train on log(1 + target) and transform predictions back" if pos else
                         "Consider a robust loss (Huber / absolute error) or a signed log transform") +
                        ", and report MAE alongside RMSE.",
                    chart={"type": "hist", "counts": h.tolist(), "edges": [float(e[0]), float(e[-1])], "label": tgt})
        if zeros > .3:
            ctx.add(severity="minor", dimension="labels", title=f"{pct(zeros)} of targets are exactly zero",
                    evidence="The target is zero-inflated.",
                    impact="A single regressor has to model both 'is it zero?' and 'how much?', usually doing neither well.",
                    fix="Consider a two-stage (hurdle) model: a classifier for zero vs non-zero, then a regressor on the non-zero rows.")
        if len(np.unique(y)) < 15:
            ctx.add(severity="info", dimension="labels", title=f"Target takes only {len(np.unique(y))} distinct values",
                    evidence="A numeric target with few levels.", impact="It may be better framed as (ordinal) classification.",
                    fix="Try both framings and compare with the metric you actually care about.")
        if P.metric == "mape" and (np.abs(y) < np.percentile(np.abs(y), 5) + 1e-9).mean() > .05:
            ctx.add(severity="minor", dimension="labels", title="MAPE is unstable for this target",
                    evidence=f"5% of targets are below {fmt(np.percentile(np.abs(y), 5))}; percentage errors on tiny values explode.",
                    impact="A few small-valued rows will dominate the MAPE number.",
                    fix="Use MAE or weighted MAPE (sum |error| / sum |target|) instead.")


# ============================================================================= leakage
def leakage(ctx: Ctx):
    P, prof = ctx.P, ctx.prof
    tgt = ctx.tgt
    if P.uc.unavailable:
        ctx.add(severity="info", dimension="leakage", title=f"{len(P.uc.unavailable)} column(s) excluded as 'not known at prediction time'",
                evidence=", ".join(P.uc.unavailable) + ".",
                impact="Good practice: these would leak future information into training.",
                fix="Nothing to do; keep this list with the model documentation.", columns=list(P.uc.unavailable))

    idx = sample_idx(P, 3000)
    y = P.y[idx]
    df = P.df.iloc[idx]
    splits = splitter("random", P.task, 3, y=y)
    base = baseline_score(P.metric if P.task == "regression" else "balanced_accuracy", y, P.task, P.pos, len(P.classes))
    single = {}
    for c in P.features:
        X, _ = encode(df, [c], prof)
        if X.shape[1] == 0:
            continue
        if P.task == "classification":
            pred = cross_val_predict(DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=SEED), X.values, y,
                                     cv=splits)
            s = score("balanced_accuracy", y, pred)
            single[c] = skill("balanced_accuracy", s, base)
        else:
            pred = cross_val_predict(DecisionTreeRegressor(max_depth=6, min_samples_leaf=5, random_state=SEED), X.values, y, cv=splits)
            single[c] = max(0.0, score("r2", y, pred))
    single = dict(sorted(single.items(), key=lambda kv: -kv[1]))
    ctx.R["single_feature"] = {"values": single, "measure": "skill over baseline (balanced accuracy)" if P.task == "classification" else "R² from this column alone"}

    leaks = {}
    unit = "skill" if P.task == "classification" else "R²"
    vals = list(single.values())
    strong_many = []
    for c, v in single.items():
        hi, mid = (.9, .7) if P.task == "classification" else (.95, .85)
        if v >= hi:
            leaks[c] = f"alone explains the target ({unit} {v:.2f})"
        elif v >= .6 and POST_HINT.search(c):
            leaks[c] = (f"alone predicts the target ({unit} {v:.2f}) AND its name suggests it is recorded after the outcome "
                        f"('{POST_HINT.search(c).group(0)}')")
        elif v >= mid:
            runner_up = max([x for k, x in single.items() if k != c] or [0])
            if runner_up > v - .1:          # other columns are about as strong: an easy problem, not a lone leak
                strong_many.append(c)
                continue
            ctx.add(severity="minor", dimension="leakage", title=f"'{c}' alone is a very strong predictor",
                    evidence=f"A small tree using only '{c}' reaches {'skill' if P.task == 'classification' else 'R²'} {v:.2f}.",
                    impact="This might be a legitimate strong driver, or a proxy for the answer. If it is recorded after the outcome, every score you see is inflated.",
                    fix=f"Confirm '{c}' is known at the moment you would make the prediction; if not, mark it as unavailable and re-run.",
                    columns=[c])

    if strong_many:
        ctx.add(severity="info", dimension="signal", title=f"{len(strong_many)} columns are each strong predictors on their own",
                evidence=f"{', '.join(strong_many[:6])} each reach {unit} ≥ {min(single[c] for c in strong_many):.2f} alone.",
                impact="When several independent measurements all predict the target, the problem is genuinely easy; a leak usually shows up as ONE column standing far above the rest.",
                fix="No action needed beyond confirming these are measured before the outcome.", columns=strong_many)

    # arithmetic identities (regression): target = a*x1 + b*x2 + c  or  x1^a * x2^b
    ident = []
    if P.task == "regression":
        num = [c for c in P.features if prof[c]["kind"] == "numeric"]
        ident = find_identities(pd.Series(P.y, index=P.df.index), P.df[num].apply(pd.to_numeric, errors="coerce"), tgt)
    for combo, formula, share, bad_rows in ident[:2]:
        broken = (f" It is violated by {len(bad_rows):,} rows ({pct(1 - share, 1)}), which are most likely data-entry errors "
                  f"(e.g. rows {', '.join(map(str, bad_rows[:5]))})." if len(bad_rows) else "")
        ctx.add(severity="critical", dimension="leakage", title="The target is a formula of other columns",
                evidence=f"{formula} holds exactly for {pct(share, 1)} of rows.{broken}",
                impact=(f"If {', '.join(combo)} are known when you predict, you don't need machine learning at all: just compute it. "
                        f"If they are not known, they leak the answer and any model using them will look near-perfect offline and fail in production."),
                fix=f"Decide which is true. Usually '{tgt}' should be predicted from upstream drivers (price, quantity, discount, product...) with {', '.join(combo)} removed.",
                columns=list(combo), auto_handled="Removed for the remaining analysis so you see what the legitimate features can do.")
        for c in combo:
            leaks.setdefault(c, "part of an exact formula for the target")

    for c, why in leaks.items():
        if not any(c in f.columns and f.severity == "critical" for f in ctx.findings):
            ctx.add(severity="critical", dimension="leakage", title=f"'{c}' leaks the target",
                    evidence=f"'{c}' {why}.",
                    impact="A column this predictive is almost always derived from the label or recorded after the outcome. Any model will lean on it and the offline score will be fiction.",
                    fix=f"Remove '{c}' unless you can prove it is available, with the same meaning, at prediction time.",
                    columns=[c], auto_handled="Removed for the remaining analysis.")
    for c in leaks:
        P.excluded[c] = "suspected leakage"
    P.features = [c for c in P.features if c not in leaks]
    ctx.R["leak_removed"] = list(leaks)

    # name-based: post-outcome words (only ask, don't remove)
    named = [c for c in P.features if POST_HINT.search(c) and c not in P.uc.unavailable]
    if named:
        ctx.add(severity="info", dimension="leakage", title=f"Check timing of: {', '.join(named[:5])}",
                evidence="These column names suggest something that happens after (or because of) the outcome.",
                impact="Information recorded after the event you are predicting is the most common source of leakage, and it is invisible to statistical tests if the correlation is moderate.",
                fix="For each, ask: 'At the moment I make this prediction, would I already know this value?' If not, mark it as unavailable.",
                columns=named)
