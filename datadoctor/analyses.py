"""Model-based analyses: signal, relevance, validation realism, model bake-off, subgroups, data-size, labels, training forecast."""
from __future__ import annotations

import os
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler

from .checks import Ctx
from .core import SEED, fmt, pct
from .evaluation import HIGHER, baseline_score, cv_eval, metric_set, score, skill, splitter, zoo
from .prep import encode, sample_idx

warnings.filterwarnings("ignore")


def hgb(task, **kw):
    kw.setdefault("random_state", SEED)
    return HistGradientBoostingClassifier(**kw) if task == "classification" else HistGradientBoostingRegressor(**kw)


def build_matrix(ctx: Ctx):
    P = ctx.P
    idx = sample_idx(P)
    X, origin = encode(P.df.iloc[idx], P.features, ctx.prof)
    ctx.idx, ctx.Xdf, ctx.origin = idx, X, origin
    ctx.X = X.to_numpy(float)
    ctx.y = P.y[idx]
    ctx.k = len(P.classes) if P.task == "classification" else 0
    ctx.t = P.t.iloc[idx].reset_index(drop=True) if P.t is not None else None
    ctx.g = P.g.iloc[idx].to_numpy() if P.g is not None else None
    ctx.R["matrix"] = {"rows_used": len(idx), "rows_total": len(P.df), "raw_features": len(P.features),
                       "encoded_features": X.shape[1]}
    # how validation will be done for everything downstream
    ctx.split_kind = "time" if ctx.t is not None else "group" if ctx.g is not None else "random"


def budget(ctx, iters):
    """Diagnostic models: multiclass boosting grows one tree per class per round, so scale rounds down with class count."""
    return iters if ctx.k <= 2 else max(20, int(iters * 2 / ctx.k ** .5 / 1.5))


def has_signal(ctx, min_skill=.1):
    """Model-based diagnostics (label noise, subgroup gaps) are only meaningful when the model learned something."""
    return ctx.R.get("signal", {}).get("skill", 1) >= min_skill


def signal_metric(ctx):
    if ctx.P.task == "regression":
        return "r2"
    return "roc_auc" if ctx.k == 2 else "balanced_accuracy"


# ============================================================================= volume
def volume(ctx: Ctx):
    P = ctx.P
    n, p = len(P.df), ctx.X.shape[1]
    ctx.R["volume"] = {"rows": n, "encoded_features": p, "rows_per_feature": n / max(p, 1)}
    if P.task == "classification":
        counts = np.bincount(P.y)
        events = counts.min()
        epv = events / max(p, 1)
        ctx.R["volume"]["events_per_variable"] = epv
        if events < 50:
            ctx.add(severity="critical" if events < 20 else "major", dimension="volume",
                    title=f"Only {events} examples of the rarest class",
                    evidence=f"{n:,} rows in total, but the smallest class has {events}.",
                    impact="What limits learning is the number of minority examples, not total rows. With this few, any model's estimate of that class is unstable and test metrics have huge error bars.",
                    fix="Collect more minority examples (or label more data). Use repeated stratified CV and report confidence intervals.")
        elif epv < 10:
            ctx.add(severity="minor", dimension="volume", title=f"Low events-per-feature ratio ({epv:.1f})",
                    evidence=f"{events} minority-class rows for {p} encoded features. The common rule of thumb is at least 10-20 per feature.",
                    impact="Flexible models will overfit; coefficients of linear models will be unstable.",
                    fix="Reduce features (drop the irrelevant ones listed in the relevance section, merge rare categories) or use strong regularisation.")
    else:
        if n < 20 * p:
            ctx.add(severity="major" if n < 5 * p else "minor", dimension="volume",
                    title=f"Few rows per feature ({n / max(p, 1):.1f})",
                    evidence=f"{n:,} rows for {p} encoded features.",
                    impact="High-variance models will fit noise; expect a large gap between train and validation error.",
                    fix="Reduce dimensionality (drop irrelevant columns, collapse categories) or use regularised linear models.")
    if n < 200:
        ctx.add(severity="major", dimension="volume", title=f"Very small dataset ({n} rows)",
                evidence="Below the size where held-out estimates are reliable.",
                impact="A single unlucky split can move the score by 10+ points. Model comparisons are mostly noise.",
                fix="Use repeated k-fold CV, prefer simple models, and treat every number in this report as a wide range.")


# ============================================================================= signal
def signal(ctx: Ctx):
    P = ctx.P
    m = signal_metric(ctx)
    n = min(3000 if ctx.k <= 2 else 2000, len(ctx.y))
    sel = np.linspace(0, len(ctx.y) - 1, n).astype(int)
    X, y = ctx.X[sel], ctx.y[sel]
    model = hgb(P.task, max_iter=budget(ctx, 60), learning_rate=.15, max_leaf_nodes=15, early_stopping=False)
    def cv_score(yy):
        sp = splitter("random", P.task, 3, y=yy)
        return cv_eval(model, X, yy, sp, P.task, [m], P.pos, ctx.k, keep_oof=False)["mean"][m]
    real = cv_score(y)
    rng = np.random.default_rng(SEED)
    null = np.array([cv_score(rng.permutation(y)) for _ in range(10)])
    pval = (1 + np.sum(null >= real)) / (1 + len(null))
    base = baseline_score(m, y, P.task, P.pos, ctx.k) if m != "r2" else 0.0
    sk = skill(m, real, base)
    z = (real - null.mean()) / (null.std() + 1e-9)
    ctx.R["signal"] = {"metric": m, "real": real, "null": null.tolist(), "p": pval, "baseline": base, "skill": sk, "z": z}
    label = {"roc_auc": "ROC-AUC", "balanced_accuracy": "balanced accuracy", "r2": "R²"}[m]
    if z < 3 or sk < .03:
        ctx.add(severity="critical", dimension="signal", title="No learnable signal: the features don't predict the target",
                evidence=f"With real labels, gradient boosting scores {label} {real:.3f}. With randomly shuffled labels it scores {null.mean():.3f} ± {null.std():.3f}. "
                         f"The real score is only {z:.1f} standard deviations above pure chance.",
                impact="Any model you train will be guessing. Bigger models, more epochs or fine-tuning cannot create information that is not in the columns.",
                fix="Go back to the problem: what information would a human expert use to predict this? Collect those features (history, behaviour, context), or redefine the target.")
    elif sk < .15:
        ctx.add(severity="major", dimension="signal", title="Weak signal",
                evidence=f"Real-label {label} {real:.3f} vs {null.mean():.3f} for shuffled labels; that is only {pct(sk)} of the way from guessing to perfect.",
                impact="There IS a pattern, but it explains little. Expect a low performance ceiling; the choice of model will matter far less than better features.",
                fix="Invest in feature engineering (aggregates over history, ratios, domain variables) before tuning models.")
    else:
        ctx.add(severity="info", dimension="signal", title=f"Clear signal ({label} {real:.2f} vs {null.mean():.2f} by chance)",
                evidence=f"Real labels beat all 10 shuffled-label runs by {z:.0f} standard deviations (p ≤ {pval:.2f}).",
                impact="The features contain real, learnable information about the target.",
                fix="Proceed to model selection.")


# ============================================================================= relevance
def relevance(ctx: Ctx):
    P = ctx.P
    m = signal_metric(ctx)
    Xtr, Xte, ytr, yte = train_test_split(ctx.X, ctx.y, test_size=.3, random_state=SEED,
                                          stratify=ctx.y if P.task == "classification" else None)
    model = hgb(P.task, max_iter=budget(ctx, 100), early_stopping=False).fit(Xtr, ytr)
    def sc(Xe):
        pred = model.predict(Xe)
        proba = model.predict_proba(Xe) if P.task == "classification" else None
        if proba is not None and proba.shape[1] < ctx.k:
            full = np.zeros((len(Xe), ctx.k)); full[:, model.classes_] = proba; proba = full
        return score(m, yte, pred, proba, P.pos, ctx.k)
    base = sc(Xte)
    raw_cols = list(dict.fromkeys(ctx.origin.values()))
    colidx = {c: [i for i, k in enumerate(ctx.Xdf.columns) if ctx.origin[k] == c] for c in raw_cols}
    rng = np.random.default_rng(SEED)
    imp, imp_sd = {}, {}
    for c, ii in colidx.items():
        drops = []
        for _ in range(4):
            Xp = Xte.copy()
            perm = rng.permutation(len(Xp))
            Xp[:, ii] = Xp[perm][:, ii]
            drops.append(base - sc(Xp))
        imp[c], imp_sd[c] = float(np.mean(drops)), float(np.std(drops))
    order = sorted(imp, key=lambda c: -imp[c])
    total = sum(max(v, 0) for v in imp.values()) or 1
    useless = [c for c in order if imp[c] <= max(2 * imp_sd[c], .001)]
    ctx.R["relevance"] = {"metric": m, "holdout_score": base, "importance": {c: imp[c] for c in order},
                          "sd": {c: imp_sd[c] for c in order}, "useless": useless}
    top = order[0] if order else None
    if top and imp[top] / total > .6 and len(order) > 3:
        ctx.add(severity="minor", dimension="leakage", title=f"One feature does most of the work: '{top}' ({pct(imp[top] / total)} of importance)",
                evidence=f"Shuffling '{top}' drops held-out {m} by {imp[top]:.3f}; the next feature by {imp[order[1]]:.3f}.",
                impact="The model is betting on a single cue. If that column changes meaning, is filled differently in production, or is a proxy for the answer, the whole model fails at once.",
                fix=f"Verify '{top}' is legitimate and stable over time; try a model without it to see what else carries signal.",
                columns=[top])
    if useless and len(useless) >= max(2, .3 * len(order)):
        ctx.add(severity="info", dimension="signal", title=f"{len(useless)} of {len(order)} features show no measurable contribution",
                evidence="Shuffling them does not change held-out performance: " + ", ".join(useless[:10]) + ("..." if len(useless) > 10 else "") + ".",
                impact=("They add dimensions, training time and overfitting risk without adding information (at least not in this model)."
                        + (" Caveat: this dataset has redundant columns, and shuffling one of a correlated pair lets its twin take over, "
                           "so some of these are 'useless' only because a copy exists." if ctx.R.get("redundant") else "")),
                fix="Try dropping them (or one of each redundant pair) and confirm the score does not drop; keep only if domain knowledge says they matter.",
                columns=useless[:20])


# ============================================================================= validation realism
def validity(ctx: Ctx):
    P = ctx.P
    m = signal_metric(ctx)
    model = hgb(P.task, max_iter=budget(ctx, 100), early_stopping=False, learning_rate=.1)
    res = {"metric": m}
    rnd = cv_eval(model, ctx.X, ctx.y, splitter("random", P.task, 5, y=ctx.y), P.task, [m], P.pos, ctx.k, keep_oof=False)
    res["random"] = rnd["mean"][m]
    label = {"roc_auc": "ROC-AUC", "balanced_accuracy": "balanced accuracy", "r2": "R²"}[m]
    if ctx.t is not None:
        # ctx.X rows are sorted by time already (sample_idx)
        tm = cv_eval(model, ctx.X, ctx.y, splitter("time", P.task, 5, y=ctx.y), P.task, [m], P.pos, ctx.k, keep_oof=False)
        res["time"] = tm["mean"][m]
        res["time_folds"] = tm["folds"][m]
        drop = res["random"] - res["time"]
        if drop > .03:
            ctx.add(severity="major" if drop > .08 else "minor", dimension="validity",
                    title=f"Random splits overestimate performance ({label} {res['random']:.2f} → {res['time']:.2f} on future data)",
                    evidence=f"Shuffled 5-fold CV gives {res['random']:.3f}; training on the past and testing on the next period gives {res['time']:.3f}.",
                    impact="In production you always predict the future from the past. A random split lets the model peek at neighbouring dates, so the optimistic number is the one you would wrongly report.",
                    fix=f"Use a time-based split (train on earlier '{P.uc.time_column}', test on later) for all model selection and reporting.",
                    auto_handled="All model comparisons in this report use forward-in-time splits.")
        # target drift over time
        tt = ctx.t
        bins = pd.qcut(tt.rank(method="first"), 8, labels=False)
        lbl = [str(pd.Timestamp(tt[bins == b].min()).date()) if hasattr(tt.iloc[0], "year") else fmt(tt[bins == b].min()) for b in range(8)]
        if P.task == "classification":
            target_k = P.pos if P.pos is not None else int(np.bincount(ctx.y).argmax())
            vals = [float((ctx.y[bins.to_numpy() == b] == target_k).mean()) for b in range(8)]
            tab = np.array([[np.sum((bins.to_numpy() == b) & (ctx.y == k)) for k in range(ctx.k)] for b in range(8)])
            tab = tab[:, tab.sum(0) > 0]
            p = stats.chi2_contingency(tab)[1] if tab.shape[1] > 1 else 1
            what = f"share of {P.classes[target_k]}"
        else:
            vals = [float(np.mean(ctx.y[bins.to_numpy() == b])) for b in range(8)]
            p = stats.kruskal(*[ctx.y[bins.to_numpy() == b] for b in range(8)])[1]
            what = f"mean {P.uc.target}"
        res["drift"] = {"labels": lbl, "values": vals, "what": what, "p": float(p)}
        rel = (max(vals) - min(vals)) / (abs(np.mean(vals)) + 1e-9)
        if p < .001 and rel > .2:
            ctx.add(severity="minor", dimension="validity", title=f"The target changes over time ({what} from {fmt(vals[0])} to {fmt(vals[-1])})",
                    evidence=f"Across 8 time periods the {what} ranges {fmt(min(vals))} to {fmt(max(vals))} (p={p:.1g}).",
                    impact="A model learns the average of the past. If the base rate keeps moving, predictions will be systematically off and probability calibration will degrade.",
                    fix="Retrain on a schedule, weight recent data more, and monitor the prediction rate vs the actual rate after deployment.")
        # adversarial validation: can a model tell early rows from late rows?
        order_ = np.arange(len(ctx.y))
        proxies = sorted({ctx.origin[c] for i, c in enumerate(ctx.Xdf.columns)
                          if ctx.origin[c] != P.uc.time_column and np.std(ctx.X[:, i]) > 0
                          and abs(stats.spearmanr(ctx.X[:, i], order_)[0]) > .8})
        res["time_proxies"] = proxies
        own = [i for i, c in enumerate(ctx.Xdf.columns) if ctx.origin[c] == P.uc.time_column or ctx.origin[c] in proxies]
        keep = [i for i in range(ctx.X.shape[1]) if i not in own]
        if keep:
            half = (np.arange(len(ctx.y)) >= len(ctx.y) // 2).astype(int)
            Xa = ctx.X[:, keep]
            adv = cv_eval(HistGradientBoostingClassifier(max_iter=100, random_state=SEED), Xa, half,
                          splitter("random", "classification", 3, y=half), "classification", ["roc_auc"], 1, 2, keep_oof=False)
            auc = adv["mean"]["roc_auc"]
            res["adversarial_auc"] = auc
            if auc > .65:
                m_ = HistGradientBoostingClassifier(max_iter=100, random_state=SEED).fit(Xa, half)
                cols = ctx.Xdf.columns[keep]
                rng = np.random.default_rng(SEED)
                base_ = score("roc_auc", half, None, m_.predict_proba(Xa), 1, 2)
                drift_imp = {}
                for raw in dict.fromkeys(ctx.origin[c] for c in cols):
                    ii = [j for j, c in enumerate(cols) if ctx.origin[c] == raw]
                    Xp = Xa.copy(); Xp[:, ii] = Xp[rng.permutation(len(Xp))][:, ii]
                    drift_imp[raw] = base_ - score("roc_auc", half, None, m_.predict_proba(Xp), 1, 2)
                top = sorted(drift_imp, key=lambda c: -drift_imp[c])[:5]
                res["drift_features"] = {c: drift_imp[c] for c in top}
                ctx.add(severity="major" if auc > .8 else "minor", dimension="validity",
                        title="Feature distributions shift between older and newer data",
                        evidence=f"A classifier can tell first-half rows from second-half rows with ROC-AUC {auc:.2f} (0.5 = indistinguishable). "
                                 f"Biggest movers: {', '.join(top[:3])}." + (f" (Time proxies {', '.join(proxies)} were excluded from this test.)" if proxies else ""),
                        impact="The model will be applied to data that looks different from what it was trained on (covariate shift). Patterns tied to the drifting features may not transfer.",
                        fix=f"Monitor {', '.join(top[:3])} in production; prefer features that are stable over time; retrain regularly.",
                        columns=top)
    if ctx.g is not None:
        gp = cv_eval(model, ctx.X, ctx.y, splitter("group", P.task, 5, y=ctx.y, groups=ctx.g), P.task, [m], P.pos, ctx.k, keep_oof=False)
        res["group"] = gp["mean"][m]
        rpe = len(ctx.g) / len(np.unique(ctx.g))
        res["rows_per_entity"] = rpe
        drop = res["random"] - res["group"]
        if drop > .03:
            ctx.add(severity="major" if drop > .08 else "minor", dimension="validity",
                    title=f"Scores drop on unseen {P.uc.entity_column} values ({res['random']:.2f} → {res['group']:.2f})",
                    evidence=f"Each {P.uc.entity_column} appears in {rpe:.1f} rows on average. Random CV puts the same entity in train and test; grouped CV does not.",
                    impact="Part of the random-split score is the model recognising entities it has already seen. For new customers/patients/devices, expect the lower number.",
                    fix=f"Split by {P.uc.entity_column} (GroupKFold) if the model will score new entities; build per-entity history features if it scores known ones.")
        else:
            ctx.add(severity="info", dimension="validity", title=f"No entity memorisation detected for '{P.uc.entity_column}'",
                    evidence=f"Random CV {res['random']:.3f} vs grouped CV {res['group']:.3f}.",
                    impact="Performance should carry over to new entities.", fix="Still split by entity for final evaluation to be safe.")
    if ctx.t is None and ctx.g is None:
        ctx.add(severity="info", dimension="validity", title="No time or entity column declared: random splits assumed valid",
                evidence="The audit can only check split realism when it knows the time order or which rows belong together.",
                impact="If rows are ordered in time or several rows describe the same person/device, random CV will be optimistic.",
                fix="If either applies, set the time / entity column in the use case and re-run.")
    ctx.R["validity"] = res


# ============================================================================= bake-off
def bakeoff(ctx: Ctx):
    P = ctx.P
    balanced = P.task == "classification" and (np.bincount(ctx.y).max() / max(np.bincount(ctx.y).min(), 1) > 3 or P.goal == "recall")
    skewed = P.task == "regression" and stats.skew(ctx.y) > 2 and (ctx.y >= 0).all()
    models = zoo(P.task, balanced=balanced, skewed_target=skewed, fast=len(ctx.y) > 4000)
    splits = splitter(ctx.split_kind, P.task, 5, y=ctx.y, groups=ctx.g)
    mets = metric_set(P.task, ctx.k)
    if P.metric not in mets:
        mets.append(P.metric)
    res = {}
    base = {mm: baseline_score(mm, ctx.y, P.task, P.pos, ctx.k) for mm in mets}
    for i, (name, spec) in enumerate(models.items()):
        ctx.progress(sub=f"training {name}")
        r = cv_eval(spec["model"], ctx.X, ctx.y, splits, P.task, mets, P.pos, ctx.k)
        r.update(family=spec["family"], interp=spec["interp"], about=spec["about"],
                 skill=skill(P.metric, r["mean"][P.metric], base[P.metric]))
        res[name] = r
    ctx.models = models
    ctx.bake = res
    ctx.R["bakeoff"] = {"split": ctx.split_kind, "folds": len(splits), "metric": P.metric, "baseline": base,
                        "balanced_weights": balanced,
                        "models": {k: {kk: v[kk] for kk in ("mean", "std", "folds", "fit_time", "latency_ms", "size_kb",
                                                             "family", "interp", "about", "skill", "train_rows")}
                                   for k, v in res.items()}}


# ============================================================================= threshold (binary)
def threshold(ctx: Ctx, best):
    P = ctx.P
    if P.task != "classification" or ctx.k != 2:
        return
    r = ctx.bake[best]
    te = r["tested"]
    pr = r["oof_proba"][te, P.pos]
    yt = (ctx.y[te] == P.pos).astype(int)
    ths = np.linspace(.05, .95, 37)
    rows = []
    beta = {"recall": 2, "precision": .5}.get(P.goal, 1)
    for t in ths:
        pred = (pr >= t).astype(int)
        tp = np.sum((pred == 1) & (yt == 1)); fp = np.sum((pred == 1) & (yt == 0)); fn = np.sum((pred == 0) & (yt == 1))
        prec = tp / (tp + fp) if tp + fp else 1.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f = (1 + beta ** 2) * prec * rec / (beta ** 2 * prec + rec) if prec + rec else 0
        rows.append((t, prec, rec, f, int(tp + fp)))
    best_i = int(np.argmax([r_[3] for r_ in rows]))
    at05 = rows[int(np.argmin(np.abs(ths - .5)))]
    ctx.R["threshold"] = {"thresholds": ths.tolist(), "precision": [r_[1] for r_ in rows], "recall": [r_[2] for r_ in rows],
                          "f": [r_[3] for r_ in rows], "beta": beta, "best": rows[best_i][0], "best_row": rows[best_i],
                          "default_row": at05, "positives": int(yt.sum()), "n": int(len(yt))}


# ============================================================================= subgroups
def subgroups(ctx: Ctx, best):
    P = ctx.P
    cols = [c for c in P.uc.group_columns if c in P.df.columns]
    auto = False
    if not cols:
        cols = [c for c, p in ctx.prof.items() if p["kind"] in ("categorical", "boolean") and 2 <= p["unique"] <= 8
                and c in P.features][:2]
        auto = True
    r = ctx.bake[best]
    te = r["tested"]
    yv = ctx.y[te]
    pred = r["oof_pred"][te]
    proba = r["oof_proba"][te] if r["oof_proba"] is not None else None
    m = P.metric if P.metric != "roc_auc" else ("balanced_accuracy" if ctx.k > 2 else "roc_auc")
    overall = score(m, yv, pred, proba, P.pos, ctx.k)
    out = {"metric": m, "overall": overall, "columns": {}, "auto": auto}
    flagged = []
    for c in cols:
        vals = P.df[c].iloc[ctx.idx].iloc[te].astype(str).where(P.df[c].iloc[ctx.idx].iloc[te].notna(), "(missing)").to_numpy()
        levels = pd.Series(vals).value_counts().head(10)
        rows = []
        for lev, cnt in levels.items():
            mask = vals == lev
            s = score(m, yv[mask], pred[mask], proba[mask] if proba is not None else None, P.pos, ctx.k) if cnt >= 15 else np.nan
            rate = float((yv[mask] == P.pos).mean()) if P.pos is not None else float(np.mean(yv[mask])) if P.task == "regression" else None
            rows.append({"level": lev, "n": int(cnt), "share": cnt / len(vals), "score": s, "rate": rate})
            if cnt >= 50 and np.isfinite(s) and has_signal(ctx) and lev != "(missing)":
                gap = (overall - s) if HIGHER[m] else (s - overall) / max(overall, 1e-9)
                if gap > (.08 if HIGHER[m] else .3):
                    flagged.append((c, lev, s, cnt))
        out["columns"][c] = rows
    ctx.R["subgroups"] = out
    for c, lev, s, cnt in flagged[:4]:
        ctx.add(severity="major" if c in P.uc.group_columns else "minor", dimension="fairness",
                title=f"Weaker performance for {c} = '{lev}'",
                evidence=f"{m} is {fmt(s, 3)} for this group ({cnt} rows) vs {fmt(overall, 3)} overall.",
                impact="An average score hides who the model fails. If decisions affect people or regions differently, this gap is the number that matters.",
                fix="Check whether the group is under-represented or has a different relationship with the target; consider more data for it, group-aware features, or a separate threshold.",
                columns=[c])
    for c, rows in out["columns"].items():
        tiny = [r_["level"] for r_ in rows if r_["share"] < .02 and r_["level"] != "(missing)"]
        if tiny and c in P.uc.group_columns:
            ctx.add(severity="minor", dimension="fairness", title=f"Under-represented groups in '{c}': {', '.join(tiny[:4])}",
                    evidence="Each is under 2% of rows.", impact="The model sees few examples of these groups and their metrics are unreliable.",
                    fix="Collect more data for them or report their metrics with confidence intervals.", columns=[c])


# ============================================================================= data size
def data_size(ctx: Ctx, best):
    P = ctx.P
    spec = ctx.models[best]["model"]
    m = P.metric
    fr = np.array([.1, .2, .35, .5, .75, 1.0])
    splits = splitter("random", P.task, 3, y=ctx.y)
    tr_s, va_s, ns = [], [], []
    rng = np.random.default_rng(SEED)
    for f in fr:
        a, b = [], []
        for tr, te in splits:
            sub = rng.choice(tr, max(20, int(len(tr) * f)), replace=False)
            if P.task == "classification" and len(np.unique(ctx.y[sub])) < ctx.k:
                continue
            mdl = clone(spec).fit(ctx.X[sub], ctx.y[sub])
            def ev(ii):
                pred = mdl.predict(ctx.X[ii])
                pr = None
                if P.task == "classification":
                    pp = mdl.predict_proba(ctx.X[ii]); pr = np.zeros((len(ii), ctx.k)); pr[:, np.unique(ctx.y[sub])] = pp
                return score(m, ctx.y[ii], pred, pr, P.pos, ctx.k)
            a.append(ev(sub)); b.append(ev(te))
        if b:
            tr_s.append(float(np.mean(a))); va_s.append(float(np.mean(b))); ns.append(int(len(splits[0][0]) * f))
    out = {"n": ns, "train": tr_s, "val": va_s, "metric": m}
    # extrapolate validation error with an inverse power law: err(n) = a + b * n^-c
    if len(ns) >= 4:
        err = np.array([1 - v for v in va_s]) if HIGHER[m] else np.array(va_s)
        try:
            f_ = lambda n, a, b, c: a + b * np.power(n, -c)
            popt, _ = curve_fit(f_, np.array(ns, float), err, p0=[err[-1] * .8, err[0], .5], bounds=([0, 0, .01], [np.inf, np.inf, 2]), maxfev=20000)
            N = ns[-1]          # largest measured training size; extrapolate at most 4x beyond it
            xs = np.array([N * k for k in (1, 2, 4)], float)
            pred = f_(xs, *popt)
            conv = (lambda e: 1 - e) if HIGHER[m] else (lambda e: e)
            out["extrapolation"] = {"n": xs.tolist(), "score": [float(conv(p)) for p in pred], "floor": float(conv(popt[0]))}
            gain = abs(conv(pred[1]) - conv(pred[0]))
            rel = gain / (abs(conv(pred[0])) + 1e-9) if not HIGHER[m] else gain
            out["gain_2x"] = float(gain)
            gap = tr_s[-1] - va_s[-1] if HIGHER[m] else (va_s[-1] - tr_s[-1]) / max(abs(va_s[-1]), 1e-9)
            out["gap"] = float(gap)
            if rel > (.02 if HIGHER[m] else .03):
                ctx.add(severity="info", dimension="volume", title="More data would still help",
                        evidence=f"The validation curve is still improving. A fitted learning curve predicts {m} ≈ {fmt(conv(pred[0]), 3)} at {N:,} training rows and ≈ {fmt(conv(pred[1]), 3)} at {2 * N:,}.",
                        impact="You are data-limited: collecting more labelled rows is a better investment than heavier models.",
                        fix="Prioritise labelling more data (especially of the rarer cases).")
            else:
                ctx.add(severity="info", dimension="volume", title="More rows alone will not help much",
                        evidence=f"The learning curve has flattened: doubling the training rows ({N:,} → {2 * N:,}) is predicted to move {m} by only {fmt(gain, 3)}.",
                        impact="You are feature-limited (or near the noise ceiling), not data-limited.",
                        fix="Spend effort on better / new features or cleaner labels rather than more rows of the same kind.")
        except Exception:
            pass
    ctx.R["data_size"] = out


# ============================================================================= label quality
def label_quality(ctx: Ctx, best):
    P = ctx.P
    r = ctx.bake[best]
    te = r["tested"]
    rows_df = P.df.iloc[ctx.idx].iloc[te]
    show_cols = [c for c in (P.features[:6]) if c in rows_df.columns]
    if P.task == "classification":
        Pm = r["oof_proba"][te]
        yv = ctx.y[te]
        if Pm is None or np.isnan(Pm).any():
            return
        thr = np.array([Pm[yv == j, j].mean() if (yv == j).any() else 1 for j in range(ctx.k)])
        own = Pm[np.arange(len(yv)), yv]
        other = Pm.copy(); other[np.arange(len(yv)), yv] = -1
        sus = (own < thr[yv]) & (other.max(1) >= thr[other.argmax(1)]) & (own < .2)
        rate = float(sus.mean())
        order = np.argsort(own[sus])[:15]
        si = np.where(sus)[0][order]
        examples = [{"row": int(rows_df.index[i]), "given": P.classes[yv[i]], "model_says": P.classes[int(other[i].argmax())],
                     "confidence": float(1 - own[i]), **{c: str(rows_df.iloc[i][c]) for c in show_cols}} for i in si]
        h, _ = np.histogram(own, bins=20, range=(0, 1))
        ctx.R["labels"] = {"type": "classification", "suspect_rate": rate, "n_suspect": int(sus.sum()), "examples": examples,
                           "own_conf_hist": h.tolist(), "columns": ["row", "given", "model_says", "confidence"] + show_cols}
        ctx.R["labels"]["reliable"] = has_signal(ctx)
        if rate > .03 and has_signal(ctx):
            ctx.add(severity="major" if rate > .1 else "minor", dimension="labels",
                    title=f"About {pct(rate, 1)} of labels look wrong",
                    evidence=f"{int(sus.sum())} rows get less than 20% probability for their own label while the model is confident in another class (out-of-fold, simplified confident learning).",
                    impact="Mislabeled rows teach the model wrong patterns and also make your test score inaccurate. Flexible models will memorise them.",
                    fix="Review the listed rows. If many are genuinely wrong, fix labels at the source; otherwise they reveal missing features.")
    else:
        pred = r["oof_pred"][te]
        res = ctx.y[te] - pred
        mad = np.median(np.abs(res - np.median(res))) + 1e-9
        z = .6745 * (res - np.median(res)) / mad
        sus = np.abs(z) > 6
        order = np.argsort(-np.abs(z[sus]))[:15]
        si = np.where(sus)[0][order]
        examples = [{"row": int(rows_df.index[i]), "actual": float(ctx.y[te][i]), "predicted": float(pred[i]), "robust_z": float(z[i]),
                     **{c: str(rows_df.iloc[i][c]) for c in show_cols}} for i in si]
        ctx.R["labels"] = {"type": "regression", "suspect_rate": float(sus.mean()), "n_suspect": int(sus.sum()),
                           "examples": examples, "columns": ["row", "actual", "predicted", "robust_z"] + show_cols}
        ctx.R["labels"]["reliable"] = has_signal(ctx)
        if sus.mean() > .01 and has_signal(ctx):
            ctx.add(severity="minor", dimension="labels", title=f"{int(sus.sum())} rows have targets far from anything the features explain",
                    evidence=f"{pct(sus.mean(), 1)} of rows have out-of-fold residuals beyond 6 robust SDs.",
                    impact="Either data errors in the target or rare events the features cannot see. With squared-error losses these rows dominate training.",
                    fix="Inspect the listed rows; fix errors, and consider a robust loss (Huber / absolute) if they are genuine.")


# ============================================================================= training forecast
def training_forecast(ctx: Ctx, best):
    P = ctx.P
    out = {}
    X, y = ctx.X, ctx.y
    strat = y if P.task == "classification" else None
    # 1) gradient boosting: loss per boosting round
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=.25, random_state=SEED, stratify=strat)
    lr = .1
    m = hgb(P.task, max_iter=400, learning_rate=lr, early_stopping=True, validation_fraction=None, n_iter_no_change=400,
            scoring="loss")
    # use our own validation set: fit on train, evaluate staged loss via warm-start chunks
    tr_l, va_l = [], []
    m.set_params(early_stopping=False, warm_start=True)
    top, step = (400, 10) if ctx.k <= 2 else (240, 12)
    its = list(range(step, top + 1, step))
    ev = np.arange(len(ytr))[:1500]          # training loss on a fixed subset is enough to draw the curve
    for it in its:
        m.set_params(max_iter=it)
        m.fit(Xtr, ytr)
        tr_l.append(_loss(P.task, m, Xtr[ev], ytr[ev], ctx.k)); va_l.append(_loss(P.task, m, Xva, yva, ctx.k))
    out["boosting"] = {"x": its, "train": tr_l, "val": va_l, "xlabel": "boosting rounds (trees)", "lr": lr,
                       "loss": "log-loss" if P.task == "classification" else "MSE (scaled)", **_read_curve(its, tr_l, va_l)}
    # 2) MLP: loss per epoch
    sc = StandardScaler().fit(Xtr)
    A, B = sc.transform(Xtr), sc.transform(Xva)
    ys = None
    if P.task == "regression":
        mu, sd = ytr.mean(), ytr.std() + 1e-9
        ys = (mu, sd)
    bs = int(min(128, max(16, len(A) // 20)))
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), batch_size=bs, learning_rate_init=1e-3, random_state=SEED) if P.task == "classification" \
        else MLPRegressor(hidden_layer_sizes=(64, 32), batch_size=bs, learning_rate_init=1e-3, random_state=SEED)
    trl, val, best_v, best_e = [], [], np.inf, 1
    t0 = time.perf_counter()
    for ep in range(1, 201):
        if P.task == "classification":
            mlp.partial_fit(A, ytr, classes=np.arange(ctx.k))
        else:
            mlp.partial_fit(A, (ytr - ys[0]) / ys[1])
        trl.append(_loss(P.task, mlp, A, ytr, ctx.k, ys)); v = _loss(P.task, mlp, B, yva, ctx.k, ys); val.append(v)
        if v < best_v - 1e-4:
            best_v, best_e = v, ep
        if ep - best_e >= 30 and ep >= 40:
            break
    mlp_time = time.perf_counter() - t0
    eps = list(range(1, len(trl) + 1))
    out["mlp"] = {"x": eps, "train": trl, "val": val, "xlabel": "epochs", "batch": bs, "lr": 1e-3,
                  "steps_per_epoch": int(np.ceil(len(A) / bs)), "sec_per_epoch": mlp_time / len(eps),
                  "loss": "log-loss" if P.task == "classification" else "MSE (standardised)", **_read_curve(eps, trl, val)}
    # 3) time estimates for the recommended model: measure fit time at 3 sizes, fit a power law, extrapolate
    spec = ctx.models[best]["model"]
    sizes, times = [], []
    for f in (.25, .5, 1.0):
        n_ = max(50, int(len(y) * f))
        sel = np.linspace(0, len(y) - 1, n_).astype(int)
        t0 = time.perf_counter(); clone(spec).fit(X[sel], y[sel]); times.append(time.perf_counter() - t0); sizes.append(n_)
    b = np.polyfit(np.log(sizes), np.log(np.maximum(times, 1e-4)), 1)[0]
    b = float(np.clip(b, .7, 2.2))
    N = len(P.df)
    t_full = times[-1] * (N / sizes[-1]) ** b
    est = {"exponent": b, "measured": {"rows": sizes, "seconds": times}, "full_rows": N,
           "single_fit_full": t_full, "cv5_full": 5 * t_full, "search30_full": 30 * 5 * t_full,
           "single_fit_10x": t_full * 10 ** b, "cpu": os.cpu_count()}
    out["time"] = est
    out["recommended"] = best
    ctx.R["training"] = out


def _loss(task, m, X, y, k, ys=None):
    if task == "classification":
        p = m.predict_proba(X)
        full = np.full((len(X), k), 1e-9)
        cls = m.classes_
        full[:, cls] = p
        return float(log_loss(y, np.clip(full, 1e-9, 1), labels=list(range(k))))
    pred = m.predict(X)
    if ys is not None:
        return float(mean_squared_error((y - ys[0]) / ys[1], pred))
    return float(mean_squared_error(y, pred) / (np.var(y) + 1e-12))


def _read_curve(x, tr, va):
    """Turn a train/val loss curve into plain-language observations."""
    tr, va = np.array(tr), np.array(va)
    bi = int(np.argmin(va))
    best_x = x[bi]
    gap = float(va[bi] - tr[bi])
    rise = float((va[-1] - va[bi]) / (abs(va[bi]) + 1e-9))
    tail = va[-max(3, len(va) // 5):]
    noise = float(np.std(np.diff(va)) / (abs(np.mean(va)) + 1e-9))
    notes = []
    if bi >= len(va) - 2 and rise < .01:
        state = "still_improving"
        notes.append(f"Validation loss was still falling at the last step ({x[-1]}). Train longer or raise the learning rate.")
    elif rise > .05:
        state = "overfits"
        notes.append(f"Validation loss bottoms out at {best_x} and then rises {pct(rise)}: classic overfitting. Stop at ~{best_x} (early stopping) or regularise.")
    else:
        state = "plateau"
        notes.append(f"Validation loss flattens around {best_x}; further training gives little benefit but also does little harm.")
    rel_gap = gap / (abs(va[bi]) + 1e-9)
    if rel_gap > .3:
        notes.append(f"At the best point, training loss is {pct(rel_gap)} lower than validation loss: the model fits the training set much better than new data (high variance).")
    elif rel_gap < .05 and va[bi] > .5 * va[0]:
        notes.append("Train and validation losses are close and both still high: the model is under-fitting (high bias). More capacity or better features would help.")
    if noise > .05:
        notes.append("The validation curve is jumpy: the validation set is small or the learning rate is high. Smooth with a larger batch or a lower learning rate.")
    if va[0] - va[bi] < .02 * abs(va[0]):
        notes.append("Validation loss barely moves from its starting value: the model is learning almost nothing (see signal section).")
    return {"best_x": best_x, "best_val": float(va[bi]), "gap": gap, "rise": rise, "state": state, "notes": notes,
            "patience": int(max(5, round(.15 * best_x)))}
