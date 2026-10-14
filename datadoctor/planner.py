"""Turns measurements into decisions: fitness scores, the model recommendation, a training plan and a written diagnosis."""
from __future__ import annotations

import numpy as np

from .checks import Ctx
from .core import DIMENSIONS, GOALS, METRIC_INFO, SEV_PENALTY, fmt, pct
from .evaluation import HIGHER

SEV_ORDER = {s: i for i, s in enumerate(["critical", "major", "minor", "info"])}


def mlabel(m):
    return METRIC_INFO[m][0]


def fmt_metric(m, v):
    if v is None or not np.isfinite(v):
        return "n/a"
    if m == "mape":
        return pct(v, 1)
    return fmt(v, 3)


def human_time(s):
    if s < 1:
        return "< 1 s"
    if s < 90:
        return f"{s:.0f} s"
    if s < 5400:
        return f"{s / 60:.0f} min"
    return f"{s / 3600:.1f} h"


# ============================================================================= model choice
def choose_model(ctx: Ctx):
    P = ctx.P
    m = P.metric
    hib = HIGHER[m]
    rows = []
    for name, r in ctx.bake.items():
        v = r["mean"][m]
        if v is None or not np.isfinite(v):
            continue
        se = (r["std"][m] or 0) / np.sqrt(max(1, len(r["folds"][m])))
        rows.append((name, v, se, r))
    rows.sort(key=lambda t: -t[1] if hib else t[1])
    best_name, best_v, best_se, _ = rows[0]
    tol = max(best_se, 1e-9)
    within = [t for t in rows if (best_v - t[1] if hib else t[1] - best_v) <= tol]
    reasons = []
    pool = within
    if P.uc.interpretability == "required":
        interp = [t for t in rows if t[3]["interp"] == 3]
        if interp:
            pool = interp[:1]
            cost = abs(best_v - interp[0][1])
            if interp[0][0] == best_name:
                reasons.append("You require an interpretable model, and the best-scoring model is already one a person can read in full: no trade-off needed.")
            else:
                reasons.append(f"You require an interpretable model, so the choice is limited to models a person can read in full. "
                               f"The price of interpretability here is {fmt_metric(m, cost)} {mlabel(m)} versus the best black-box model ({best_name}).")
    if P.uc.deployment == "edge":
        small = [t for t in pool if t[3]["size_kb"] < 2048 and t[3]["latency_ms"] < 1]
        if small:
            if small[0][0] != pool[0][0]:
                reasons.append(f"Edge deployment favours small, fast models; {small[0][0]} is {small[0][3]['size_kb']:.0f} KB with "
                               f"{small[0][3]['latency_ms']:.3f} ms per prediction.")
            pool = small
    key = lambda t: (-t[3]["interp"] if P.uc.interpretability != "not_needed" else 0, t[3]["latency_ms"])
    if len(pool) > 1 and P.uc.interpretability != "required":
        chosen = sorted(pool, key=key)[0] if P.uc.interpretability == "nice" or P.uc.deployment == "edge" else pool[0]
    else:
        chosen = pool[0]
    if chosen[0] != best_name and P.uc.interpretability != "required":
        c3, b3 = chosen[3], ctx.bake[best_name]
        why = ("it is easier to explain" if c3["interp"] > b3["interp"] else
               f"it is smaller and faster ({c3['size_kb']:.0f} KB vs {b3['size_kb']:.0f} KB, {c3['latency_ms']:.3f} vs {b3['latency_ms']:.3f} ms per prediction)")
        reasons.append(f"{chosen[0]} is within one standard error of the top score ({best_name}: {fmt_metric(m, best_v)} vs "
                       f"{fmt_metric(m, chosen[1])}), so the gap is noise. It is preferred because {why} (one-standard-error rule).")
    elif chosen[0] == best_name:
        reasons.append(f"{best_name} has the best cross-validated {mlabel(m)} ({fmt_metric(m, best_v)} ± {fmt_metric(m, ctx.bake[best_name]['std'][m])}).")
        if len(within) > 1:
            others = ", ".join(t[0] for t in within if t[0] != best_name)
            reasons.append(f"{others} {'is' if len(within) == 2 else 'are'} statistically tied with it; keep as a fallback.")
    if ctx.bake[chosen[0]]["skill"] <= .02:
        reasons.insert(0, f"Warning: no model beats the no-skill baseline on {mlabel(m)} (best skill {pct(max(0, ctx.bake[best_name]['skill']), 1)}). "
                          "The ranking below is noise; the choice only matters once the data problems are fixed.")
    simplest = next((t for t in rows if t[3]["family"] == "linear"), None)
    ctx.choice = {"recommended": chosen[0], "best_score": best_name, "reasons": reasons,
                  "ranking": [t[0] for t in rows],
                  "linear_baseline": simplest[0] if simplest else None,
                  "tied": [t[0] for t in within]}
    ctx.R["choice"] = ctx.choice
    return chosen[0]


# ============================================================================= fitness
def fitness(ctx: Ctx):
    dims = {}
    for k, (label, w, q) in DIMENSIONS.items():
        fs = [f for f in ctx.findings if f.dimension == k]
        pen = sum(SEV_PENALTY[f.severity] for f in fs)
        sc = int(round(100 * np.exp(-pen / 90)))   # 1 critical -> 61, 1 major -> 80, 1 minor -> 93
        dims[k] = {"label": label, "weight": w, "question": q, "score": sc,
                   "counts": {s: sum(f.severity == s for f in fs) for s in ("critical", "major", "minor", "info")}}
    sg = ctx.R.get("signal", {})
    if sg:
        s_skill = max(0.0, sg["skill"])
        dims["signal"]["score"] = int(min(dims["signal"]["score"], 100 * (1 - np.exp(-s_skill / .25))))
    overall = sum(d["score"] * d["weight"] for d in dims.values())
    caps = []
    if any(f.severity == "critical" and f.dimension == "signal" for f in ctx.findings):
        overall = min(overall, 30); caps.append("no learnable signal")
    if any(f.severity == "critical" and f.dimension == "leakage" for f in ctx.findings):
        overall = min(overall, 60); caps.append("leakage in the raw data")
    overall = int(round(overall))
    if overall >= 80:
        verdict, tone = "Fit for purpose", "good"
    elif overall >= 60:
        verdict, tone = "Usable after fixes", "warn"
    elif overall >= 40:
        verdict, tone = "Needs major work", "bad"
    else:
        verdict, tone = "Not fit for this use case yet", "bad"
    ctx.R["fitness"] = {"overall": overall, "verdict": verdict, "tone": tone, "dimensions": dims, "caps": caps}


# ============================================================================= plan
HYPER = {
    "boosting": [("learning_rate", "0.02 – 0.2 (log scale)", "lower = more trees needed, usually better"),
                 ("max_leaf_nodes", "15 – 63", "tree complexity"),
                 ("min_samples_leaf", "10 – 100", "raise if overfitting"),
                 ("l2_regularization", "0 – 1", "shrinks leaf values"),
                 ("max_iter", "via early stopping", "let validation loss decide")],
    "ensemble": [("n_estimators", "200 – 600", "more trees = more stable, never overfits by itself"),
                 ("max_features", "sqrt, 0.3, 0.5", "decorrelates trees"),
                 ("min_samples_leaf", "1 – 10", "raise if overfitting"),
                 ("max_depth", "None, 12, 20", "cap model size")],
    "linear": [("C / alpha", "1e-3 – 1e2 (log scale)", "regularisation strength"),
               ("penalty", "l2, l1, elasticnet", "l1 also selects features")],
    "tree": [("max_depth", "3 – 10", "main complexity knob"), ("min_samples_leaf", "5 – 50", "smooths leaves"),
             ("ccp_alpha", "0 – 0.02", "cost-complexity pruning")],
    "instance": [("n_neighbors", "5 – 50", "small = high variance"), ("weights", "uniform, distance", "")],
    "neural": [("hidden_layer_sizes", "(32,), (64,32), (128,64)", "capacity"), ("learning_rate_init", "1e-4 – 1e-2", "most important"),
               ("alpha (L2)", "1e-5 – 1e-2", "weight decay"), ("batch_size", "32 – 256", "")],
}


def plan(ctx: Ctx):
    P = ctx.P
    rec = ctx.choice["recommended"]
    r = ctx.bake[rec]
    fam = r["family"]
    m = P.metric
    steps = []
    crit = [f for f in ctx.findings if f.severity in ("critical", "major")]

    no_signal = any(f.severity == "critical" and f.dimension == "signal" for f in ctx.findings)
    if no_signal:
        steps.append({"title": "Stop: get better features before training anything",
                      "detail": "The signal test shows the current columns cannot predict the target. No model or tuning will fix that. "
                                "Revisit what information drives the outcome and collect it.", "refs": [f.id for f in ctx.findings if f.dimension == "signal"]})

    # 1. data fixes
    fixes = [f for f in crit if f.dimension in ("quality", "labels")]
    if fixes or ctx.R.get("duplicates"):
        steps.append({"title": "Clean the data at the source",
                      "detail": " ".join(dict.fromkeys(f.fix for f in fixes[:5])) or "Deduplicate before splitting.",
                      "refs": [f.id for f in fixes]})
    # 2. columns
    drop = {c: why for c, why in P.excluded.items()}
    if drop:
        steps.append({"title": f"Drop {len(drop)} column(s) from the feature set",
                      "detail": "; ".join(f"{c} ({w})" for c, w in list(drop.items())[:12]) + ".",
                      "refs": [f.id for f in ctx.findings if f.dimension == "leakage" and f.severity != "info"]})
    # 3. preprocessing
    prof = ctx.prof
    kinds = {k: [c for c in P.features if prof[c]["kind"] == k] for k in ("numeric", "categorical", "boolean", "datetime", "time", "text")}
    pre = []
    if kinds["numeric"]:
        pre.append(f"Numeric ({len(kinds['numeric'])}): median-impute + missing flag" +
                   ("; standardise (needed for linear / k-NN / neural models)" if fam in ("linear", "instance", "neural") else "; no scaling needed for tree models"))
    if kinds["categorical"]:
        hc = [c for c in kinds["categorical"] if prof[c]["unique"] > 25]
        pre.append(f"Categorical ({len(kinds['categorical'])}): normalise spellings, one-hot encode, merge levels under 1% into 'other'" +
                   (f"; high-cardinality ({', '.join(hc[:3])}): target-encode inside CV folds" if hc else ""))
    if kinds["datetime"]:
        pre.append(f"Dates ({', '.join(kinds['datetime'])}): derive month, weekday, days-since-start; never feed raw timestamps to the model")
    if kinds["time"]:
        pre.append(f"Time of day ({', '.join(kinds['time'])}): convert to hour (optionally sin/cos encode)")
    if kinds["text"]:
        pre.append(f"Free text ({', '.join(kinds['text'])}): TF-IDF or sentence embeddings as a separate feature block")
    if P.task == "regression" and "log target" in rec:
        pre.append("Target: train on log(1 + y) and invert predictions")
    if ctx.R["bakeoff"].get("balanced_weights"):
        pre.append("Class imbalance: class_weight='balanced' (or resampling inside the training fold only)")
    pre.append("Wrap every step in one sklearn Pipeline so it is fit on training folds only (prevents preprocessing leakage)")
    steps.append({"title": "Build the preprocessing pipeline", "detail": pre, "refs": []})
    # 4. validation
    sk = ctx.split_kind
    val = {"time": f"Forward-in-time validation: sort by '{P.uc.time_column}', use TimeSeriesSplit (5 folds) for model selection and hold out the most recent ~15% as a final test set you touch once.",
           "group": f"Grouped validation: GroupKFold on '{P.uc.entity_column}' so no entity appears in both train and test; keep a group-disjoint final test set.",
           "random": ("Stratified 5-fold CV for selection, plus a stratified 15-20% final test set you touch once." if P.task == "classification"
                      else "5-fold CV for selection, plus a 15-20% final test set you touch once.")}[sk]
    steps.append({"title": "Set up honest validation", "detail": val, "refs": [f.id for f in ctx.findings if f.dimension == "validity"]})
    # 5. model
    base = ctx.R["bakeoff"]["baseline"][m]
    steps.append({"title": f"Train {rec} (baseline to beat: {fmt_metric(m, base)} {mlabel(m)})",
                  "detail": ctx.choice["reasons"] + [r["about"]], "refs": []})
    # 6. tuning
    hp = HYPER.get(fam, [])
    tm = ctx.R["training"]["time"]
    steps.append({"title": "Tune hyperparameters (random search, ~30 trials)",
                  "detail": [f"{a}: {b}" + (f" ({c})" if c else "") for a, b, c in hp] +
                            [f"Estimated cost on the full {tm['full_rows']:,} rows: about {human_time(tm['search30_full'])} for 30 trials × 5 folds on this machine."],
                  "refs": []})
    # 7. metric + threshold
    mt = [f"Optimise {mlabel(m)} — {GOALS[P.task][P.goal][0].lower()}."]
    th = ctx.R.get("threshold")
    if th:
        b = th["best_row"]; d = th["default_row"]
        mt.append(f"Do not use the default 0.5 cut-off. On out-of-fold predictions, threshold {b[0]:.2f} gives precision {pct(b[1])} / recall {pct(b[2])} "
                  f"(vs {pct(d[1])} / {pct(d[2])} at 0.5). Re-tune it on validation data, never on the test set.")
    if P.task == "classification":
        mt.append("Also report: confusion matrix, per-class recall, and calibration (reliability curve) if probabilities drive decisions.")
    else:
        mt.append("Also report: MAE and RMSE together (their ratio tells you how heavy the error tail is), and a residual plot vs prediction.")
    steps.append({"title": "Measure what matters", "detail": mt, "refs": []})
    # 8. after training
    post = ["Check performance per subgroup (see Coverage section) before sign-off."]
    if ctx.R["validity"].get("drift_features"):
        post.append("Monitor drift in: " + ", ".join(list(ctx.R["validity"]["drift_features"])[:4]) + ".")
    if ctx.t is not None:
        post.append("Plan a retraining cadence; the target / features move over time.")
    post.append("Keep a model card: data period, excluded columns and why, metric on the untouched test set.")
    steps.append({"title": "Before shipping", "detail": post, "refs": [f.id for f in ctx.findings if f.dimension == "fairness"]})
    ctx.R["plan"] = steps
    ctx.R["code"] = code_snippet(ctx, rec)


# ============================================================================= narrative
def narrative(ctx: Ctx):
    P, R = ctx.P, ctx.R
    fit = R["fitness"]
    m = P.metric
    rec = ctx.choice["recommended"]
    r = ctx.bake[rec]
    sg = R["signal"]
    goal_txt = GOALS[P.task][P.goal][0].lower()
    use = P.uc.title or f"predict {P.uc.target}"
    paras = []
    paras.append(f"You want to <b>{use}</b>: a {P.task} problem on '{P.uc.target}'"
                 + (f" with '{P.uc.positive_class}' as the class of interest" if P.pos is not None else "")
                 + f". Priority: {goal_txt}. The dataset has {P.n_raw:,} rows; {len(P.features)} columns are usable as features after "
                   f"excluding {len(P.excluded)}.")
    crit = sorted([f for f in ctx.findings if f.severity in ("critical", "major")], key=lambda f: SEV_ORDER[f.severity])
    sig_word = ("no usable signal" if sg["skill"] < .03 else "a weak signal" if sg["skill"] < .15 else
                "a moderate signal" if sg["skill"] < .4 else "a strong signal")
    paras.append(f"<b>Verdict: {fit['verdict']} ({fit['overall']}/100).</b> The features carry {sig_word} about the target "
                 f"(real labels {fmt(sg['real'], 3)} vs {fmt(float(np.mean(sg['null'])), 3)} with shuffled labels). "
                 + (f"The score is capped because of {' and '.join(fit['caps'])}. " if fit["caps"] else ""))
    if crit:
        items = "".join(f"<li><b>{f.title}.</b> {f.impact.split('. ')[0].rstrip('.')}.</li>" for f in crit[:4])
        paras.append(f"The main problems, in order of severity:<ul>{items}</ul>")
    else:
        paras.append("No critical or major problems were found; the remaining notes are refinements.")
    split_txt = {"time": "training on the past and testing on later periods", "group": f"testing on unseen {P.uc.entity_column} values",
                 "random": "5-fold cross-validation"}[ctx.split_kind]
    base = R["bakeoff"]["baseline"][m]
    if r["skill"] <= .02:
        paras.append(f"<b>What to expect.</b> Using {split_txt}, none of the {len(ctx.bake)} candidate models beats a no-skill baseline on {mlabel(m)} "
                     f"(best: {rec} at {fmt_metric(m, r['mean'][m])} vs {fmt_metric(m, base)} for the baseline). Training a bigger model, more epochs or "
                     f"fine-tuning would only memorise noise. The next step is new information, not a new algorithm.")
    else:
        paras.append(f"<b>What to expect.</b> Using {split_txt}, the recommended model ({rec}) reaches {mlabel(m)} "
                     f"<b>{fmt_metric(m, r['mean'][m])} ± {fmt_metric(m, r['std'][m])}</b>, against {fmt_metric(m, base)} for a no-skill baseline "
                     f"({pct(max(0, r['skill']))} of the way from baseline to perfect). " + _size_sentence(R))
    if R.get("leak_removed"):
        one = len(R["leak_removed"]) == 1
        paras.append(f"Note: {', '.join(R['leak_removed'])} {'was' if one else 'were'} removed as suspected leakage before modelling; "
                     f"with {'it' if one else 'them'} the scores would look far better and mean nothing.")
    R["narrative"] = paras
    # use-case fit lines
    fitlines = []
    best = ctx.choice["best_score"]
    inter = [n for n in ctx.choice["ranking"] if ctx.bake[n]["interp"] == 3]
    if inter:
        gap = ctx.bake[best]["mean"][m] - ctx.bake[inter[0]]["mean"][m]
        fitlines.append({"label": "Interpretability", "value": f"best readable model: {inter[0]} ({fmt_metric(m, ctx.bake[inter[0]]['mean'][m])})",
                         "note": f"costs {fmt_metric(m, abs(gap))} {mlabel(m)} vs {best}" if inter[0] != best else "the best model is also readable"})
    fitlines.append({"label": "Deployment", "value": f"{rec}: {r['size_kb']:.0f} KB, {r['latency_ms']:.3f} ms / prediction",
                     "note": {"edge": "edge target: aim for < 2 MB and < 1 ms", "batch": "batch scoring: latency is not a concern",
                              "server": "fine for an API"}[P.uc.deployment]})
    th = R.get("threshold")
    if th:
        b = th["best_row"]
        fitlines.append({"label": "Operating point", "value": f"threshold {b[0]:.2f} → precision {pct(b[1])}, recall {pct(b[2])}",
                         "note": f"flags {b[4]} of {th['n']} rows ({th['positives']} true positives exist)"})
    R["usecase_fit"] = fitlines


def _size_sentence(R):
    ds = R.get("data_size", {})
    ex = ds.get("extrapolation")
    if not ex:
        return ""
    a, b = ex["score"][0], ex["score"][1]
    m = ds["metric"]
    change = abs(b - a) / (abs(a) + 1e-9)
    if change < .02:
        return "The learning curve has flattened: more rows of the same kind will barely help."
    return (f"The learning curve is still improving: doubling the training rows is projected to improve {mlabel(m)} by about {pct(change)} "
            f"(measured with random CV, so compare the relative change, not the absolute value).")


# ============================================================================= code
SK_SCORING = {"balanced_accuracy": "'balanced_accuracy'", "roc_auc": "'roc_auc'", "accuracy": "'accuracy'",
              "mae": "'neg_mean_absolute_error'", "rmse": "'neg_root_mean_squared_error'", "mape": "'neg_mean_absolute_percentage_error'"}

MODEL_CODE = {
    ("classification", "Logistic regression"): ("from sklearn.linear_model import LogisticRegression", "LogisticRegression(max_iter=3000{cw})"),
    ("classification", "Decision tree"): ("from sklearn.tree import DecisionTreeClassifier", "DecisionTreeClassifier(max_depth=5, min_samples_leaf=10{cw})"),
    ("classification", "k-nearest neighbours"): ("from sklearn.neighbors import KNeighborsClassifier", "KNeighborsClassifier(15)"),
    ("classification", "Random forest"): ("from sklearn.ensemble import RandomForestClassifier", "RandomForestClassifier(300, min_samples_leaf=2, n_jobs=-1{cw})"),
    ("classification", "Gradient boosting"): ("from sklearn.ensemble import HistGradientBoostingClassifier", "HistGradientBoostingClassifier(learning_rate=0.08, early_stopping=True{cw})"),
    ("classification", "Neural network (MLP)"): ("from sklearn.neural_network import MLPClassifier", "MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True, max_iter=300)"),
    ("regression", "Ridge regression"): ("from sklearn.linear_model import Ridge", "Ridge(1.0)"),
    ("regression", "Decision tree"): ("from sklearn.tree import DecisionTreeRegressor", "DecisionTreeRegressor(max_depth=6, min_samples_leaf=10)"),
    ("regression", "k-nearest neighbours"): ("from sklearn.neighbors import KNeighborsRegressor", "KNeighborsRegressor(15)"),
    ("regression", "Random forest"): ("from sklearn.ensemble import RandomForestRegressor", "RandomForestRegressor(300, min_samples_leaf=2, n_jobs=-1)"),
    ("regression", "Gradient boosting"): ("from sklearn.ensemble import HistGradientBoostingRegressor", "HistGradientBoostingRegressor(learning_rate=0.08, early_stopping=True)"),
    ("regression", "Gradient boosting (log target)"): ("from sklearn.ensemble import HistGradientBoostingRegressor\nfrom sklearn.compose import TransformedTargetRegressor",
                                                       "TransformedTargetRegressor(HistGradientBoostingRegressor(learning_rate=0.08, early_stopping=True), func=np.log1p, inverse_func=np.expm1)"),
    ("regression", "Neural network (MLP)"): ("from sklearn.neural_network import MLPRegressor\nfrom sklearn.compose import TransformedTargetRegressor",
                                             "TransformedTargetRegressor(MLPRegressor(hidden_layer_sizes=(64, 32), early_stopping=True, max_iter=400), transformer=StandardScaler())"),
}


def code_snippet(ctx: Ctx, rec):
    P = ctx.P
    prof = ctx.prof
    num = [c for c in P.features if prof[c]["kind"] in ("numeric",)]
    cat = [c for c in P.features if prof[c]["kind"] in ("categorical", "boolean")]
    dates = [c for c in P.features if prof[c]["kind"] == "datetime"]
    imp, ctor = MODEL_CODE[(P.task, rec)]
    cw = ", class_weight='balanced'" if ctx.R["bakeoff"].get("balanced_weights") else ""
    ctor = ctor.format(cw=cw)
    scale = ctx.bake[rec]["family"] in ("linear", "instance", "neural")
    m = P.metric
    if m in SK_SCORING:
        scoring = SK_SCORING[m]
        sc_imp = ""
    else:
        beta = 2 if m == "f2" else .5
        pos = f", pos_label={P.pos}" if P.pos is not None else ", average='macro'"
        scoring = f"make_scorer(fbeta_score, beta={beta}{pos})"
        sc_imp = "from sklearn.metrics import make_scorer, fbeta_score\n"
    split = {"time": "TimeSeriesSplit(n_splits=5)", "group": "GroupKFold(n_splits=5)",
             "random": "StratifiedKFold(5, shuffle=True, random_state=42)" if P.task == "classification" else "KFold(5, shuffle=True, random_state=42)"}[ctx.split_kind]
    split_imp = split.split("(")[0]
    L = [
        "# Generated by DataDoctor — a starting point, not a finished model.",
        "import numpy as np, pandas as pd",
        "from sklearn.compose import ColumnTransformer",
        "from sklearn.impute import SimpleImputer",
        "from sklearn.pipeline import make_pipeline",
        "from sklearn.preprocessing import OneHotEncoder" + (", StandardScaler" if scale else ""),
        f"from sklearn.model_selection import cross_validate, {split_imp}",
        sc_imp + imp,
        "",
        f"df = pd.read_csv('your_data.csv')",
    ]
    if ctx.R.get("variants"):
        L.append("# normalise inconsistent category spellings (e.g. 'paypal' -> 'PayPal')")
        L.append(f"for c in {list(ctx.R['variants'])[:8]!r}:")
        L.append("    key = df[c].astype(str).str.lower().str.replace(r'[^0-9a-z]', '', regex=True)")
        L.append("    canon = df.groupby(key)[c].agg(lambda s: s.value_counts().index[0])")
        L.append("    df[c] = key.map(canon).where(df[c].notna())")
    L.append("df = df.drop_duplicates()")
    L.append(f"df = df[df[{P.uc.target!r}].notna()]")
    if dates:
        tc = P.uc.time_column if P.uc.time_column in dates else dates[0]
        L.append(f"for c in {dates!r}:")
        L.append(f"    d = pd.to_datetime(df[c], dayfirst={bool(prof[dates[0]].get('dayfirst'))}, errors='coerce')")
        L.append("    df[c + '_month'], df[c + '_weekday'] = d.dt.month, d.dt.dayofweek")
        num += [f"{c}_month" for c in dates] + [f"{c}_weekday" for c in dates]
        if ctx.split_kind == "time":
            L.append(f"df = df.sort_values({tc!r}, key=lambda s: pd.to_datetime(s, dayfirst={bool(prof[tc].get('dayfirst'))}, errors='coerce'))")
    if P.task == "classification" and P.pos is not None:
        L.append(f"y = (df[{P.uc.target!r}].astype(str) == {P.classes[P.pos]!r}).astype(int)   # 1 = {P.classes[P.pos]}")
    else:
        L.append(f"y = df[{P.uc.target!r}]")
    L.append(f"X = df[{num + cat!r}]")
    if ctx.split_kind == "group":
        L.append(f"groups = df[{P.uc.entity_column!r}]")
    L += ["",
          "pre = ColumnTransformer([",
          f"    ('num', make_pipeline(SimpleImputer(strategy='median', add_indicator=True){', StandardScaler()' if scale else ''}), {num!r}),",
          f"    ('cat', make_pipeline(SimpleImputer(strategy='constant', fill_value='missing'), OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=0.01{', sparse_output=False' if 'boosting' in ctx.bake[rec]['family'] else ''})), {cat!r}),",
          "])",
          f"model = make_pipeline(pre, {ctor})",
          "",
          f"cv = cross_validate(model, X, y, cv={split}, scoring={scoring}{', groups=groups' if ctx.split_kind == 'group' else ''}, return_train_score=True)",
          "print('validation:', cv['test_score'].mean().round(4), '+/-', cv['test_score'].std().round(4))",
          "print('train     :', cv['train_score'].mean().round(4))   # a big gap = overfitting",
          "model.fit(X, y)"]
    return "\n".join(L)
