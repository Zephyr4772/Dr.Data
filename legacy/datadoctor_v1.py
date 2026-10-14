"""
DataDoctor - one-file dataset audit that produces a PDF report.

Usage:
    python datadoctor.py data.csv --target label            # audit your own CSV
    python datadoctor.py --demo dirty                       # sabotaged demo dataset
    python datadoctor.py --demo clean                       # clean demo dataset
    python datadoctor.py --demo noisy                       # demo with flipped labels

Needs: numpy pandas scikit-learn matplotlib reportlab
Scope: tabular classification, small datasets (up to a few tens of thousands of rows).
"""
import argparse, datetime, os, tempfile, time, warnings
from dataclasses import dataclass
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import (StratifiedKFold, cross_val_predict, cross_val_score,
                                     learning_curve, train_test_split, validation_curve)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

warnings.filterwarnings("ignore")

NAVY, BLUE, RED, AMBER, GREEN, GREY = "#1f2a44", "#3b6fd4", "#d64545", "#e6a23c", "#3aa66b", "#8a93a6"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlesize": 10, "figure.dpi": 150})
SEED = 42
MAX_ROWS = 6000   # model-based checks run on a stratified sample of at most this many rows
BAL = "balanced_accuracy"


# ----------------------------------------------------------------------------- data prep
@dataclass
class Finding:
    sev: str      # FAIL / WARN / INFO
    area: str
    title: str
    fix: str = ""


def prepare(df, target):
    df = df.dropna(subset=[target]).reset_index(drop=True)
    y, classes = pd.factorize(df[target])
    X_raw = df.drop(columns=[target])
    X = X_raw.copy()
    for c in list(X.columns):
        s = X[c]
        if pd.api.types.is_bool_dtype(s):
            X[c] = s.astype(int)
        elif not pd.api.types.is_numeric_dtype(s):          # text / category / dates
            s = s.fillna("missing").astype(str)
            if s.nunique() > .8 * len(s) and len(s) > 20:     # ID-like text: useless as a feature
                X = X.drop(columns=c)
                continue
            top = s.value_counts().index[:15]                  # cap one-hot levels
            X[c] = s.where(s.isin(top), "other")
    X = pd.get_dummies(X, dtype=float)
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    X = X.loc[:, X.nunique() > 1]
    return df, X_raw, X, y, classes


def savefig(fig, tmp, name):
    path = os.path.join(tmp, name + ".png")
    fig.savefig(path, bbox_inches="tight", dpi=170)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------- audit engine
class Audit:
    def __init__(self, df, target, name="dataset", max_rows=MAX_ROWS):
        self.name, self.target = name, target
        df = df.dropna(subset=[target]).reset_index(drop=True)
        self.n_full = len(df)
        self.full_dup = df.duplicated().mean()
        if len(df) > max_rows:
            try:
                _, df = train_test_split(df, test_size=max_rows, stratify=df[target], random_state=SEED)
            except ValueError:
                df = df.sample(max_rows, random_state=SEED)
            df = df.reset_index(drop=True)
        self.df, self.X_raw, self.X, self.y, self.classes = prepare(df, target)
        self.n, self.p = self.X.shape
        self.k = len(self.classes)
        self.counts = np.bincount(self.y)
        if self.k < 2 or self.counts.min() < 2:
            raise SystemExit("Need at least 2 classes with >=2 rows each (classification only).")
        self.cv = StratifiedKFold(n_splits=max(2, min(5, int(self.counts.min()))), shuffle=True, random_state=SEED)
        self.baseline = 1.0 / self.k  # balanced accuracy of a constant predictor
        self.findings, self.figs, self.stats = [], {}, {}
        self.tmp = tempfile.mkdtemp()
        self.leak_cols = []
        self.Xm = self.X

    def add(self, sev, area, title, fix=""):
        self.findings.append(Finding(sev, area, title, fix))

    # -- 1. structure / hygiene --
    def sanity(self):
        raw, df = self.X_raw, self.df
        miss = raw.isna().mean()
        for c in miss[miss > .05].index:
            self.add("FAIL" if miss[c] > .30 else "WARN", "Missing data",
                     f"'{c}' has {miss[c]:.0%} missing values",
                     "Impute inside a pipeline (fit on train only) or drop the column if >50% empty.")
        dup = self.full_dup
        self.stats["dup"] = dup
        if dup > .01:
            self.add("FAIL" if dup > .10 else "WARN", "Duplicates", f"{dup:.1%} of rows are exact duplicates",
                     "Drop duplicates BEFORE splitting, otherwise copies land in both train and test and inflate scores.")
        feat_dup = raw.duplicated(keep=False)
        if feat_dup.any():
            g = df[feat_dup].groupby(list(raw.columns), dropna=False)[self.target].nunique()
            conflicts = int((g > 1).sum())
            self.stats["conflicts"] = conflicts
            if conflicts:
                self.add("FAIL", "Duplicates", f"{conflicts} feature-identical rows carry conflicting labels",
                         "Same input, different label: the model cannot fit both. Inspect and resolve them.")
        for c in raw.columns[raw.nunique() <= 1]:
            self.add("WARN", "Structure", f"'{c}' is constant", "Zero information. Drop it.")
        for c in raw.columns:
            if raw[c].nunique() > .8 * len(raw) and raw[c].dtype != float and len(raw) > 20:
                self.add("WARN", "Shortcut risk", f"'{c}' looks like an ID / row index ({raw[c].nunique()} unique values)",
                         "Identifiers let flexible models memorise rows. Drop it.")
                self.leak_cols.append(c)
        for c in raw.columns:
            if any(w in c.lower() for w in ("final", "outcome", "result", "target", "label")) and c != self.target:
                self.add("WARN", "Shortcut risk", f"'{c}' has a leak-prone name",
                         "Check it is available at prediction time. If it is produced after the label, remove it.")
        ratio = self.counts.max() / self.counts.min()
        self.stats["imbalance"] = ratio
        if ratio > 3:
            self.add("FAIL" if ratio > 10 else "WARN", "Class balance", f"Class imbalance {ratio:.1f}:1",
                     "Use stratified splits, class weights or resampling, and report F1 / PR-AUC / balanced accuracy.")
        rpf = self.n / max(self.p, 1)
        self.stats["rpf"] = rpf
        if rpf < 5:
            self.add("FAIL", "Sample size", f"Only {rpf:.1f} rows per feature ({self.n} rows, {self.p} features)",
                     "High overfitting risk. Use fewer features (PCA / selection), strong regularisation, simple models.")
        elif rpf < 10:
            self.add("WARN", "Sample size", f"Low rows-per-feature ratio ({rpf:.1f})", "Prefer regularised / simple models.")
        if self.counts.min() < 30:
            self.add("WARN", "Sample size", f"Smallest class has only {self.counts.min()} rows",
                     "Metrics for that class will be very noisy. Use repeated stratified CV.")
        num = self.X_raw.select_dtypes("number")
        if num.shape[1]:
            q1, q3 = num.quantile(.25), num.quantile(.75)
            iqr = (q3 - q1).replace(0, np.nan)
            out = (((num < q1 - 3 * iqr) | (num > q3 + 3 * iqr)).mean()).fillna(0)
            self.stats["outliers"] = out
            for c in out[out > .05].index:
                self.add("WARN", "Outliers", f"'{c}' has {out[c]:.0%} extreme outliers (beyond 3 x IQR)",
                         "Check for entry errors; consider clipping / log transform / robust scaling.")
            sk = num.skew().abs()
            self.stats["skew"] = sk
            hs = sk[sk > 2].index.tolist()
            if hs:
                self.add("INFO", "Distributions", f"{len(hs)} heavily skewed feature(s): {', '.join(map(str, hs[:5]))}",
                         "Log / power transform helps linear models and MLPs (trees do not care).")
        c = self.X.corr().abs()
        pairs = [(a, b, c.loc[a, b]) for i, a in enumerate(c.columns) for b in c.columns[i + 1:] if c.loc[a, b] > .95]
        self.stats["corr_pairs"] = pairs
        if pairs:
            a, b, v = max(pairs, key=lambda t: t[2])
            self.add("WARN", "Redundancy", f"{len(pairs)} near-duplicate feature pair(s), e.g. '{a}' & '{b}' (r={v:.2f})",
                     "Drop one of each pair; unstable coefficients and misleading importances otherwise.")

    # -- 2. shortcut / leakage --
    def leakage(self):
        cols = self.X.var().sort_values(ascending=False).index[:60]
        scores = {}
        for c in cols:
            scores[c] = cross_val_score(DecisionTreeClassifier(max_depth=3, random_state=SEED),
                                        self.X[[c]], self.y, cv=self.cv, scoring=BAL).mean()
        self.stats["single"] = pd.Series(scores).sort_values(ascending=False)
        for c, s in self.stats["single"].items():
            if s > .95:
                self.leak_cols.append(c)
                self.add("FAIL", "Shortcut risk", f"'{c}' alone predicts the label ({s:.2f} balanced accuracy)",
                         "Almost certainly leakage (derived from the label or known only after the fact). Remove it and re-audit.")
        Xtr, Xva, ytr, yva = train_test_split(self.X, self.y, test_size=.3, stratify=self.y, random_state=SEED)
        rf = RandomForestClassifier(200, random_state=SEED, n_jobs=-1).fit(Xtr, ytr)
        pi = permutation_importance(rf, Xva, yva, n_repeats=5, random_state=SEED, scoring=BAL)
        imp = pd.Series(pi.importances_mean, index=self.X.columns).clip(lower=0).sort_values(ascending=False)
        self.stats["imp"] = imp
        share = imp.iloc[0] / imp.sum() if imp.sum() > 0 else 0
        self.stats["top_share"] = share
        if share > .60 and len(imp) > 2:
            self.add("WARN", "Shortcut risk", f"One feature ('{imp.index[0]}') drives {share:.0%} of the model's skill",
                     "The model leans on a single cue. Verify it is legitimate and will exist (with the same meaning) in production.")
        removed = [c for c in dict.fromkeys(self.leak_cols) if c in self.X.columns]
        if removed and len(removed) < self.X.shape[1]:
            self.Xm = self.X.drop(columns=removed)
        self.stats["removed"] = removed if self.Xm is not self.X else []

    # -- 3. signal --
    def signal(self):
        rng = np.random.default_rng(SEED)
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        real = cross_val_score(model, self.Xm, self.y, cv=self.cv, scoring=BAL).mean()
        shuf = [cross_val_score(model, self.Xm, rng.permutation(self.y), cv=self.cv, scoring=BAL).mean() for _ in range(15)]
        self.stats.update(real=real, shuf=np.array(shuf))
        thr = np.mean(shuf) + 2 * np.std(shuf)
        self.stats["no_signal"] = real < thr
        if real < thr:
            self.add("FAIL", "Signal", f"Real-label score ({real:.2f}) is not clearly above shuffled-label noise ({np.mean(shuf):.2f})",
                     "Features carry no learnable information about the target. Training will not help; revisit features or the target.")
        elif real < self.baseline + .15:
            self.add("WARN", "Signal", f"Weak signal: balanced accuracy {real:.2f} vs {self.baseline:.2f} for guessing",
                     "Expect a low ceiling. Feature engineering will matter more than model choice.")

    # -- 4. bias / variance --
    def bias_variance(self):
        sizes = np.linspace(.25, 1.0, 5)
        res = {}
        models = {"Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
                  "Random forest": RandomForestClassifier(100, random_state=SEED, n_jobs=-1)}
        for nm, m in models.items():
            s, tr, va = learning_curve(m, self.Xm, self.y, cv=self.cv, train_sizes=sizes, scoring=BAL, n_jobs=-1,
                                       shuffle=True, random_state=SEED)
            res[nm] = (s, tr, va)
        self.stats["lc"] = res
        for nm, (s, tr, va) in res.items():
            gap = tr.mean(1)[-1] - va.mean(1)[-1]
            slope = va.mean(1)[-1] - va.mean(1)[-2]
            self.stats[f"gap_{nm}"] = gap
            if gap > .10 and tr.mean(1)[-1] > .9:
                self.add("WARN", "Overfitting", f"{nm}: train {tr.mean(1)[-1]:.2f} vs validation {va.mean(1)[-1]:.2f} (gap {gap:.2f})",
                         "High variance. Add regularisation, simplify the model, get more data or reduce features.")
            if va.mean(1)[-1] < self.baseline + .15 and tr.mean(1)[-1] < self.baseline + .25:
                self.add("WARN", "Underfitting", f"{nm}: both train ({tr.mean(1)[-1]:.2f}) and validation scores are low",
                         "High bias. Try a more flexible model or better features.")
            if slope > .01 and nm == "Random forest":
                self.add("INFO", "Data size", "Validation curve is still rising at 100% of the data",
                         "More labelled data would likely improve results.")
        depths = np.arange(1, 13)
        tr, va = validation_curve(DecisionTreeClassifier(random_state=SEED), self.Xm, self.y, param_name="max_depth",
                                  param_range=depths, cv=self.cv, scoring=BAL, n_jobs=-1)
        self.stats["depth"] = (depths, tr.mean(1), va.mean(1))
        self.stats["best_depth"] = int(depths[np.argmax(va.mean(1))])

    # -- 5. robustness --
    def robustness(self):
        Xtr, Xva, ytr, yva = train_test_split(self.Xm, self.y, test_size=.3, stratify=self.y, random_state=SEED)
        sc = StandardScaler().fit(Xtr)
        A, B = sc.transform(Xtr), sc.transform(Xva)
        rng = np.random.default_rng(SEED)
        sig = [0, .1, .25, .5, 1.0]
        out = {}
        for nm, m in {"Logistic regression": LogisticRegression(max_iter=2000),
                      "Random forest": RandomForestClassifier(100, random_state=SEED, n_jobs=-1)}.items():
            m.fit(A, ytr)
            out[nm] = [np.mean([(lambda p: __import__("sklearn.metrics", fromlist=["x"]).balanced_accuracy_score(yva, p))(
                m.predict(B + rng.normal(0, s, B.shape))) for _ in range(3)]) for s in sig]
        self.stats["noise"] = (sig, out)
        for nm, v in out.items():
            if v[0] - v[3] > .15:
                self.add("WARN", "Robustness", f"{nm} loses {v[0] - v[3]:.2f} balanced accuracy under mild feature noise (0.5 std)",
                         "Predictions are fragile to measurement noise. Consider smoothing, binning or regularisation.")
        # label noise (simplified confident learning, leakage columns excluded)
        Xc = self.Xm.drop(columns=[c for c in self.leak_cols if c in self.Xm.columns])
        if Xc.shape[1] == 0:
            Xc = self.Xm
        P = cross_val_predict(RandomForestClassifier(200, random_state=SEED, n_jobs=-1, min_samples_leaf=2),
                              Xc, self.y, cv=self.cv, method="predict_proba")
        t = np.array([P[self.y == j, j].mean() for j in range(self.k)])
        own = P[np.arange(self.n), self.y]
        other = P.copy(); other[np.arange(self.n), self.y] = -1
        sus = (own < t[self.y]) & (other.max(1) >= t[other.argmax(1)])
        self.stats.update(own_conf=own, suspects=np.where(sus)[0], noise_rate=sus.mean())
        if sus.mean() > .05:
            self.add("FAIL" if sus.mean() > .15 else "WARN", "Label noise",
                     f"~{sus.mean():.0%} of labels look wrong ({int(sus.sum())} rows confidently contradicted by the model)",
                     "Review the listed suspect rows (see appendix). Noisy labels cap accuracy and make models memorise errors.")

    # -- 6. training budget --
    def budget(self):
        Xtr, Xva, ytr, yva = train_test_split(self.Xm, self.y, test_size=.25, stratify=self.y, random_state=SEED)
        sc = StandardScaler().fit(Xtr)
        Xtr, Xva = sc.transform(Xtr), sc.transform(Xva)
        bs = int(min(64, max(8, len(Xtr) // 8)))
        mlp = MLPClassifier((64, 64), batch_size=bs, random_state=SEED)
        cl = np.unique(self.y)
        tr_l, va_l, best, best_ep, t0 = [], [], 1e9, 1, time.time()
        for ep in range(1, 301):
            mlp.partial_fit(Xtr, ytr, classes=cl)
            tr_l.append(log_loss(ytr, mlp.predict_proba(Xtr), labels=cl))
            v = log_loss(yva, mlp.predict_proba(Xva), labels=cl)
            va_l.append(v)
            if v < best:
                best, best_ep = v, ep
            if ep - best_ep >= 20:
                break
        per_ep = (time.time() - t0) / ep
        rec = int(np.ceil(best_ep * 1.2))
        self.stats.update(mlp=(tr_l, va_l, best_ep), rec_epochs=rec, per_epoch=per_ep, batch=bs,
                          rec_time=rec * per_ep, mlp_steps=rec * int(np.ceil(len(Xtr) / bs)))
        if best_ep >= 290:
            self.add("INFO", "Training", "Validation loss was still improving after 300 epochs",
                     "Increase max epochs or the learning rate, and standardise the inputs.")
        if best_ep <= 3:
            self.add("INFO", "Training", f"Validation loss bottoms out after only {best_ep} epoch(s)",
                     "Easy or tiny problem, or an overly large learning rate. Use early stopping and a lower LR.")

    # -- scoring / risks --
    def finalize(self):
        score = 100
        cap = {}
        for f in self.findings:
            d = 15 if f.sev == "FAIL" else 5 if f.sev == "WARN" else 0
            if f.area == "Shortcut risk" and f.sev == "FAIL":
                d = 25
            if f.area == "Signal" and f.sev == "FAIL":
                d = 30
            cap[f.area] = cap.get(f.area, 0) + d
        score -= sum(min(v, 30) for v in cap.values())
        self.score = int(max(0, min(100, score)))
        areas = lambda *a: [f for f in self.findings if f.area in a and f.sev != "INFO"]
        s = self.stats
        # overfit
        pts, why = 0, []
        if s["rpf"] < 5: pts += 2; why.append("very few rows per feature")
        elif s["rpf"] < 10: pts += 1; why.append("low rows per feature")
        g = max(s["gap_Logistic regression"], s["gap_Random forest"])
        if g > .2: pts += 2; why.append(f"train/validation gap {g:.2f}")
        elif g > .1: pts += 1; why.append(f"train/validation gap {g:.2f}")
        if s["dup"] > .01: pts += 1; why.append("duplicate rows leak between folds")
        if s["noise_rate"] > .05: pts += 1; why.append("noisy labels get memorised")
        if self.leak_cols: pts += 1; why.append("leak columns inflate scores")
        over = (pts, why)
        # underfit
        pts, why = 0, []
        if s["no_signal"]: pts += 3; why.append("no signal over shuffled labels")
        elif s["real"] < self.baseline + .15: pts += 2; why.append("weak signal")
        elif s["real"] < self.baseline + .30: pts += 1; why.append("moderate signal")
        if areas("Underfitting"): pts += 1; why.append("train and validation both low")
        under = (pts, why)
        # shortcut
        pts, why = 0, []
        if any(f.sev == "FAIL" and f.area == "Shortcut risk" for f in self.findings): pts += 3; why.append("single-feature leak found")
        if s["top_share"] > .6: pts += 2; why.append("one feature dominates")
        if any("ID" in f.title for f in self.findings): pts += 1; why.append("ID-like column present")
        if any("leak-prone" in f.title for f in self.findings): pts += 1; why.append("leak-prone column name")
        short = (pts, why)
        # data quality
        n_issue = len(areas("Missing data", "Duplicates", "Structure", "Outliers", "Redundancy", "Class balance", "Label noise"))
        qual = (0 if n_issue == 0 else 2 if n_issue <= 2 else 3, [f"{n_issue} hygiene issue(s) flagged"] if n_issue else ["no hygiene issues flagged"])
        lvl = lambda p: "LOW" if p <= 1 else "MEDIUM" if p == 2 else "HIGH"
        self.risks = [("Overfitting", lvl(over[0]), over[1]), ("Underfitting", lvl(under[0]), under[1]),
                      ("Shortcut learning", lvl(short[0]), short[1]), ("Data quality", lvl(qual[0]), qual[1])]
        self.verdict = ("Ready to train" if self.score >= 80 else
                        "Train with caution: fix the flagged issues first" if self.score >= 55 else
                        "Not ready: fix the critical issues before training")

    def run(self):
        for step in (self.sanity, self.leakage, self.signal, self.bias_variance, self.robustness, self.budget, self.finalize):
            step()
        return self

    # ------------------------------------------------------------------------- figures
    def make_figures(self):
        s, tmp, F = self.stats, self.tmp, self.figs
        # gauge
        fig, ax = plt.subplots(figsize=(3, 1.8))
        col = GREEN if self.score >= 80 else AMBER if self.score >= 55 else RED
        ax.add_patch(Wedge((0, 0), 1, 0, 180, width=.26, color="#e5e8ef"))
        ax.add_patch(Wedge((0, 0), 1, 180 - 1.8 * self.score, 180, width=.26, color=col))
        ax.text(0, .22, str(self.score), ha="center", fontsize=30, fontweight="bold", color=NAVY)
        ax.text(0, -.02, "/ 100 health score", ha="center", fontsize=8, color=GREY)
        ax.set_xlim(-1.1, 1.1); ax.set_ylim(-.12, 1.1); ax.set_aspect("equal"); ax.axis("off")
        F["gauge"] = savefig(fig, tmp, "gauge")
        # overview: class balance + missing
        fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.6))
        names = [str(c) for c in self.classes]
        a.bar(names, self.counts, color=BLUE)
        for i, v in enumerate(self.counts): a.text(i, v, str(v), ha="center", va="bottom", fontsize=8)
        a.set_title("Class balance"); a.set_ylabel("rows")
        m = self.X_raw.isna().mean().sort_values(ascending=False).head(8) * 100
        if m.sum() > 0:
            b.barh(m.index[::-1].astype(str), m.values[::-1], color=AMBER)
            b.set_title("Missing values (%)")
        else:
            b.text(.5, .5, "No missing values", ha="center", va="center", fontsize=11, color=GREEN); b.axis("off")
            b.set_title("Missing values")
        fig.tight_layout(); F["overview"] = savefig(fig, tmp, "overview")
        # distributions
        num = self.X_raw.select_dtypes("number")
        if num.shape[1]:
            cols = list(num.var().sort_values(ascending=False).index[:9])
            fig, axs = plt.subplots(3, 3, figsize=(7.2, 5.2))
            for ax, c in zip(axs.ravel(), cols):
                for j, cl in enumerate(self.classes):
                    ax.hist(num.loc[self.df[self.target] == cl, c].dropna(), bins=20, alpha=.55, label=str(cl))
                ax.set_title(str(c)[:22], fontsize=8); ax.tick_params(labelsize=6)
            for ax in axs.ravel()[len(cols):]: ax.axis("off")
            axs.ravel()[0].legend(fontsize=6)
            fig.suptitle("Feature distributions by class (highest-variance features)", fontweight="bold", fontsize=10)
            fig.tight_layout(); F["dist"] = savefig(fig, tmp, "dist")
        # correlation
        cols = list(self.X.var().sort_values(ascending=False).index[:18])
        cm_ = self.X[cols].corr()
        fig, ax = plt.subplots(figsize=(5.2, 4.4))
        im = ax.imshow(cm_, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(cols))); ax.set_yticks(range(len(cols)))
        ax.set_xticklabels([str(c)[:14] for c in cols], rotation=90, fontsize=6)
        ax.set_yticklabels([str(c)[:14] for c in cols], fontsize=6)
        fig.colorbar(im, fraction=.04); ax.set_title("Feature correlation"); ax.spines[:].set_visible(False)
        F["corr"] = savefig(fig, tmp, "corr")
        # leakage + importance
        fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.2))
        sg = s["single"].head(10)[::-1]
        a.barh(sg.index.astype(str), sg.values, color=[RED if v > .95 else BLUE for v in sg.values])
        a.axvline(.95, color=RED, ls="--", lw=1); a.axvline(self.baseline, color=GREY, ls=":", lw=1)
        a.set_xlim(0, 1.02); a.set_title("Single-feature predictive power"); a.set_xlabel("balanced accuracy (depth-3 tree, CV)")
        a.tick_params(axis="y", labelsize=7)
        ip = s["imp"].head(10)[::-1]
        b.barh(ip.index.astype(str), ip.values, color=BLUE)
        b.set_title("Permutation importance (RF)"); b.set_xlabel("drop in balanced accuracy"); b.tick_params(axis="y", labelsize=7)
        fig.tight_layout(); F["leak"] = savefig(fig, tmp, "leak")
        # signal
        fig, ax = plt.subplots(figsize=(5.6, 2.4))
        ax.hist(s["shuf"], bins=10, color=GREY, alpha=.8, label="shuffled labels (15 runs)")
        ax.axvline(s["real"], color=GREEN if not s["no_signal"] else RED, lw=2.5, label=f"real labels: {s['real']:.2f}")
        ax.set_xlabel("CV balanced accuracy"); ax.set_title("Is there a signal beyond chance?"); ax.legend(fontsize=7)
        F["signal"] = savefig(fig, tmp, "signal")
        # learning curves
        fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.9), sharey=True)
        for ax, (nm, (sz, tr, va)) in zip(axs, s["lc"].items()):
            for arr, lab, c in ((tr, "train", RED), (va, "validation", BLUE)):
                ax.plot(sz, arr.mean(1), "o-", color=c, label=lab)
                ax.fill_between(sz, arr.mean(1) - arr.std(1), arr.mean(1) + arr.std(1), color=c, alpha=.15)
            ax.axhline(self.baseline, color=GREY, ls=":", lw=1)
            ax.set_title(nm); ax.set_xlabel("training rows")
        axs[0].set_ylabel("balanced accuracy"); axs[0].legend(fontsize=7)
        fig.tight_layout(); F["lc"] = savefig(fig, tmp, "lc")
        # complexity
        d, tr, va = s["depth"]
        fig, ax = plt.subplots(figsize=(5.6, 2.6))
        ax.plot(d, tr, "o-", color=RED, label="train"); ax.plot(d, va, "o-", color=BLUE, label="validation")
        ax.axvline(s["best_depth"], color=GREEN, ls="--", lw=1)
        ax.text(s["best_depth"] + .15, ax.get_ylim()[0] + .01, f"best depth = {s['best_depth']}", color=GREEN, fontsize=8)
        ax.set_xlabel("decision-tree max depth (model complexity)"); ax.set_ylabel("balanced accuracy")
        ax.set_title("Underfit to overfit as complexity grows"); ax.legend(fontsize=7)
        F["depth"] = savefig(fig, tmp, "depth")
        # noise robustness
        sig, out = s["noise"]
        fig, ax = plt.subplots(figsize=(5.6, 2.5))
        for (nm, v), c in zip(out.items(), (BLUE, AMBER)): ax.plot(sig, v, "o-", color=c, label=nm)
        ax.set_xlabel("noise added to features (in std units)"); ax.set_ylabel("balanced accuracy")
        ax.set_title("Stability under feature noise"); ax.legend(fontsize=7)
        F["noise"] = savefig(fig, tmp, "noise")
        # label confidence
        fig, ax = plt.subplots(figsize=(5.6, 2.4))
        ax.hist(s["own_conf"], bins=20, color=BLUE)
        ax.axvline(.5, color=RED, ls="--", lw=1)
        ax.set_xlabel("out-of-fold probability the model gives to the row's OWN label")
        ax.set_title(f"Label sanity (estimated suspect labels: {s['noise_rate']:.1%})"); ax.set_ylabel("rows")
        F["labels"] = savefig(fig, tmp, "labels")
        # mlp curve
        tr_l, va_l, be = s["mlp"]
        fig, ax = plt.subplots(figsize=(5.6, 2.6))
        ax.plot(range(1, len(tr_l) + 1), tr_l, color=RED, label="train loss")
        ax.plot(range(1, len(va_l) + 1), va_l, color=BLUE, label="validation loss")
        ax.axvline(be, color=GREEN, ls="--", lw=1.2)
        ax.text(be + .5, max(va_l) * .9, f"best epoch = {be}", color=GREEN, fontsize=8)
        ax.set_xlabel("epoch"); ax.set_ylabel("log-loss"); ax.set_title("MLP training curve (early stopping)"); ax.legend(fontsize=7)
        F["mlp"] = savefig(fig, tmp, "mlp")


# ----------------------------------------------------------------------------- PDF
def build_pdf(a: Audit, path):
    a.make_figures()
    S = getSampleStyleSheet()
    H1 = ParagraphStyle("H1", parent=S["Heading1"], textColor=colors.HexColor(NAVY), fontSize=15, spaceBefore=6, spaceAfter=6)
    H2 = ParagraphStyle("H2", parent=S["Heading2"], textColor=colors.HexColor(BLUE), fontSize=11, spaceBefore=8, spaceAfter=3)
    B = ParagraphStyle("B", parent=S["BodyText"], fontSize=9, leading=12.5)
    SM = ParagraphStyle("SM", parent=B, fontSize=8, leading=10.5)
    CAP = ParagraphStyle("CAP", parent=SM, textColor=colors.HexColor(GREY), alignment=TA_CENTER)
    W = A4[0] - 3.6 * cm

    def img(key, w=W, ratio=None):
        from PIL import Image as PI
        iw, ih = PI.open(a.figs[key]).size
        w = min(w, W)
        return Image(a.figs[key], width=w, height=w * ih / iw)

    def fig(key, caption, w=W):
        return KeepTogether([img(key, w), Paragraph(caption, CAP), Spacer(1, 6)])

    def tbl(data, widths, header=True, extra=None):
        t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
        st = [("FONTSIZE", (0, 0), (-1, -1), 8), ("VALIGN", (0, 0), (-1, -1), "TOP"),
              ("GRID", (0, 0), (-1, -1), .3, colors.HexColor("#d5d9e3")),
              ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f6f8fc")])]
        if header:
            st += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(NAVY)), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                   ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
        t.setStyle(TableStyle(st + (extra or [])))
        return t

    P = lambda t, st=B: Paragraph(t, st)
    sevcol = {"FAIL": RED, "WARN": AMBER, "INFO": BLUE, "LOW": GREEN, "MEDIUM": AMBER, "HIGH": RED}
    s = a.stats
    story = []

    # ---- page 1: summary
    hdr = Table([[P(f"<font color='white' size=20><b>Dataset Audit Report</b></font><br/>"
                    f"<font color='#c9d3ee' size=10>{escape(a.name)} &nbsp;|&nbsp; target: <b>{escape(a.target)}</b> &nbsp;|&nbsp; "
                    f"{datetime.date.today():%d %b %Y}</font>")]], colWidths=[W])
    hdr.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(NAVY)), ("LEFTPADDING", (0, 0), (-1, -1), 14),
                             ("TOPPADDING", (0, 0), (-1, -1), 14), ("BOTTOMPADDING", (0, 0), (-1, -1), 14)]))
    story += [hdr, Spacer(1, 10)]
    n_f = sum(f.sev == "FAIL" for f in a.findings); n_w = sum(f.sev == "WARN" for f in a.findings)
    samp = (f" Model-based checks used a stratified sample of {a.n:,} of {a.n_full:,} rows; the duplicate rate was measured on all rows."
            if a.n_full > a.n else "")
    left = [P(f"<b>Verdict:</b> <font color='{sevcol['LOW'] if a.score >= 80 else sevcol['MEDIUM'] if a.score >= 55 else sevcol['HIGH']}'>"
              f"<b>{a.verdict}</b></font>"),
            Spacer(1, 4),
            P(f"<b>{a.n}</b> rows, <b>{a.p}</b> features (after encoding), <b>{a.k}</b> classes. "
              f"The audit raised <font color='{RED}'><b>{n_f} critical</b></font> and <font color='{AMBER}'><b>{n_w} warning</b></font> finding(s)." + samp),
            Spacer(1, 4),
            P(f"<b>MLP training budget:</b> about <b>{s['rec_epochs']} epochs</b> (batch {s['batch']}, ~{s['mlp_steps']} optimizer steps), "
              f"roughly <b>{s['rec_time']:.1f} s</b> on this machine. Best validation epoch was {s['mlp'][2]}; recommendation adds a 20% margin.", B)]
    top = Table([[img("gauge", 5.2 * cm), left]], colWidths=[5.6 * cm, W - 5.6 * cm])
    top.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    story += [top, Spacer(1, 6), P("Risk assessment", H1)]
    rows = [["Risk", "Level", "Why"]] + [[r, l, P(", ".join(w) or "-", SM)] for r, l, w in a.risks]
    ex = [("TEXTCOLOR", (1, i + 1), (1, i + 1), colors.HexColor(sevcol[l])) for i, (_, l, _) in enumerate(a.risks)]
    ex += [("FONTNAME", (1, 1), (1, -1), "Helvetica-Bold"), ("FONTSIZE", (0, 1), (1, -1), 10)]
    story += [tbl(rows, [3.6 * cm, 2.4 * cm, W - 6 * cm], extra=ex), Spacer(1, 8), P("Findings", H1)]
    order = {"FAIL": 0, "WARN": 1, "INFO": 2}
    fl = sorted(a.findings, key=lambda f: order[f.sev])
    if fl:
        rows = [["Level", "Area", "Finding", "What to do"]] + [[f.sev, P(f.area, SM), P(escape(f.title), SM), P(escape(f.fix), SM)] for f in fl]
        ex = [("TEXTCOLOR", (0, i + 1), (0, i + 1), colors.HexColor(sevcol[f.sev])) for i, f in enumerate(fl)]
        ex += [("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold")]
        story.append(tbl(rows, [1.2 * cm, 2.4 * cm, (W - 3.6 * cm) * .45, (W - 3.6 * cm) * .55], extra=ex))
    else:
        story.append(P("No issues found. This dataset passed every check."))
    story.append(PageBreak())

    # ---- page 2: structure
    story += [P("1. Data overview and hygiene", H1),
              P("Class balance and missingness decide which metrics and preprocessing are valid. Heavy imbalance makes plain accuracy misleading; "
                "missing values must be imputed inside the cross-validation loop to avoid leaking statistics from the test fold."),
              fig("overview", "Figure 1. Class distribution and columns with missing values.")]
    num = a.X_raw.select_dtypes("number")
    rows = [["Column", "Type", "Missing", "Unique", "Mean", "Std", "|Skew|", "Outliers"]]
    for c in list(a.X_raw.columns)[:22]:
        isn = c in num.columns
        rows.append([escape(str(c))[:24], str(a.X_raw[c].dtype)[:8], f"{a.X_raw[c].isna().mean():.1%}", a.X_raw[c].nunique(),
                     f"{num[c].mean():.3g}" if isn else "-", f"{num[c].std():.3g}" if isn else "-",
                     f"{s['skew'][c]:.1f}" if isn and "skew" in s else "-",
                     f"{s['outliers'][c]:.0%}" if isn and "outliers" in s else "-"])
    story += [P("Column profile" + (" (first 22 columns)" if a.X_raw.shape[1] > 22 else ""), H2),
              tbl(rows, [4.2 * cm, 1.8 * cm, 1.7 * cm, 1.5 * cm, 1.9 * cm, 1.9 * cm, 1.6 * cm, W - 14.6 * cm]), Spacer(1, 6)]
    if "dist" in a.figs:
        story.append(fig("dist", "Figure 2. Distributions split by class. Strong separation means an easy feature; identical overlap means weak information.", W))
    story += [PageBreak(), P("2. Redundancy and correlation", H1),
              P("Highly correlated features carry the same information twice. They destabilise linear-model coefficients, "
                "split importance between copies, and add dimensions without adding signal."),
              fig("corr", "Figure 3. Pairwise correlation of the 18 highest-variance features.", 11.5 * cm)]
    cp = s["corr_pairs"]
    if cp:
        rows = [["Feature A", "Feature B", "|r|"]] + [[escape(str(x))[:30], escape(str(y))[:30], f"{v:.3f}"] for x, y, v in sorted(cp, key=lambda t: -t[2])[:12]]
        story += [P(f"Near-duplicate pairs (|r| > 0.95): {len(cp)}", H2), tbl(rows, [6 * cm, 6 * cm, 2.5 * cm])]
    story.append(PageBreak())

    # ---- page 3: shortcuts
    story += [P("3. Shortcut learning and leakage", H1),
              P("A model will happily learn whatever predicts the label fastest, including cues that will not exist in production. "
                "Two tests: (a) can any <i>single</i> feature predict the label almost perfectly? (b) does one feature account for most of the model's skill? "
                "Red bars above the dashed line (0.95) are leakage suspects."),
              fig("leak", "Figure 4. Left: single-feature predictive power. Right: permutation importance of a random forest on held-out rows.")]
    if a.leak_cols:
        story.append(P("<b>Suspect columns:</b> " + ", ".join(escape(str(c)) for c in dict.fromkeys(a.leak_cols)) +
                       ". Remove them (or prove they are known at prediction time) and re-run the audit.", B))
    else:
        story.append(P("No single-feature leaks detected."))
    story += [Spacer(1, 6), P("Is there learnable signal at all?", H2),
              P("We cross-validate a logistic regression on the real labels and on 15 randomly shuffled copies of the labels. "
                "If the real score does not clearly beat the shuffled distribution, the features contain no usable information."),
              fig("signal", "Figure 5. Real-label score against the shuffled-label baseline.", 12 * cm), PageBreak()]

    # ---- page 4: bias/variance
    note = ([P(f"<b>Note:</b> suspected leak/ID columns ({escape(', '.join(map(str, s['removed'])))}) were removed for sections 4-6, "
               "so these results describe what the model can learn from the legitimate features only.", B), Spacer(1, 4)]
            if s.get("removed") else [])
    story += [P("4. Will it overfit or underfit?", H1)] + note + [
              P("Learning curves show how train and validation scores change with data size. A large gap that does not close means high variance (overfitting); "
                "two low, converged curves mean high bias (underfitting); a validation curve that is still climbing means more data would help."),
              fig("lc", "Figure 6. Learning curves (band = 1 std over CV folds; dotted = guessing baseline)."),
              P("Model-complexity sweep", H2),
              P(f"Increasing tree depth moves the model from underfit (left) to overfit (right). Validation accuracy peaks at depth "
                f"<b>{s['best_depth']}</b>; deeper trees only memorise the training set."),
              fig("depth", "Figure 7. Train vs validation score as complexity increases.", 12 * cm), PageBreak()]

    # ---- page 5: robustness + labels + budget
    story += [P("5. Robustness and label quality", H1),
              P("A trustworthy dataset yields models that stay accurate under small perturbations and whose labels agree with the evidence in the features."),
              fig("noise", "Figure 8. Held-out balanced accuracy when Gaussian noise is added to the (standardised) features.", 12 * cm),
              fig("labels", "Figure 9. Out-of-fold confidence in each row's own label. A pile-up near 0 indicates mislabeled rows (simplified confident learning).", 12 * cm),
              P("6. Training budget", H1),
              P(f"A small MLP (2 x 64 hidden units, Adam, batch {s['batch']}) was trained with early stopping. Validation loss was lowest at epoch "
                f"<b>{s['mlp'][2]}</b>. Recommended: <b>{s['rec_epochs']} epochs</b> with early stopping (patience about 10-20), about "
                f"<b>{s['rec_time']:.1f} s</b> total here ({s['per_epoch'] * 1000:.0f} ms per epoch). Treat this as an estimate for this dataset and this machine."),
              fig("mlp", "Figure 10. Train and validation loss per epoch; the dashed line marks the minimum validation loss.", 12 * cm), PageBreak()]

    # ---- recommendations
    recs = []
    for f in fl:
        if f.sev != "INFO" and f.fix and f.fix not in recs:
            recs.append(f.fix)
    recs += ["Split with stratification and fit every preprocessing step (imputer, scaler, encoder) inside a Pipeline on the training fold only.",
             "Report balanced accuracy / F1 / ROC-AUC alongside accuracy, using repeated stratified k-fold on small data.",
             f"Start with simple models (logistic regression, shallow trees up to depth {s['best_depth']}) as the baseline before trying MLPs."]
    story += [P("7. Recommended actions", H1)] + [P(f"{i + 1}. {escape(r)}") for i, r in enumerate(recs)]
    story += [Spacer(1, 8), P("Methodology and limits", H2),
              P("Checks: IQR outliers (3 x IQR), Pearson correlation, single-feature depth-3 trees (5-fold CV), permutation importance, label-shuffle test (15 shuffles), "
                "learning curves and depth sweep with stratified CV, Gaussian-noise stability test, simplified confident-learning label audit, early-stopped MLP. "
                "The health score is a heuristic: 15 points per critical finding, 5 per warning, 25-30 for leakage or no signal, capped per area. "
                "All estimates come from this dataset only; small datasets give wide error bars. This tool targets tabular classification.", SM)]
    sus = s["suspects"]
    if len(sus):
        story += [PageBreak(), P("Appendix: suspect label rows", H1),
                  P("Row indices (0-based, after dropping rows with a missing target) whose label is confidently contradicted by the model. "
                    "Inspect them manually before deciding to relabel or remove.", B)]
        rows = [["Row", "Given label", "Model's own-label prob"]] + [[int(i), str(a.classes[a.y[i]]), f"{s['own_conf'][i]:.2f}"]
                                                                       for i in sus[np.argsort(s["own_conf"][sus])][:40]]
        story.append(tbl(rows, [3 * cm, 4 * cm, 5 * cm]))

    def footer(canvas, doc):
        canvas.saveState(); canvas.setFont("Helvetica", 7.5); canvas.setFillColor(colors.HexColor(GREY))
        canvas.drawString(1.8 * cm, 1.1 * cm, "DataDoctor audit  |  " + a.name)
        canvas.drawRightString(A4[0] - 1.8 * cm, 1.1 * cm, f"Page {doc.page}")
        canvas.restoreState()

    SimpleDocTemplate(path, pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.6 * cm, bottomMargin=1.8 * cm,
                      title=f"DataDoctor audit - {a.name}", author="DataDoctor").build(story, onFirstPage=footer, onLaterPages=footer)
    return path


# ----------------------------------------------------------------------------- demo data + CLI
def demo(kind):
    from sklearn.datasets import load_breast_cancer
    d = load_breast_cancer(as_frame=True).frame
    rng = np.random.default_rng(SEED)
    if kind == "clean":
        return d, "target", "breast_cancer (clean)"
    if kind == "noisy":
        flip = rng.random(len(d)) < .12
        d.loc[flip, "target"] = 1 - d.loc[flip, "target"]
        return d, "target", "breast_cancer (12% labels flipped)"
    d["patient_id"] = np.arange(len(d)) * 7 + 1000                         # ID column
    d["leaky_score"] = d["target"] + rng.normal(0, .05, len(d))            # leak
    d["const_flag"] = 1                                                    # constant
    d["mean radius copy"] = d["mean radius"] * 1.001 + rng.normal(0, .01, len(d))  # near-duplicate
    d.loc[rng.random(len(d)) < .3, "mean texture"] = np.nan                # 30% missing
    d = pd.concat([d, d.sample(frac=.12, random_state=SEED)], ignore_index=True)  # duplicates
    return d, "target", "breast_cancer (sabotaged)"


def main():
    ap = argparse.ArgumentParser(description="DataDoctor: audit a classification dataset and write a PDF report.")
    ap.add_argument("csv", nargs="?"); ap.add_argument("--target"); ap.add_argument("--out", default="audit_report.pdf")
    ap.add_argument("--demo", choices=["clean", "dirty", "noisy"])
    args = ap.parse_args()
    if args.csv:
        if not args.target:
            ap.error("--target is required with a CSV")
        df, target, name = pd.read_csv(args.csv), args.target, os.path.basename(args.csv)
        match = {c.strip().lower(): c for c in df.columns}
        if target.strip().lower() in match:
            target = match[target.strip().lower()]
        else:
            cand = [c for c in df.columns if 2 <= df[c].nunique() <= 10]
            print(f"\nColumn '{target}' not found in {name}.\n\nYour columns ({df.shape[1]}):")
            for c in df.columns:
                print(f"  - {c}   [{df[c].dtype}, {df[c].nunique()} unique]")
            print("\nColumns that could work as a classification target (2-10 distinct values):")
            print("  " + (", ".join(cand) if cand else "none - create a category column first, e.g. high/low from a numeric one"))
            raise SystemExit(f"\nRe-run with:  python datadoctor.py {args.csv} --target <column>")
    else:
        df, target, name = demo(args.demo or "dirty")
    t0 = time.time()
    a = Audit(df, target, name).run()
    build_pdf(a, args.out)
    print(f"Health score {a.score}/100 - {a.verdict}")
    for f in sorted(a.findings, key=lambda f: {"FAIL": 0, "WARN": 1, "INFO": 2}[f.sev]):
        print(f"  [{f.sev}] {f.area}: {f.title}")
    print(f"Report written to {args.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()