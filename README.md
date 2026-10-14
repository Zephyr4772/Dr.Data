# DataDoctor

**A pre-training audit for tabular datasets. You give it a dataset and a use case, and it tells you whether the data can support that use case. If it can, it gives you a plan for training the model.**

Most ML projects start by training a model. DataDoctor works on the step before that, which usually gets skipped: *is this data fit for the question I'm asking it?* The same table can be fine for one use case and useless for another. For example, the included retail dataset predicts profit well (once you remove a hidden leak) but carries **zero** signal for predicting returns. No amount of fine-tuning changes that.

---

## What it does

```
 dataset + use case ──► profile ──► hygiene ──► target ──► leakage ──► signal ──► relevance
                                                                                     │
   diagnosis + plan ◄── training forecast ◄── data size ◄── subgroups ◄── bake-off ◄── validation realism
```

| Stage | Question it answers | How |
|---|---|---|
| **Profiling** | What is each column, really? | Types are inferred from the values, not the dtypes. It detects numbers stored as text, day-first dates, times of day, IDs, free text and personal data (PII). |
| **Hygiene** | Are values clean, consistent, plausible? | Missingness, including *informative* missingness (a chi-square / Mann-Whitney test of the target on missing vs present rows). Exact and key duplicates, and conflicting labels on identical inputs. Spelling variants (`PayPal` / `paypal`). Impossible values (negative quantities, discount > 100%, age 4). Robust-z outliers, split into heavy tails vs glitches. Redundant encodings, and **columns that are exact formulas of other columns, plus the rows that break the formula**. |
| **Target** | Is the target well-posed for the goal? | Class balance and the accuracy paradox, rare classes, skew, zero-inflation, whether MAPE is safe. |
| **Leakage** | Could a model cheat? | It measures how well each column predicts the target on its own. It runs a **robust exact-formula search** (e.g. `Profit = Sales_Amount − Cost_Amount`) and looks for post-outcome naming cues. Columns you declare as *not known at prediction time* are excluded. Leaks are removed before every later stage. |
| **Signal** | Is there anything to learn at all? | **Label-permutation test**: the same model is trained on the real labels and on 10 shuffled copies. If the real score sits inside the chance cloud, stop. |
| **Relevance** | Which features matter? | Grouped permutation importance on held-out data, with a warning that redundant features look useless. |
| **Validation realism** | Will the offline score hold up? | Random vs **forward-in-time** vs **grouped-by-entity** CV on the same model. Also target drift over time, and **adversarial validation** (can a model tell old rows from new?), with time-proxy columns excluded. |
| **Model bake-off** | Which model, and why? | Six model families with identical preprocessing and folds. It measures every metric, fit time, latency and model size. The pick uses the **one-standard-error rule** plus your interpretability and deployment constraints. For binary tasks it also tunes the **decision threshold** to your goal. |
| **Subgroups** | Does it work for everyone? | Out-of-fold metric per group, under-represented groups. This is skipped when there's no signal, because the gaps would be noise. |
| **Data size** | Would more data help? | A learning curve fitted with an inverse power law and extrapolated. It tells you whether you're limited by data or by features. |
| **Labels** | Are labels trustworthy? | Simplified confident learning (classification) or extreme out-of-fold residuals (regression). It lists the suspect rows. |
| **Training forecast** | What will training look like? | Real loss curves from short runs (boosting rounds and MLP epochs), read automatically: best iteration, overfitting onset, gap. Fit time is measured at 3 sizes and extrapolated to a full fit, CV and a 30-trial search. A **field guide** shows six loss-curve patterns and highlights the one closest to your data. |
| **Plan** | What do I do now? | Ordered steps, hyperparameter ranges, the validation protocol, and **generated scikit-learn starter code** that implements the plan. |

Every finding comes with **evidence** (what was measured), **why it matters for your use case**, **what to do**, and what the tool already handled automatically.

### The use case drives the analysis
The form asks for:
- the target and problem type
- your goal (e.g. *catch positives* → F2 + threshold tuning; *avoid big misses* → RMSE)
- the class of interest
- a time column and an entity column
- columns not known at prediction time
- subgroups to check for fairness
- interpretability needs and deployment target (server / batch / edge)

So the same CSV gets a different verdict, metric, split strategy and model recommendation depending on what you want to do with it.

---

## Run it

```bash
py -3 -m pip install -r requirements.txt
py -3 server.py
```
Open http://127.0.0.1:8000. Upload a CSV/Excel file or open one of the case files. Reports are saved in `reports/` and reopen at `#job=<id>`. **Export PDF** prints the full report.

Command line (no UI):
```bash
py -3 -m datadoctor sales_transaction.csv --target Profit --goal mae --time Order_Date --entity Customer_ID --groups Region
```

A run takes roughly 5–45 s. Model-based checks use a representative sample of at most 5,000 rows (stratified, and kept in time order when there's a time column).

---

## Demo script for class (≈6 minutes)

1. **Breast cancer (clean)** → *Fit for purpose, 97*. Many columns are strong on their own, and the tool explains that this means *an easy problem*, not a leak.
2. **Breast cancer (sabotaged)** → *60*. Show that it recovers every planted defect: the leaky `biopsy_result_score`, 9% duplicates, the `"Malignant "` spelling variant, a constant column, the ID, and ~6% flipped labels (it estimates 6.5% and lists the rows).
3. **Retail → profit**. It finds `Profit = Sales_Amount − Cost_Amount` holds for 99.4% of rows. *"You don't need ML, or you're leaking."* The 106 rows that break the formula are flagged as data errors. Then show validation realism: R² 0.87 with random CV vs 0.75 when predicting the future.
4. **Retail → returns**. *Not fit, 30.* The label-shuffle chart shows the real score sitting inside the chance cloud. No model beats the no-skill baseline, and the report says: *"the next step is new information, not a new algorithm."* This is the punchline against fine-tuning everything.
5. Open **Training forecast** → the loss-curve field guide, and **Action plan** → the generated code.

---

## Project structure

```
datadoctor/
  core.py        UseCase, Finding, metric/goal tables, fitness dimensions
  profiling.py   robust CSV loading, semantic column typing, use-case suggestions
  prep.py        analysis frame (variant merge, dedup, target encoding), feature encoding, sampling
  checks.py      hygiene, target and leakage checks (incl. robust formula search)
  evaluation.py  metrics, no-skill baselines, split strategies, model zoo, CV loop
  analyses.py    signal, relevance, validation realism, bake-off, threshold, subgroups,
                 data size, label audit, training forecast
  planner.py     model choice, fitness scoring, action plan, narrative, code generation
  engine.py      orchestration + progress stages
  __main__.py    CLI
server.py        FastAPI: upload / demos / background jobs
web/             vanilla HTML/CSS/JS front-end with a hand-written SVG chart kit (no CDN needed)
legacy/          the original single-file version
```

## Limits
- Tabular data only. Free text is reduced to length features, and images aren't supported.
- All numbers are estimated from the dataset itself, so small datasets give wide error bars.
- The fitness score is a transparent heuristic, `100·exp(−penalty/90)` per dimension. The findings are the real output.
- The tool can't know your domain. "Check timing of X" prompts need a human answer.
