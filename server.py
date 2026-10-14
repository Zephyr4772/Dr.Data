"""DataDoctor web server.

    py -3 server.py            ->  http://127.0.0.1:8000
"""
from __future__ import annotations

import io
import json
import re
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from datadoctor import UseCase, load_csv, profile_columns, run, suggest_usecase
from datadoctor.core import GOALS, jsonable
from datadoctor.engine import STAGES

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)
app = FastAPI(title="DataDoctor")
DATASETS: dict[str, dict] = {}
JOBS: dict[str, dict] = {}
POOL = ThreadPoolExecutor(max_workers=2)
LOCK = threading.Lock()


# ----------------------------------------------------------------------------- demo datasets
def _cancer(kind):
    from sklearn.datasets import load_breast_cancer
    d = load_breast_cancer(as_frame=True)
    df = d.frame.copy()
    df["diagnosis"] = np.where(df.pop("target") == 1, "benign", "malignant")
    if kind == "clean":
        return df
    rng = np.random.default_rng(42)
    df.insert(0, "patient_id", np.arange(len(df)) * 7 + 1000)
    df["biopsy_result_score"] = (df["diagnosis"] == "malignant") * 1.0 + rng.normal(0, .08, len(df))   # recorded after diagnosis
    df["scanner_flag"] = 1
    df["mean radius (mm)"] = df["mean radius"] * 1.0 + rng.normal(0, .01, len(df))
    df.loc[rng.random(len(df)) < .3, "mean texture"] = np.nan
    flip = rng.random(len(df)) < .06
    df.loc[flip, "diagnosis"] = np.where(df.loc[flip, "diagnosis"] == "benign", "malignant", "benign")
    df.loc[rng.random(len(df)) < .03, "diagnosis"] = "Malignant "
    df = pd.concat([df, df.sample(frac=.1, random_state=42)], ignore_index=True)
    return df


def _diabetes():
    from sklearn.datasets import load_diabetes
    d = load_diabetes(as_frame=True, scaled=False)
    df = d.frame.rename(columns={"target": "disease_progression", "s1": "tc", "s2": "ldl", "s3": "hdl", "s4": "tch", "s5": "ltg", "s6": "glu"})
    df["sex"] = df["sex"].map({1.0: "F", 2.0: "M"})
    return df


DEMOS = {
    "sales_returns": dict(name="Retail sales → will this order be returned?", file="sales_transaction.csv",
                          blurb="18k real-looking transactions with messy categories, impossible values and a weak target.",
                          usecase=dict(target="Return_Flag", task="classification", goal="recall", title="Flag orders likely to be returned",
                                       time_column="Order_Date", entity_column="Customer_ID", group_columns=["Customer_Gender", "Region"],
                                       unavailable=["Customer_Rating", "Delivery_Days"])),
    "sales_profit": dict(name="Retail sales → predict profit per order", file="sales_transaction.csv",
                         blurb="Same data, regression framing. Hides an exact-formula leak.",
                         usecase=dict(target="Profit", task="regression", goal="mae", title="Forecast profit of an order at checkout",
                                      time_column="Order_Date", entity_column="Customer_ID", group_columns=["Region"])),
    "cancer_dirty": dict(name="Breast cancer (sabotaged) → diagnosis", loader=lambda: _cancer("dirty"),
                         blurb="Classic dataset with planted leakage, ID column, duplicates, flipped labels and missing values.",
                         usecase=dict(target="diagnosis", task="classification", goal="recall", positive_class="malignant",
                                      title="Screen biopsies for malignancy", interpretability="nice")),
    "cancer_clean": dict(name="Breast cancer (clean) → diagnosis", loader=lambda: _cancer("clean"),
                         blurb="The untouched original, for comparison.",
                         usecase=dict(target="diagnosis", task="classification", goal="recall", positive_class="malignant",
                                      title="Screen biopsies for malignancy")),
    "diabetes": dict(name="Diabetes → disease progression (small data)", loader=_diabetes,
                     blurb="442 patients, 10 features: a small-data regression problem.",
                     usecase=dict(target="disease_progression", task="regression", goal="mae", title="Predict disease progression after one year",
                                  group_columns=["sex"], interpretability="required")),
}


def _register(df, notes, name):
    prof = profile_columns(df)
    did = uuid.uuid4().hex[:10]
    DATASETS[did] = {"df": df, "notes": notes, "prof": prof, "name": name}
    sug = suggest_usecase(df, prof)
    preview = df.head(8).astype(object).where(df.head(8).notna(), None).values.tolist()
    return jsonable({"dataset_id": did, "name": name, "notes": notes, "rows": len(df), "cols": df.shape[1],
                     "profile": list(prof.values()), "suggest": sug, "preview": {"columns": list(df.columns), "rows": preview},
                     "goals": {t: {k: v[0] for k, v in g.items()} for t, g in GOALS.items()}})


# ----------------------------------------------------------------------------- API
@app.get("/api/demos")
def demos():
    return [{"id": k, "name": v["name"], "blurb": v["blurb"], "target": v["usecase"]["target"], "task": v["usecase"]["task"]} for k, v in DEMOS.items()]


@app.post("/api/demo/{demo_id}")
def load_demo(demo_id: str):
    if demo_id not in DEMOS:
        raise HTTPException(404, "unknown demo")
    d = DEMOS[demo_id]
    if "file" in d:
        df, notes = load_csv((ROOT / d["file"]).read_bytes(), d["file"])
        name = d["file"]
    else:
        df = d["loader"]()
        notes = {"encoding": "built-in", "delimiter": ",", "rows": len(df), "cols": df.shape[1]}
        name = demo_id + ".csv"
    out = _register(df, notes, name)
    out["preset"] = d["usecase"]
    return out


@app.post("/api/dataset")
async def upload(file: UploadFile = File(...)):
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "empty file")
    try:
        if file.filename.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(raw))
            notes = {"encoding": "excel", "delimiter": "", "rows": len(df), "cols": df.shape[1]}
        else:
            df, notes = load_csv(raw, file.filename)
    except Exception as e:
        raise HTTPException(400, f"Could not read the file: {e}")
    if len(df) < 20 or df.shape[1] < 2:
        raise HTTPException(400, "Need at least 20 rows and 2 columns.")
    return _register(df, notes, file.filename)


class AnalyzeReq(BaseModel):
    dataset_id: str
    usecase: dict


@app.post("/api/analyze")
def analyze(req: AnalyzeReq):
    ds = DATASETS.get(req.dataset_id)
    if not ds:
        raise HTTPException(404, "dataset expired; upload again")
    try:
        uc = UseCase.from_dict(req.usecase)
    except TypeError as e:
        raise HTTPException(400, str(e))
    jid = uuid.uuid4().hex[:10]
    JOBS[jid] = {"status": "running", "stage": None, "sub": "", "done": [], "error": None, "result": None}

    def progress(stage=None, sub=""):
        j = JOBS[jid]
        if stage and stage != j["stage"]:
            if j["stage"]:
                j["done"].append(j["stage"])
            j["stage"] = stage
        j["sub"] = sub

    def work():
        try:
            res = run(ds["df"], ds["notes"], ds["prof"], uc, progress=progress)
            res["dataset_name"] = ds["name"]
            JOBS[jid].update(status="done", result=res)
            (REPORTS / f"{jid}.json").write_text(json.dumps(res), encoding="utf-8")
        except Exception as e:
            traceback.print_exc()
            JOBS[jid].update(status="error", error=str(e))

    POOL.submit(work)
    return {"job_id": jid, "stages": STAGES}


@app.get("/api/job/{jid}")
def job(jid: str):
    j = JOBS.get(jid)
    if not j:
        f = REPORTS / f"{jid}.json"
        if re.fullmatch(r"\w+", jid) and f.exists():
            return {"status": "done", "stage": None, "sub": "", "done": [], "error": None, "result": json.loads(f.read_text(encoding="utf-8"))}
        raise HTTPException(404, "unknown job")
    return {k: v for k, v in j.items()}


@app.middleware("http")
async def no_cache(request, call_next):
    resp = await call_next(request)
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.get("/")
def index():
    return FileResponse(ROOT / "web" / "index.html")


app.mount("/", StaticFiles(directory=ROOT / "web"), name="web")

if __name__ == "__main__":
    print("DataDoctor running at http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
