"""Orchestrates the full audit and returns one JSON-serialisable report."""
from __future__ import annotations

import time

import numpy as np

from . import analyses as A
from . import checks as C
from . import planner as PL
from .core import DIMENSIONS, GOALS, METRIC_INFO, UseCase, jsonable
from .prep import prepare

STAGES = [
    ("prepare", "Preparing the analysis dataset"),
    ("hygiene", "Checking data hygiene"),
    ("target", "Examining the target"),
    ("leakage", "Hunting for leakage"),
    ("matrix", "Encoding features"),
    ("signal", "Testing for real signal (label-shuffle test)"),
    ("relevance", "Measuring feature relevance"),
    ("validity", "Testing validation realism (time / entity splits, drift)"),
    ("bakeoff", "Model bake-off"),
    ("subgroups", "Checking subgroups & decision threshold"),
    ("datasize", "Learning curve & data-size forecast"),
    ("labels", "Auditing label quality"),
    ("training", "Simulating training: loss curves & time"),
    ("plan", "Writing the diagnosis and plan"),
]


def run(raw, notes, prof, uc: UseCase, progress=None):
    progress = progress or (lambda **k: None)
    timings = {}
    t_all = time.perf_counter()

    def stage(key):
        progress(stage=key, sub="")
        timings[key] = time.perf_counter()

    def done(key):
        timings[key] = time.perf_counter() - timings[key]

    stage("prepare")
    P = prepare(raw, prof, uc)
    ctx = C.Ctx(raw, prof, P, notes, progress=lambda **k: progress(**k))
    done("prepare")
    for key, fn in (("hygiene", C.hygiene), ("target", C.target_checks), ("leakage", C.leakage)):
        stage(key); fn(ctx); done(key)
    if not P.features:
        raise ValueError("No usable feature columns are left after exclusions.")
    stage("matrix"); A.build_matrix(ctx); A.volume(ctx); done("matrix")
    for key, fn in (("signal", A.signal), ("relevance", A.relevance), ("validity", A.validity), ("bakeoff", A.bakeoff)):
        stage(key); fn(ctx); done(key)
    best = PL.choose_model(ctx)
    stage("subgroups"); A.threshold(ctx, best); A.subgroups(ctx, best); done("subgroups")
    stage("datasize"); A.data_size(ctx, best); done("datasize")
    stage("labels"); A.label_quality(ctx, best); done("labels")
    stage("training"); A.training_forecast(ctx, best); done("training")
    stage("plan")
    PL.fitness(ctx)
    PL.plan(ctx)
    PL.narrative(ctx)
    done("plan")

    order = {"critical": 0, "major": 1, "minor": 2, "info": 3}
    findings = sorted(ctx.findings, key=lambda f: (order[f.severity], list(DIMENSIONS).index(f.dimension)))
    report = {
        "usecase": uc.to_dict(),
        "task": P.task, "metric": P.metric, "metric_label": METRIC_INFO[P.metric][0], "goal": P.goal,
        "goal_label": GOALS[P.task][P.goal][0], "classes": P.classes, "positive": P.uc.positive_class,
        "load": notes, "repairs": P.repairs, "excluded": P.excluded, "features": P.features,
        "split_kind": ctx.split_kind,
        "profile": list(prof.values()),
        "findings": findings,
        "dimensions_meta": {k: {"label": v[0], "weight": v[1], "question": v[2]} for k, v in DIMENSIONS.items()},
        "timings": {k: round(v, 2) for k, v in timings.items()},
        "elapsed": round(time.perf_counter() - t_all, 1),
        **{k: v for k, v in ctx.R.items()},
    }
    return jsonable(report)
