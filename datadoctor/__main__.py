"""Command-line audit:  py -3 -m datadoctor data.csv --target Churn --goal recall --time signup_date"""
import argparse
import json
import re
import sys
from pathlib import Path

from . import UseCase, load_csv, profile_columns, run
from .engine import STAGES


def main():
    ap = argparse.ArgumentParser(prog="datadoctor", description="Use-case-aware dataset audit and ML plan.")
    ap.add_argument("csv")
    ap.add_argument("--target", required=True)
    ap.add_argument("--task", default="auto", choices=["auto", "classification", "regression"])
    ap.add_argument("--goal", default="", help="classification: balanced|recall|precision|ranking  regression: mae|rmse|mape")
    ap.add_argument("--title", default="")
    ap.add_argument("--positive", default=None, help="class of interest (binary)")
    ap.add_argument("--time", default=None, help="time column")
    ap.add_argument("--entity", default=None, help="entity column (customer / patient id)")
    ap.add_argument("--unavailable", default="", help="comma-separated columns not known at prediction time")
    ap.add_argument("--groups", default="", help="comma-separated subgroup columns")
    ap.add_argument("--interpretability", default="nice", choices=["not_needed", "nice", "required"])
    ap.add_argument("--deployment", default="server", choices=["server", "batch", "edge"])
    ap.add_argument("--out", default="datadoctor_report.json")
    a = ap.parse_args()

    df, notes = load_csv(Path(a.csv).read_bytes(), a.csv)
    prof = profile_columns(df)
    split = lambda s: [c.strip() for c in s.split(",") if c.strip()]
    uc = UseCase(target=a.target, task=a.task, goal=a.goal, title=a.title, positive_class=a.positive, time_column=a.time,
                 entity_column=a.entity, unavailable=split(a.unavailable), group_columns=split(a.groups),
                 interpretability=a.interpretability, deployment=a.deployment)
    labels = dict(STAGES)
    r = run(df, notes, prof, uc, progress=lambda stage=None, sub="": stage and print(f"  · {labels[stage]}", file=sys.stderr))
    Path(a.out).write_text(json.dumps(r, indent=1), encoding="utf-8")

    strip = lambda h: re.sub(r"<[^>]+>", "", h.replace("<li>", "\n   - "))
    print(f"\n{r['fitness']['verdict'].upper()}  ({r['fitness']['overall']}/100)\n")
    for p in r["narrative"]:
        print(strip(p), "\n")
    print("Findings:")
    for f in r["findings"]:
        if f["severity"] != "info":
            print(f"  [{f['severity']:8}] {f['title']}")
    print(f"\nRecommended model: {r['choice']['recommended']}")
    print(f"Full report (JSON): {a.out}   ·   for the visual report run:  py -3 server.py")


if __name__ == "__main__":
    main()
