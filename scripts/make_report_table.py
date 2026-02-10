from __future__ import annotations

import argparse
import csv
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench_csv", default="results/bench_summary.csv")
    ap.add_argument("--task", choices=["qa", "summarization", "all"], default="all")
    ap.add_argument("--budget", type=int, default=None)
    args = ap.parse_args()

    rows = []
    with open(args.bench_csv, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["budget"] = int(r["budget"])
            r["best_score"] = float(r["best_score"])
            r["runtime_s"] = float(r["runtime_s"])
            r["peak_mem_mb"] = float(r["peak_mem_mb"])
            r["heuristics"] = int(r["heuristics"])
            rows.append(r)

    if args.task != "all":
        rows = [r for r in rows if r["task"] == args.task]
    if args.budget is not None:
        rows = [r for r in rows if r["budget"] == args.budget]

    rows.sort(key=lambda r: (r["task"], r["budget"], r["heuristics"], r["algo"]))

    # Print LaTeX table
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\begin{tabular}{@{}lcccccc@{}}")
    print(r"\toprule")
    print(r"Task & Budget & Heur & Algo & Best score & Runtime(s) & Peak MB \\")
    print(r"\midrule")
    for r in rows:
        print(
            f"{r['task']} & {r['budget']} & {r['heuristics']} & {r['algo']} & "
            f"{r['best_score']:.4f} & {r['runtime_s']:.2f} & {r['peak_mem_mb']:.1f} \\\\"
        )
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{Phase 2 benchmark summary.}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()
