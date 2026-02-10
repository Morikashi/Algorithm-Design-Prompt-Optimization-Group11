from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def read_bench_summary(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # cast numeric fields
            r["budget"] = int(r["budget"])
            r["eval_count"] = int(r["eval_count"])
            r["best_score"] = float(r["best_score"])
            r["runtime_s"] = float(r["runtime_s"])
            r["peak_mem_mb"] = float(r["peak_mem_mb"])
            r["heuristics"] = int(r["heuristics"])
            rows.append(r)
    return rows


def read_trace(path: str):
    xs, ys = [], []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            xs.append(int(r["eval_idx"]))
            ys.append(float(r["best_so_far"]))
    return xs, ys


def plot_convergence_for_run(trace_path: str, out_path: str, title: str):
    xs, ys = read_trace(trace_path)
    plt.figure()
    plt.plot(xs, ys)
    plt.xlabel("Evaluation index")
    plt.ylabel("Best-so-far score")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_runtime_bars(rows, out_path: str, title: str):
    # Group by (task, budget, heuristics), compare algos
    groups = defaultdict(list)
    for r in rows:
        key = (r["task"], r["budget"], r["heuristics"])
        groups[key].append(r)

    # Make one plot per group
    for (task, budget, heur), items in groups.items():
        items = sorted(items, key=lambda x: x["algo"])
        labels = [x["algo"] for x in items]
        vals = [x["runtime_s"] for x in items]

        plt.figure()
        plt.bar(labels, vals)
        plt.xlabel("Algorithm")
        plt.ylabel("Runtime (s)")
        plt.title(f"{title} | task={task} budget={budget} heur={heur}")
        plt.tight_layout()
        base, ext = os.path.splitext(out_path)
        plt.savefig(f"{base}_{task}_B{budget}_H{heur}{ext}")
        plt.close()


def plot_memory_bars(rows, out_path: str, title: str):
    groups = defaultdict(list)
    for r in rows:
        key = (r["task"], r["budget"], r["heuristics"])
        groups[key].append(r)

    for (task, budget, heur), items in groups.items():
        items = sorted(items, key=lambda x: x["algo"])
        labels = [x["algo"] for x in items]
        vals = [x["peak_mem_mb"] for x in items]

        plt.figure()
        plt.bar(labels, vals)
        plt.xlabel("Algorithm")
        plt.ylabel("Peak memory (MB)")
        plt.title(f"{title} | task={task} budget={budget} heur={heur}")
        plt.tight_layout()
        base, ext = os.path.splitext(out_path)
        plt.savefig(f"{base}_{task}_B{budget}_H{heur}{ext}")
        plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench_csv", default="results/bench_summary.csv")
    ap.add_argument("--out_dir", default="results/plots")
    ap.add_argument("--max_convergence_plots", type=int, default=6)
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    rows = read_bench_summary(args.bench_csv)

    # 1) Convergence plots for up to N runs (most useful for report figures)
    for r in rows[: args.max_convergence_plots]:
        trace_path = r["trace_path"]
        run_id = r["run_id"]
        out_path = os.path.join(args.out_dir, f"convergence_{run_id}.png")
        plot_convergence_for_run(
            trace_path=trace_path,
            out_path=out_path,
            title=f"Convergence ({r['algo']}, {r['task']}, budget={r['budget']})",
        )

    # 2) Runtime comparison bars
    plot_runtime_bars(
        rows=rows,
        out_path=os.path.join(args.out_dir, "runtime.png"),
        title="Runtime comparison",
    )

    # 3) Memory comparison bars
    plot_memory_bars(
        rows=rows,
        out_path=os.path.join(args.out_dir, "memory.png"),
        title="Memory comparison",
    )

    print("Plots saved to:", args.out_dir)


if __name__ == "__main__":
    main()
