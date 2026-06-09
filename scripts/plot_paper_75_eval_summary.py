#!/usr/bin/env python3
"""Generate paper §7.5 eval-summary figure: scan over every checkpoint we
have evaluated, compare best@k reward & best@k F1 between v8 (JanusPO) and
T2-C (vanilla GRPO), with n=4 vs n=8 distinguished by marker.

This figure complements sec75_trajectory.{pdf,png} by surfacing how much
of any "peak" datapoint is sampling fluke versus a real ckpt-level signal.

Outputs:
  emnlppaper/figures/sec75_eval_summary.{pdf,png}
  emnlppaper/figures/sec75_eval_summary_data.csv
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "eval_results"
OUT_DIR = ROOT / "../emnlppaper/figures"

RUNS = {
    "JanusPO (v8, ours)": "gtpo_3b_deepsearch_v8_janus",
    "GRPO (T2-C, baseline)": "gtpo_3b_deepsearch_v8_grpo_paired",
}
COLORS = {
    "JanusPO (v8, ours)": "#1f4e9c",
    "GRPO (T2-C, baseline)": "#c0392b",
    "SFT only (step 273)": "#444",
}

KEY_RE = re.compile(
    r"\s*['\"]?val-(?:core|aux)/(DR_gaia|DR_hle_100)/"
    r"(reward|f1_score)/best@(\d+)/mean['\"]?:\s*([\-0-9\.eE]+)"
)

NAME_RE = re.compile(r"^(.+)_step_(\d+)(?:_n(\d+))?_summary\.txt$")
# Plain "_summary.txt" (no _n suffix) means it was n=4 by default.


def parse_summary(path: Path) -> dict:
    out = defaultdict(dict)  # task -> metric_name(best@k) -> value
    if not path.exists():
        return out
    text = path.read_text(errors="ignore")
    for m in KEY_RE.finditer(text):
        task, metric, k, val = m.group(1), m.group(2), int(m.group(3)), float(m.group(4))
        out[task][f"{metric}_best@{k}"] = val
    return out


def discover_evals() -> list[dict]:
    """Walk eval_results and pull every per-step summary into a list of records."""
    records = []
    for fp in sorted(EVAL_DIR.glob("*_summary.txt")):
        name = fp.name
        if name.startswith("round") or name.endswith("_batch_summary.txt"):
            continue
        m = NAME_RE.match(name)
        if not m:
            continue
        exp, step_str, n_str = m.group(1), m.group(2), m.group(3)
        step = int(step_str)
        n = int(n_str) if n_str else 4
        method = None
        for label, exp_name in RUNS.items():
            if exp == exp_name:
                method = label
                break
        if method is None:
            continue
        parsed = parse_summary(fp)
        for task in ("DR_gaia", "DR_hle_100"):
            if not parsed.get(task):
                continue
            records.append(
                {
                    "method": method,
                    "step": step,
                    "n": n,
                    "task": task,
                    "f1_best@4": parsed[task].get("f1_score_best@4"),
                    "f1_best@8": parsed[task].get("f1_score_best@8"),
                    "reward_best@4": parsed[task].get("reward_best@4"),
                    "reward_best@8": parsed[task].get("reward_best@8"),
                }
            )
    # SFT-only baseline (T2-A): pull from t2a_sft_only_zs_summary.txt
    sft_path = EVAL_DIR / "t2a_sft_only_zs_summary.txt"
    if sft_path.exists():
        sft = parse_summary(sft_path)
        for task in ("DR_gaia", "DR_hle_100"):
            if sft.get(task):
                records.append(
                    {
                        "method": "SFT only (step 273)",
                        "step": 0,  # plotted as horizontal reference, not on x-axis
                        "n": 4,
                        "task": task,
                        "f1_best@4": sft[task].get("f1_score_best@4"),
                        "f1_best@8": sft[task].get("f1_score_best@8"),
                        "reward_best@4": sft[task].get("reward_best@4"),
                        "reward_best@8": sft[task].get("reward_best@8"),
                    }
                )
    return records


def main():
    records = discover_evals()
    print(f"discovered {len(records)} (method,step,n,task) records")

    # ─── Plot: 2 rows (reward / f1) × 2 cols (GAIA / HLE) ────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.5))
    plt.subplots_adjust(hspace=0.40, wspace=0.28)

    panels = [
        (axes[0, 0], "DR_gaia", "reward", "GAIA reward (best@k, ↑ better)"),
        (axes[0, 1], "DR_hle_100", "reward", "HLE reward (best@k, ↑ better)"),
        (axes[1, 0], "DR_gaia", "f1_score", "GAIA token-F1 (best@k, ↑ better)"),
        (axes[1, 1], "DR_hle_100", "f1_score", "HLE token-F1 (best@k, ↑ better)"),
    ]

    legend_seen = set()
    for ax, task, kind, title in panels:
        # records use "f1_best@k" / "reward_best@k" (we strip "_score" for brevity)
        rkind = "f1" if kind == "f1_score" else kind
        # SFT baseline (horizontal dashed line, taken from best@4 of SFT eval)
        sft_vals = [
            r[f"{rkind}_best@4"]
            for r in records
            if r["method"] == "SFT only (step 273)" and r["task"] == task and r[f"{rkind}_best@4"] is not None
        ]
        if sft_vals:
            sft_y = sft_vals[0]
            if kind == "f1_score":
                sft_y *= 100
            ax.axhline(
                sft_y,
                color=COLORS["SFT only (step 273)"],
                linestyle="--",
                linewidth=1.1,
                label="SFT only (best@4)" if "sft" not in legend_seen else None,
            )
            legend_seen.add("sft")

        for method in ("JanusPO (v8, ours)", "GRPO (T2-C, baseline)"):
            # n=4 series: solid line + circle
            pts4 = sorted(
                (r["step"], r[f"{rkind}_best@4"])
                for r in records
                if r["method"] == method and r["task"] == task and r["n"] == 4 and r[f"{rkind}_best@4"] is not None
            )
            if pts4:
                xs, ys = zip(*pts4)
                if kind == "f1_score":
                    ys = [100 * y for y in ys]
                ax.plot(
                    xs,
                    ys,
                    color=COLORS[method],
                    marker="o",
                    markersize=8,
                    linewidth=1.5,
                    alpha=0.9,
                    label=f"{method}, best@4" if (method, "n4") not in legend_seen else None,
                )
                legend_seen.add((method, "n4"))

            # n=8 series: square (often only 1-2 points)
            pts8 = sorted(
                (r["step"], r[f"{rkind}_best@8"])
                for r in records
                if r["method"] == method and r["task"] == task and r["n"] == 8 and r[f"{rkind}_best@8"] is not None
            )
            if pts8:
                xs, ys = zip(*pts8)
                if kind == "f1_score":
                    ys = [100 * y for y in ys]
                ax.plot(
                    xs,
                    ys,
                    color=COLORS[method],
                    marker="s",
                    markersize=11,
                    linestyle="",
                    markerfacecolor="white",
                    markeredgecolor=COLORS[method],
                    markeredgewidth=2.0,
                    label=f"{method}, best@8 (verification)"
                    if (method, "n8") not in legend_seen
                    else None,
                )
                legend_seen.add((method, "n8"))

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("training step")
        if kind == "reward":
            ax.set_ylabel("reward")
        else:
            ax.set_ylabel("F1 (%)")
        ax.grid(True, alpha=0.25)
        ax.set_xlim(left=-5)

    # combined legend below
    handles, labels = [], []
    for ax in axes.flat:
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in labels:
                handles.append(hi)
                labels.append(li)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(
        "Paper §7.5 — best@k eval across all v8/T2-C checkpoints (n=4 = ●, n=8 verification = □)",
        fontsize=12,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUT_DIR / "sec75_eval_summary.pdf"
    png = OUT_DIR / "sec75_eval_summary.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=180, bbox_inches="tight")
    print(f"saved: {pdf}")
    print(f"saved: {png}")

    # csv
    csv_path = OUT_DIR / "sec75_eval_summary_data.csv"
    with csv_path.open("w") as f:
        f.write("method,step,n,task,f1_best@4,f1_best@8,reward_best@4,reward_best@8\n")
        for r in sorted(records, key=lambda x: (x["method"], x["task"], x["step"], x["n"])):
            f.write(
                f"{r['method']},{r['step']},{r['n']},{r['task']},"
                f"{r['f1_best@4']},{r['f1_best@8']},{r['reward_best@4']},{r['reward_best@8']}\n"
            )
    print(f"saved: {csv_path}")


if __name__ == "__main__":
    main()
