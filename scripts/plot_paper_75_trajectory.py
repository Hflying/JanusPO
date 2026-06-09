#!/usr/bin/env python3
"""Generate paper §7.5 trajectory figure: train-val mean@1 trajectories of
JanusPO (v8) vs vanilla GRPO (T2-C), plus train-side response_length and
score evolution.

Outputs (PDF + PNG):
  emnlppaper/figures/sec75_trajectory.{pdf,png}

Usage:
  python scripts/plot_paper_75_trajectory.py
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
LOGS = {
    "JanusPO (v8, ours)": ROOT / "checkpoints/gtpo_3b_deepsearch_v8_janus/run.log",
    "GRPO (T2-C, baseline)": ROOT
    / "checkpoints/gtpo_3b_deepsearch_v8_grpo_paired/run.log",
}
COLORS = {
    "JanusPO (v8, ours)": "#1f4e9c",
    "GRPO (T2-C, baseline)": "#c0392b",
}
SFT_BASELINES = {
    "gaia_reward": -0.981,
    "hle_reward": -0.976,
    "gaia_f1": 0.0,
    "hle_f1": 0.003,
}
SWEETSPOTS = {
    "JanusPO (v8, ours)": (75, "v8 step 75 (n=8 verified)"),
    "GRPO (T2-C, baseline)": (25, "T2-C step 25 (collapse trough)"),
}

# ---------------------------------------------------------------------------
# log parsing
# ---------------------------------------------------------------------------
STEP_RE = re.compile(r"step:(\d+)\s*-\s")
# allow '-' inside keys (e.g. val-core/...) and require key start with letter/underscore
KV_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_/@\.\-]*):([\-0-9.]+(?:e[+\-]?\d+)?)")

TRAIN_KEYS = (
    "critic/score/mean",
    "response_length/mean",
    "response_length/clip_ratio",
    "actor/entropy_loss",
)
VAL_KEYS = (
    "val-core/DR_gaia/reward/mean@1",
    "val-aux/DR_gaia/f1_score/mean@1",
    "val-aux/DR_gaia/num_turns/mean@1",
    "val-core/DR_hle_100/reward/mean@1",
    "val-aux/DR_hle_100/f1_score/mean@1",
    "val-aux/DR_hle_100/num_turns/mean@1",
)


def parse_log(path: Path) -> tuple[dict[str, dict[int, float]], dict[str, dict[int, float]]]:
    """Return (train_metrics, val_metrics) keyed by metric name → {step: value}."""
    train: dict[str, dict[int, float]] = defaultdict(dict)
    val: dict[str, dict[int, float]] = defaultdict(dict)
    if not path.exists():
        print(f"[warn] missing log: {path}")
        return train, val
    text = path.read_text(errors="ignore")
    for line in text.splitlines():
        m = STEP_RE.search(line)
        if not m:
            continue
        step = int(m.group(1))
        kv = dict(KV_RE.findall(line))
        for k in TRAIN_KEYS:
            if k in kv:
                try:
                    train[k][step] = float(kv[k])
                except ValueError:
                    pass
        for k in VAL_KEYS:
            if k in kv:
                try:
                    val[k][step] = float(kv[k])
                except ValueError:
                    pass
    return train, val


def series(d: dict[int, float]) -> tuple[list[int], list[float]]:
    if not d:
        return [], []
    xs = sorted(d.keys())
    return xs, [d[x] for x in xs]


# ---------------------------------------------------------------------------
# plot
# ---------------------------------------------------------------------------

def main():
    parsed = {label: parse_log(p) for label, p in LOGS.items()}

    fig, axes = plt.subplots(3, 2, figsize=(12.0, 10.5))
    plt.subplots_adjust(hspace=0.42, wspace=0.28)

    # row 1 — GAIA val reward / val f1
    # row 2 — HLE  val reward / val f1
    # row 3 — train response_length/mean  / train critic/score/mean
    panels = [
        (axes[0, 0], "val", "val-core/DR_gaia/reward/mean@1",
         "GAIA val reward (mean@1)", "reward",
         SFT_BASELINES["gaia_reward"]),
        (axes[0, 1], "val", "val-aux/DR_gaia/f1_score/mean@1",
         "GAIA val token-F1 (mean@1)", "F1",
         SFT_BASELINES["gaia_f1"]),
        (axes[1, 0], "val", "val-core/DR_hle_100/reward/mean@1",
         "HLE val reward (mean@1)", "reward",
         SFT_BASELINES["hle_reward"]),
        (axes[1, 1], "val", "val-aux/DR_hle_100/f1_score/mean@1",
         "HLE val token-F1 (mean@1)", "F1",
         SFT_BASELINES["hle_f1"]),
        (axes[2, 0], "train", "response_length/mean",
         "Train response length (degradation signal)", "tokens", None),
        (axes[2, 1], "train", "critic/score/mean",
         "Train batch reward (smoothed)", "reward", None),
    ]

    for ax, kind, key, title, ylabel, sft in panels:
        for label, (train, val) in parsed.items():
            d = (val if kind == "val" else train).get(key, {})
            xs, ys = series(d)
            if kind == "train" and len(ys) > 7:
                window = 5
                smooth = []
                for i in range(len(ys)):
                    lo = max(0, i - window)
                    smooth.append(sum(ys[lo : i + 1]) / (i - lo + 1))
                ys = smooth
            ax.plot(
                xs,
                ys,
                color=COLORS[label],
                linewidth=1.7,
                label=label if (ax is axes[0, 0]) else None,
                alpha=0.9,
            )
            sw_step, sw_label = SWEETSPOTS[label]
            if d.get(sw_step) is not None:
                y_at = d[sw_step]
                if kind == "train" and len(series(d)[0]) > 7:
                    pass  # raw value still informative
                ax.plot(
                    sw_step,
                    y_at,
                    marker="*",
                    markersize=14,
                    markerfacecolor=COLORS[label],
                    markeredgecolor="black",
                    markeredgewidth=0.8,
                    linestyle="",
                )

        if sft is not None:
            ax.axhline(
                sft,
                color="#666",
                linestyle="--",
                linewidth=1.0,
                label="SFT-only baseline" if (ax is axes[0, 0]) else None,
            )

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("training step")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.set_xlim(left=0)

    fig.suptitle(
        "Paper §7.5 — JanusPO vs vanilla GRPO multi-turn DeepSearch trajectories",
        fontsize=13,
        y=0.995,
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=3,
            frameon=False,
            fontsize=10,
            bbox_to_anchor=(0.5, -0.005),
        )

    out_dir = ROOT / "../emnlppaper/figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "sec75_trajectory.pdf"
    png_path = out_dir / "sec75_trajectory.png"
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    print(f"saved: {pdf_path}")
    print(f"saved: {png_path}")

    # also dump the underlying numbers as csv for the appendix
    csv_path = out_dir / "sec75_trajectory_data.csv"
    rows = []
    for label, (train, val) in parsed.items():
        for kind, store in (("val", val), ("train", train)):
            for k, d in store.items():
                for step, v in d.items():
                    rows.append((label, kind, k, step, v))
    rows.sort(key=lambda r: (r[0], r[1], r[2], r[3]))
    with csv_path.open("w") as f:
        f.write("run,kind,metric,step,value\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    print(f"saved: {csv_path}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
