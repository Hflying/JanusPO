#!/usr/bin/env python3
"""
Extract best@k (=pass@k) + mean@k + worst@k from run.logs of the 6 pass@k
evals, plot the pass@k curves, and dump a tidy CSV for the paper.

Inputs  : checkpoints/_eval/<label>_<bench>_passk/run.log      (×6)
Outputs : emnlppaper/figures/passk_gsm8k.{pdf,png}
          emnlppaper/figures/passk_math500.{pdf,png}
          emnlppaper/tables/passk.csv
"""
from __future__ import annotations
import re
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("")
FIG_DIR = Path("/figures")
TAB_DIR = Path("/tables")
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

CKPTS = [
    ("sft",      "SFT init (Qwen2.5-3B-Instruct)", "#7f7f7f", "s"),
    ("grpo30",   "Vanilla GRPO (clean)",           "#1f77b4", "^"),
    ("janus200", "JanusPO (ours)",                 "#2ca02c", "o"),
]
BENCHES = [
    # (label, [(data_source_key, metric_kind), ...], display, scale_to_unit)
    # gsm8k / math500: reward is 0/1, so pass@k = reward/best@k/mean
    # aime: reward is -1/+1, but acc is 0/1 → use acc/best@k/mean and average aime-2024 + aime-2025
    ("gsm8k",   [("openai/gsm8k", "reward")],                                     "GSM8K (in-domain)",            False),
    ("math500", [("DigitalLearningGmbH/MATH-lighteval", "reward")],               "MATH-500 (zero-shot)",         False),
    ("aime",    [("aime-2024", "acc"), ("aime-2025", "acc")],                     "AIME 2024+2025 (competition)", False),
]

K_VALUES = [1, 2, 4, 8, 16]


def parse_passk(log_path: Path, sources: list[tuple[str, str]]) -> dict[int, dict[str, float]]:
    """
    Parse pass@k from a run.log; if multiple sources are provided, average them
    (each AIME year has 30 problems, so simple mean is correct).
    sources: list of (data_source_key, metric_kind) where metric_kind in {"reward","acc"}.
    Returns {k: {"pass": best@k, "mean": mean@k}}.
    """
    per_source: list[dict[int, dict[str, float]]] = []
    if not log_path.exists():
        return {k: {} for k in K_VALUES}
    text = log_path.read_text(errors="ignore")
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    for ds_key, kind_pref in sources:
        data: dict[int, dict[str, float]] = {k: {} for k in K_VALUES}
        for m in re.finditer(
            rf"(?:val-core|val-aux)/{re.escape(ds_key)}/{re.escape(kind_pref)}/([a-z]+)@(\d+)(?:/([a-z]+))?:([0-9.]+)",
            text,
        ):
            kind, k, suffix, val = m.group(1), int(m.group(2)), m.group(3), float(m.group(4))
            if k not in K_VALUES:
                continue
            if kind == "best":
                if suffix == "mean" or suffix is None:
                    data[k]["pass"] = val
            elif kind == "mean":
                if suffix is None:
                    data[k]["mean"] = val
        if "mean" in data.get(16, {}) and "pass" not in data.get(1, {}):
            data[1]["pass"] = data[16]["mean"]
        per_source.append(data)

    combined: dict[int, dict[str, float]] = {k: {} for k in K_VALUES}
    for k in K_VALUES:
        for field in ("pass", "mean"):
            vals = [src[k][field] for src in per_source if field in src.get(k, {})]
            if vals:
                combined[k][field] = sum(vals) / len(vals)
    return combined


def emit_csv(all_data: dict, path: Path):
    rows = []
    for (ckpt_label, _, _, _) in CKPTS:
        for (bench_label, _sources, _display, _scale) in BENCHES:
            for k in K_VALUES:
                d = all_data.get((ckpt_label, bench_label), {}).get(k, {})
                rows.append(dict(
                    ckpt=ckpt_label,
                    benchmark=bench_label,
                    k=k,
                    passk=d.get("pass"),
                    passk_std=d.get("pass_std"),
                    meank=d.get("mean"),
                ))
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    all_data: dict[tuple[str, str], dict] = {}
    for ckpt_label, _display, _color, _marker in CKPTS:
        for bench_label, sources, _display, _scale in BENCHES:
            log_path = ROOT / f"checkpoints/_eval/{ckpt_label}_{bench_label}_passk/run.log"
            all_data[(ckpt_label, bench_label)] = parse_passk(log_path, sources)

    # print a tidy table to stdout
    for bench_label, _sources, bench_display, _scale in BENCHES:
        print(f"\n=== {bench_display} ===")
        header = f"{'ckpt':<10}" + "".join(f"  pass@{k:<3}" for k in K_VALUES)
        print(header)
        for ckpt_label, display, _, _ in CKPTS:
            row = f"{ckpt_label:<10}"
            for k in K_VALUES:
                v = all_data.get((ckpt_label, bench_label), {}).get(k, {}).get("pass")
                row += f"  {v*100:>6.2f}" if v is not None else "  ------"
            print(row)

    emit_csv(all_data, TAB_DIR / "passk.csv")
    print(f"\nwrote CSV to {TAB_DIR/'passk.csv'}")

    for bench_label, _sources, bench_display, _scale in BENCHES:
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        for ckpt_label, display, color, marker in CKPTS:
            xs, ys = [], []
            for k in K_VALUES:
                v = all_data.get((ckpt_label, bench_label), {}).get(k, {}).get("pass")
                if v is not None:
                    xs.append(k); ys.append(v * 100)
            ax.plot(xs, ys, marker=marker, color=color, lw=1.4, ms=5, label=display)
        ax.set_xscale("log", base=2)
        ax.set_xticks(K_VALUES)
        ax.set_xticklabels([str(k) for k in K_VALUES])
        ax.set_xlabel("k  (log scale)")
        ax.set_ylabel("pass@k (%)")
        ax.set_title(f"pass@k on {bench_display}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        out_pdf = FIG_DIR / f"passk_{bench_label}.pdf"
        out_png = FIG_DIR / f"passk_{bench_label}.png"
        fig.savefig(out_pdf, bbox_inches="tight")
        fig.savefig(out_png, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"wrote {out_pdf}")

    # also a combined 1xN figure for the paper (one panel per benchmark)
    n = len(BENCHES)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 3.6), sharey=False)
    if n == 1:
        axes = [axes]
    for ax, (bench_label, _sources, bench_display, _scale) in zip(axes, BENCHES):
        for ckpt_label, display, color, marker in CKPTS:
            xs, ys = [], []
            for k in K_VALUES:
                v = all_data.get((ckpt_label, bench_label), {}).get(k, {}).get("pass")
                if v is not None:
                    xs.append(k); ys.append(v * 100)
            ax.plot(xs, ys, marker=marker, color=color, lw=1.5, ms=6, label=display)
        ax.set_xscale("log", base=2)
        ax.set_xticks(K_VALUES)
        ax.set_xticklabels([str(k) for k in K_VALUES])
        ax.set_xlabel(r"$k$")
        ax.set_ylabel("pass@$k$ (%)")
        ax.set_title(bench_display)
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle(
        "pass@$k$ on Qwen2.5-3B: JanusPO dominates at every $k$, not just mean@1",
        y=1.02, fontsize=11,
    )
    fig.tight_layout()
    out = FIG_DIR / "passk_combined.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
