#!/usr/bin/env python3
"""
Mine phase2 (entropy ablation + β_KL sweep) results from run.logs and
produce:
  1. emnlppaper/tables/phase2_summary.csv   (numbers for §6.3 table)
  2. emnlppaper/figures/entropy_ablation.{pdf,png}  (JanusPO vs JanusPO-no-entropy trajectories)
  3. emnlppaper/figures/beta_sweep.{pdf,png}        (DAPO β sweep + JanusPO reference)
"""
from __future__ import annotations
import csv
import re
from pathlib import Path
from statistics import mean, stdev
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("")
FIG = Path("/figures")
TAB = Path("/tables")
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)


def extract(log: Path) -> dict:
    """Return per-step trajectory dict {step: {metric: value}} plus aggregates."""
    if not log.exists():
        return {}
    text = log.read_text(errors="ignore")
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    traj: dict[int, dict[str, float]] = {}
    # the log alternates between training-step lines and validation lines
    # capture metrics one at a time then attach to the step that immediately
    # precedes them.
    for m in re.finditer(
        r"step:(\d+)\s+-\s+([^\n]+)", text):
        step = int(m.group(1))
        payload = m.group(2)
        d = traj.setdefault(step, {})
        for kv in re.finditer(r"([a-zA-Z0-9_/@.\-]+):(-?[0-9.eE+\-]+)", payload):
            try:
                d[kv.group(1)] = float(kv.group(2))
            except ValueError:
                pass
    return traj


def aggregate(traj: dict) -> dict:
    """Pick out key headline numbers."""
    steps = sorted(traj.keys())
    if not steps:
        return {}
    val_key = "val-core/openai/gsm8k/reward/mean@1"
    val_pts = [(s, traj[s][val_key]) for s in steps if val_key in traj[s]]
    val_only = [v for _, v in val_pts]
    # response length
    resp_key = "actor/response_length/mean"
    resp = [traj[s][resp_key] for s in steps if resp_key in traj[s]]
    # actor/entropy_loss (training entropy)
    ent_keys = [k for k in next(iter(traj.values()), {}).keys() if "entropy" in k.lower()]
    ent_loss = []
    for s in steps:
        for k in ("actor/entropy_loss", "actor/entropy"):
            if k in traj[s]:
                ent_loss.append(traj[s][k]); break
    # step-1 ppo_kl (for first-update story)
    s1 = traj.get(1, {})
    return {
        "n_steps": steps[-1],
        "n_val": len(val_pts),
        "final_val": val_pts[-1][1] if val_pts else None,
        "peak_val": max(val_only) if val_only else None,
        "peak_step": val_pts[val_only.index(max(val_only))][0] if val_only else None,
        "val_std_after_peak": stdev(val_only[val_only.index(max(val_only)):]) if len(val_only) > 3 else None,
        "step1_ppo_kl": s1.get("actor/ppo_kl"),
        "step1_grad_norm": s1.get("actor/grad_norm"),
        "step1_entropy": s1.get("actor/entropy_loss") or s1.get("actor/entropy"),
        "final_resp_len": resp[-1] if resp else None,
        "val_points": val_pts,
        "entropy_loss_trace": ent_loss,
    }


RUNS = {
    # canonical 3B JanusPO baseline (entropy_coeff=0.001, asym KL active)
    "janus_baseline": "plan_c_asymkl_v2_ub25_v1",
    # entropy ablation: same script, entropy_coeff=0
    "janus_no_entropy": "janus_no_entropy_v1",
    # DAPO baseline (β_KL = 0)
    "dapo_beta_0":     "dapo_3b_gsm8k_v1",
    # DAPO β sweep
    "dapo_beta_1em4":  "dapo_betakl_1em4_v1",
    "dapo_beta_1em3":  "dapo_betakl_1em3_v1",
    "dapo_beta_1em2":  "dapo_betakl_1em2_v1",
}

DISPLAY = {
    "janus_baseline":    ("JanusPO (entropy coeff = 1e-3, AsymKL on)",      "#2ca02c", "o"),
    "janus_no_entropy":  ("JanusPO ablation: entropy coeff = 0",            "#e07b00", "s"),
    "dapo_beta_0":       (r"DAPO ($\beta_{\mathrm{KL}}=0$, original)",       "#7f7f7f", "v"),
    "dapo_beta_1em4":    (r"DAPO + $\beta_{\mathrm{KL}}=10^{-4}$",           "#9467bd", "^"),
    "dapo_beta_1em3":    (r"DAPO + $\beta_{\mathrm{KL}}=10^{-3}$",           "#1f77b4", "D"),
    "dapo_beta_1em2":    (r"DAPO + $\beta_{\mathrm{KL}}=10^{-2}$",           "#d62728", "P"),
}


def main():
    aggregates = {}
    trajs = {}
    for key, dirname in RUNS.items():
        log = ROOT / "checkpoints" / dirname / "run.log"
        traj = extract(log)
        agg = aggregate(traj)
        aggregates[key] = agg
        trajs[key] = traj
        if agg:
            print(f"{key:20s}  steps={agg['n_steps']:>4}  "
                  f"final={agg['final_val']:.3f}  peak={agg['peak_val']:.3f}@{agg['peak_step']:>3}  "
                  f"std_after_peak={agg['val_std_after_peak']!r}")

    # ── write CSV ────────────────────────────────────────────────────────
    csv_path = TAB / "phase2_summary.csv"
    fields = ["run", "n_steps", "final_val", "peak_val", "peak_step",
              "val_std_after_peak", "step1_ppo_kl", "step1_grad_norm",
              "step1_entropy", "final_resp_len"]
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for key, agg in aggregates.items():
            if not agg:
                continue
            row = {f: agg.get(f) for f in fields}
            row["run"] = key
            w.writerow(row)
    print(f"\nwrote {csv_path}")

    # ── entropy ablation figure ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharex=True)

    ax = axes[0]
    for key in ("janus_baseline", "janus_no_entropy"):
        label, color, marker = DISPLAY[key]
        pts = aggregates[key].get("val_points", [])
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, [y * 100 for y in ys], marker=marker, color=color,
                    lw=1.4, ms=4, label=label)
    ax.set_xlabel("training step")
    ax.set_ylabel("GSM8K val mean@1 (%)")
    ax.set_title("Validation accuracy")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(0, 100)

    # right panel: training entropy (variance reducer story)
    ax = axes[1]
    for key in ("janus_baseline", "janus_no_entropy"):
        label, color, marker = DISPLAY[key]
        traj = trajs[key]
        steps = sorted(traj.keys())
        ent = []
        ent_steps = []
        for s in steps:
            for k in ("actor/entropy_loss", "actor/entropy"):
                if k in traj[s]:
                    ent.append(traj[s][k]); ent_steps.append(s); break
        if ent:
            ax.plot(ent_steps, ent, color=color, lw=0.8, alpha=0.55, label=label)
    ax.set_xlabel("training step")
    ax.set_ylabel("actor entropy (per-token)")
    ax.set_title("Per-step training entropy (the variance source)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    fig.suptitle("Entropy bonus ablation: $\\lambda_H = 0$ vs JanusPO",
                 y=1.03, fontsize=11)
    fig.tight_layout()
    out = FIG / "entropy_ablation.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"wrote {out}")

    # ── β_KL sweep figure ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for key in ("dapo_beta_0", "dapo_beta_1em4", "dapo_beta_1em3", "dapo_beta_1em2",
                "janus_baseline"):
        label, color, marker = DISPLAY[key] if key in DISPLAY else (key, "k", "o")
        pts = aggregates.get(key, {}).get("val_points", [])
        if pts:
            xs, ys = zip(*pts)
            ls = "--" if "dapo" in key else "-"
            ax.plot(xs, [y * 100 for y in ys], marker=marker, color=color,
                    lw=1.4, ms=4, ls=ls, label=label)
    ax.set_xlabel("training step")
    ax.set_ylabel("GSM8K val mean@1 (%)")
    ax.set_title(r"DAPO $\beta_{\mathrm{KL}}$ sweep: four orders of magnitude, "
                 r"none recovers JanusPO")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=7.5)
    ax.set_ylim(-2, 100)
    fig.tight_layout()
    out = FIG / "beta_sweep.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"wrote {out}")

    # ── variance comparison printout (for prose) ─────────────────────────
    print("\n=== Validation noise comparison (std of mean@1 over all val points) ===")
    for key in ("janus_baseline", "janus_no_entropy"):
        pts = aggregates[key].get("val_points", [])
        if len(pts) > 3:
            vals = [v for _, v in pts]
            print(f"  {key:20s}  mean={mean(vals):.3f}  std={stdev(vals):.4f}  n={len(vals)}")


if __name__ == "__main__":
    main()
