#!/usr/bin/env python3
"""Plot DAPO vs vanilla GRPO vs JanusPO trajectories on GSM8K (Qwen2.5-3B).

Generates the §6.2 figure for the EMNLP submission, showing that:
  • DAPO (AsymClip + no KL) peaks at ~58% step 10, then collapses;
    response length crashes from 250 → 3 tokens; grad_norm hits 0 by step 120.
  • Vanilla GRPO is stable but plateaus around 75% (paper baseline).
  • JanusPO (AsymClip + AsymKL + entropy guard + turn-weighted reward)
    climbs steadily to 84.5% with no collapse.

Output: emnlppaper/figures/dapo_collapse.{pdf,png}
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]  # ARPO/
PAPER_FIG_DIR = ROOT.parent / "emnlppaper" / "figures"
PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)

RUNS = {
    "DAPO (AsymClip + no KL)": dict(
        log=ROOT / "checkpoints/dapo_3b_gsm8k_v1/run.log",
        color="#d62728",
        marker="x",
    ),
    "RLOO (LOO baseline + no entropy)": dict(
        log=ROOT / "checkpoints/rloo_3b_gsm8k_v1/run.log",
        color="#9467bd",
        marker="^",
    ),
    "GSPO (seq-level ratio + no entropy)": dict(
        log=ROOT / "checkpoints/gspo_3b_gsm8k_v1/run.log",
        color="#ff7f0e",
        marker="D",
    ),
    "Vanilla GRPO (clean)": dict(
        log=ROOT / "checkpoints/grpo_clean_v2/run.log",
        color="#7f7f7f",
        marker="s",
    ),
    "JanusPO (ours)": dict(
        log=ROOT / "checkpoints/plan_c_asymkl_v2_ub25_v1/run.log",
        color="#2ca02c",
        marker="o",
    ),
}

# Right-most cap; DAPO ran to 233, others may run further
MAX_STEP = 233

VAL_KEY = "val-core/openai/gsm8k/reward/mean@1"

STEP_RE = re.compile(r"step:(\d+) -")
KV_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_/@\.\-]*):([\-0-9.]+(?:e[+\-]?\d+)?)")


def parse_log(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        print(f"warning: missing {path}")
        return rows
    with path.open() as f:
        for line in f:
            sm = STEP_RE.search(line)
            if not sm:
                continue
            step = int(sm.group(1))
            kv: dict[str, float] = {}
            for k, v in KV_RE.findall(line):
                try:
                    kv[k] = float(v)
                except ValueError:
                    continue
            # Keep every logged step, not just val steps — ppo_kl is logged
            # every step, val only every test_freq. We'll carry val_acc as NaN
            # when not present.
            rows.append({
                "step": step,
                "val_acc": kv.get(VAL_KEY),
                "resp_len": kv.get("response_length/mean"),
                "entropy": kv.get("actor/entropy_loss"),
                "grad_norm": kv.get("actor/grad_norm"),
                "ppo_kl": kv.get("actor/ppo_kl"),
                "kl_loss": kv.get("actor/kl_loss"),
            })
    rows = [r for r in rows if r["step"] <= MAX_STEP]
    return rows


def main() -> None:
    series = {name: parse_log(meta["log"]) for name, meta in RUNS.items()}
    for n, s in series.items():
        if s:
            with_val = [r for r in s if r["val_acc"] is not None]
            if with_val:
                best = max(with_val, key=lambda r: r["val_acc"])
                print(f"  {n:30s} {len(s):3d} pts  best={best['val_acc']*100:5.2f}% @ {best['step']}")
            else:
                print(f"  {n:30s} {len(s):3d} pts  (no val metrics)")
        else:
            print(f"  {n:30s}  NO DATA")

    # 2x3 grid:  (val_acc | resp_len | entropy) / (ppo_kl | kl_loss | grad_norm)
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 6.4), sharex=True)
    (ax_val, ax_resp, ax_ent), (ax_ppo, ax_kl, ax_gn) = axes

    for name, meta in RUNS.items():
        rows = series[name]
        if not rows:
            continue
        steps = np.array([r["step"] for r in rows])
        for ax, key, scale in [
            (ax_val,  "val_acc",   100.0),
            (ax_resp, "resp_len",    1.0),
            (ax_ent,  "entropy",     1.0),
            (ax_ppo,  "ppo_kl",      1.0),
            (ax_kl,   "kl_loss",     1.0),
            (ax_gn,   "grad_norm",   1.0),
        ]:
            ys_raw = [r[key] for r in rows]
            ys = np.array([v * scale if v is not None else np.nan for v in ys_raw])
            ax.plot(
                steps, ys,
                color=meta["color"], marker=meta["marker"],
                markersize=4, linewidth=1.6,
                label=name if ax is ax_val else None,
            )

    # peak / collapse annotations on the val_acc panel
    def _peak(series_rows):
        have_val = [r for r in series_rows if r["val_acc"] is not None]
        return max(have_val, key=lambda r: r["val_acc"]) if have_val else None

    dapo = series["DAPO (AsymClip + no KL)"]
    if dapo:
        peak = _peak(dapo)
        if peak:
            ax_val.annotate(
                f"DAPO peak\n{peak['val_acc']*100:.1f}%",
                xy=(peak["step"], peak["val_acc"] * 100),
                xytext=(peak["step"] + 18, peak["val_acc"] * 100 + 6),
                fontsize=8, color="#d62728",
                arrowprops=dict(arrowstyle="->", color="#d62728", lw=0.7),
            )
        # mark dead-policy regime
        dead_start = next(
            (r["step"] for r in dapo
             if r["val_acc"] is not None and r["val_acc"] < 0.01),
            None,
        )
        if dead_start is not None:
            ax_val.axvspan(dead_start, MAX_STEP, color="#d62728", alpha=0.07,
                           label="DAPO dead-policy zone (acc<1%)")

    janus = series["JanusPO (ours)"]
    if janus:
        peak = _peak(janus)
        if peak:
            ax_val.annotate(
                f"JanusPO peak\n{peak['val_acc']*100:.1f}%",
                xy=(peak["step"], peak["val_acc"] * 100),
                xytext=(peak["step"] - 90, peak["val_acc"] * 100 - 12),
                fontsize=8, color="#2ca02c",
                arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=0.7),
            )

    grpo = series["Vanilla GRPO (clean)"]
    if grpo:
        peak = _peak(grpo)
        if peak:
            ax_val.annotate(
                f"GRPO peak\n{peak['val_acc']*100:.1f}%",
                xy=(peak["step"], peak["val_acc"] * 100),
                xytext=(peak["step"] + 30, peak["val_acc"] * 100 - 18),
                fontsize=8, color="#444",
                arrowprops=dict(arrowstyle="->", color="#444", lw=0.7),
            )

    rloo = series.get("RLOO (LOO baseline + no entropy)")
    if rloo:
        peak = _peak(rloo)
        if peak:
            ax_val.annotate(
                f"RLOO peak\n{peak['val_acc']*100:.1f}%",
                xy=(peak["step"], peak["val_acc"] * 100),
                xytext=(peak["step"] + 55, peak["val_acc"] * 100 + 3),
                fontsize=8, color="#9467bd",
                arrowprops=dict(arrowstyle="->", color="#9467bd", lw=0.7),
            )

    gspo = series.get("GSPO (seq-level ratio + no entropy)")
    if gspo:
        peak = _peak(gspo)
        if peak:
            ax_val.annotate(
                f"GSPO peak\n{peak['val_acc']*100:.1f}%",
                xy=(peak["step"], peak["val_acc"] * 100),
                xytext=(peak["step"] + 25, peak["val_acc"] * 100 - 15),
                fontsize=8, color="#ff7f0e",
                arrowprops=dict(arrowstyle="->", color="#ff7f0e", lw=0.7),
            )

    ax_val.set_ylabel("Val mean@1 acc (%)")
    ax_val.set_title("(a) GSM8K validation accuracy")
    ax_val.set_ylim(-3, 95)
    ax_val.grid(alpha=0.3)
    ax_val.legend(loc="center right", fontsize=7, framealpha=0.9)

    ax_resp.set_ylabel("response length (tokens)")
    ax_resp.set_title("(b) Mean response length")
    ax_resp.set_yscale("symlog", linthresh=10)
    ax_resp.grid(alpha=0.3, which="both")
    ax_resp.axhline(10, color="#888", linestyle=":", linewidth=0.8)
    ax_resp.text(MAX_STEP * 0.62, 6, "garbage-token regime\n(<10 tok)",
                 fontsize=7, color="#666", ha="left")

    ax_ent.set_ylabel(r"entropy ($\mathcal{H}$)")
    ax_ent.set_title("(c) Policy entropy")
    ax_ent.grid(alpha=0.3)

    ax_ppo.set_ylabel(r"$\mathtt{actor/ppo\_kl}$")
    ax_ppo.set_title(r"(d) PPO ratio displacement $\log(\pi/\pi_{\mathrm{old}})$")
    ax_ppo.set_xlabel("training step")
    # shade the asymmetric-clip zone [log(1-0.20), log(1+0.28)] ≈ [-0.223, 0.247]
    # EVERY method's update exits this band by step 1, so the band itself is
    # visual proof that asymmetric clip alone is not sufficient.
    import math
    clip_lo, clip_hi = math.log(1.0 - 0.20), math.log(1.0 + 0.28)
    ax_ppo.axhspan(clip_lo, clip_hi, color="#888", alpha=0.15,
                   label=f"DAPO clip zone [{clip_lo:.2f}, {clip_hi:.2f}]")
    ax_ppo.axhline(0.0, color="#444", linestyle=":", linewidth=0.8)
    ax_ppo.legend(loc="lower right", fontsize=7, framealpha=0.9)
    ax_ppo.grid(alpha=0.3)
    ax_ppo.axhline(-0.22, color="#888", linestyle=":", linewidth=0.8)
    ax_ppo.axhline(+0.25, color="#888", linestyle=":", linewidth=0.8)
    ax_ppo.grid(alpha=0.3)

    ax_kl.set_ylabel(r"$\mathtt{actor/kl\_loss}$")
    ax_kl.set_title(r"(e) Ref-anchored KL $D_{\mathrm{KL}}(\pi\,\|\,\pi_{\mathrm{ref}})$")
    ax_kl.set_xlabel("training step")
    ax_kl.set_yscale("symlog", linthresh=1.0)
    ax_kl.grid(alpha=0.3)

    ax_gn.set_ylabel(r"$\|\nabla\|_2$")
    ax_gn.set_title("(f) Gradient norm")
    ax_gn.set_xlabel("training step")
    ax_gn.grid(alpha=0.3)
    ax_gn.set_yscale("symlog", linthresh=1.0)

    fig.suptitle(
        "DAPO, RLOO and GSPO all collapse without a KL/entropy anchor on Qwen2.5-3B / GSM8K\n"
        r"(matched setting: single GH200, batch=32, lr$=10^{-6}$, $G$=8, 233 steps)",
        fontsize=11, y=1.0,
    )
    fig.tight_layout()
    out_pdf = PAPER_FIG_DIR / "rl_collapse.pdf"
    out_png = PAPER_FIG_DIR / "rl_collapse.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=150)
    # keep the old filename as a copy for any backward-compat references
    fig.savefig(PAPER_FIG_DIR / "dapo_collapse.pdf", bbox_inches="tight")
    fig.savefig(PAPER_FIG_DIR / "dapo_collapse.png", bbox_inches="tight", dpi=150)
    print(f"\nwrote {out_pdf}\nwrote {out_png}")


if __name__ == "__main__":
    main()
