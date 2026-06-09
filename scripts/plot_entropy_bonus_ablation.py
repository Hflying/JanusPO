#!/usr/bin/env python3
"""
Plot the entropy-bonus ablation: janus_main (lambda=0.005) vs janus_no_entropy (lambda=0).

Two-panel figure showing (a) val_acc trajectory variance contrast,
(b) policy entropy. Both runs use identical AsymClip + AsymKL + everything else.

Output: emnlppaper/figures/entropy_bonus_ablation.{pdf,png}
"""
import re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("")
OUT = Path("/figures")

RUNS = {
    r"$\lambda_{\mathrm{ent}}\!=\!0.005$ (default JanusPO)": dict(
        log=ROOT / "checkpoints/plan_c_asymkl_v2_ub25_v1/run.log",
        color="#2ca02c",
        marker="o",
    ),
    r"$\lambda_{\mathrm{ent}}\!=\!0$ (no entropy bonus)": dict(
        log=ROOT / "checkpoints/janus_no_entropy_v1/run.log",
        color="#d62728",
        marker="s",
    ),
}

STEP_RE = re.compile(r"step:(\d+) - ")
VAL_RE = re.compile(r"val-core/openai/gsm8k/reward/mean@1:([0-9.]+)")
ENT_RE = re.compile(r"actor/entropy_loss:([\-0-9.]+)")
KL_RE = re.compile(r"actor/ppo_kl:([\-0-9.]+)")
RESP_RE = re.compile(r"response_length/mean:([0-9.]+)")


def parse(p: Path) -> list[dict]:
    if not p.exists():
        return []
    rows = []
    for line in p.read_text(errors="ignore").splitlines():
        sm = STEP_RE.search(line)
        if not sm:
            continue
        step = int(sm.group(1))
        rec = {"step": step}
        for name, rgx in [("val", VAL_RE), ("ent", ENT_RE), ("kl", KL_RE), ("resp", RESP_RE)]:
            m = rgx.search(line)
            if m:
                rec[name] = float(m.group(1))
        rows.append(rec)
    return rows


def main():
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), sharex=True)
    ax_val, ax_ent = axes

    for label, meta in RUNS.items():
        rows = parse(meta["log"])
        if not rows:
            print(f"WARN: empty {meta['log']}")
            continue
        val_rows = [r for r in rows if "val" in r]
        steps_v = [r["step"] for r in val_rows]
        vals = [r["val"] * 100 for r in val_rows]

        ax_val.plot(steps_v, vals, marker=meta["marker"], color=meta["color"],
                    lw=1.6, ms=5, label=label)

        # Entropy is logged every step; subsample for plotting
        ent_rows = [r for r in rows if "ent" in r]
        steps_e = [r["step"] for r in ent_rows[::2]]
        ents = [r["ent"] for r in ent_rows[::2]]
        ax_ent.plot(steps_e, ents, color=meta["color"], lw=1.0, alpha=0.7, label=label)

    ax_val.set_xlabel("training step")
    ax_val.set_ylabel("Val mean@1 acc (%)")
    ax_val.set_title("(a) Validation accuracy trajectory")
    ax_val.set_ylim(20, 95)
    ax_val.grid(alpha=0.3)
    ax_val.legend(loc="lower left", fontsize=8.5, framealpha=0.95)

    ax_ent.set_xlabel("training step")
    ax_ent.set_ylabel(r"policy entropy $\mathcal{H}(\pi_\theta)$")
    ax_ent.set_title("(b) Policy entropy")
    ax_ent.grid(alpha=0.3)
    ax_ent.set_ylim(0, 2.5)

    fig.suptitle(
        "Entropy bonus is a variance reducer, not a collapse preventer "
        r"(both runs: identical AsymClip + AsymKL + lr / batch / seed)",
        fontsize=10.5, y=1.0,
    )
    fig.tight_layout()
    out_pdf = OUT / "entropy_bonus_ablation.pdf"
    out_png = OUT / "entropy_bonus_ablation.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")

    # Print stats
    print("\n=== summary stats ===")
    for label, meta in RUNS.items():
        rows = parse(meta["log"])
        val_rows = [r for r in rows if "val" in r]
        if not val_rows:
            continue
        last12 = [r["val"] for r in val_rows[-12:]]
        mu = np.mean(last12); sd = np.std(last12, ddof=1)
        rng = max(last12) - min(last12)
        deltas = np.abs(np.diff([r["val"] for r in val_rows]))
        print(f"  {label}")
        print(f"    last12: mean={mu*100:.2f}%  std={sd*100:.2f}pp  range={rng*100:.2f}pp")
        print(f"    mean|delta|={np.mean(deltas)*100:.2f}pp  max|delta|={deltas.max()*100:.2f}pp")


if __name__ == "__main__":
    main()
