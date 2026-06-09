#!/usr/bin/env python3
"""
Mine first-update diagnostics from verl training logs.

For each run we report four invariants measured at the very first gradient
step (step=1, after the first PPO optimization pass):
  • ppo_kl     — verl's actor/ppo_kl (mean over tokens of [π − π_old]; negative
                 value means new policy underestimates the old's probabilities)
  • kl_loss    — actor/kl_loss (ref-anchored KL, low-variance estimator)
  • entropy    — actor/entropy_loss (mean token entropy, higher = more explore)
  • grad_norm  — pre-clip gradient norm
  • resp_len   — response_length/mean (mean generated tokens)

We also extract three "peak / final" shape-summary numbers to contextualise:
  • peak val_acc, step at which peak was hit
  • step at which response_length first drops below 20 (collapse onset)
  • final val_acc at the last logged step

Output: a terse Markdown+LaTeX table on stdout, plus a JSON dump for
downstream plotting.
"""
from __future__ import annotations
import json
import re
from pathlib import Path

ROOT = Path("")

RUNS_3B = [
    ("GRPO clean",     "grpo_clean_v2"),
    ("DAPO",           "dapo_3b_gsm8k_v1"),
    ("RLOO",           "rloo_3b_gsm8k_v1"),
    ("GSPO",           "gspo_3b_gsm8k_v1"),
    ("JanusPO (ours)", "plan_c_asymkl_v2_ub25_v1"),
]

RUNS_7B = [
    ("DAPO 7B",          "dapo_7b_gsm8k_v1"),
    ("JanusPO 7B paged", "plan_c_asymkl_v2_7b_paged32_seed2_v1"),
    ("JanusPO 7B 8bit",  "plan_c_asymkl_v2_7b_8bit_seed2_v1"),
    ("JanusPO 7B bf16",  "plan_c_asymkl_v2_7b_bf16_v1"),
]

ANSI = re.compile(r"\x1b\[[0-9;]*m")

METRICS = [
    ("ppo_kl",     r"actor/ppo_kl:(-?[0-9.]+)"),
    ("kl_loss",    r"actor/kl_loss:(-?[0-9.]+)"),
    ("entropy",    r"actor/entropy_loss:(-?[0-9.]+)"),
    ("pg_loss",    r"actor/pg_loss:(-?[0-9.]+)"),
    ("grad_norm",  r"actor/grad_norm:(-?[0-9.]+)"),
    ("resp_len",   r"response_length/mean:([0-9.]+)"),
    ("clipfrac",   r"actor/pg_clipfrac:(-?[0-9.]+)"),
    ("val_acc",    r"val-core/openai/gsm8k/reward/mean@1:([0-9.]+)"),
]


def parse_step_lines(text: str) -> dict[int, dict[str, float]]:
    """Return {step: {metric: value}}. A step's record is one log line
    starting with "step:N -" which verl emits after each gradient update."""
    text = ANSI.sub("", text)
    out: dict[int, dict[str, float]] = {}
    for m in re.finditer(r"step:(\d+) - ([^\n]+)", text):
        step = int(m.group(1))
        body = m.group(2)
        rec = out.setdefault(step, {})
        for name, pat in METRICS:
            mm = re.search(pat, body)
            if mm:
                try:
                    rec[name] = float(mm.group(1))
                except ValueError:
                    pass
    return out


def summarise(run_name: str, ckpt_dir: str) -> dict:
    log_path = ROOT / f"checkpoints/{ckpt_dir}/run.log"
    if not log_path.exists():
        return {"run": run_name, "error": f"no run.log at {log_path}"}
    text = log_path.read_text(errors="ignore")
    steps = parse_step_lines(text)
    if not steps:
        return {"run": run_name, "error": "no step lines parsed"}

    step1 = steps.get(1, {})
    sorted_steps = sorted(steps.keys())
    # val_acc only logged at test_freq intervals
    val_records = [(s, steps[s]["val_acc"]) for s in sorted_steps if "val_acc" in steps[s]]
    peak_step, peak_val = (None, None)
    if val_records:
        peak_step, peak_val = max(val_records, key=lambda x: x[1])
    final_step, final_val = val_records[-1] if val_records else (None, None)

    # "collapse onset": first step where resp_len < 20 AND val logged after
    collapse_step = None
    for s in sorted_steps:
        if steps[s].get("resp_len", 999) < 20:
            collapse_step = s
            break

    out = {
        "run": run_name,
        "ckpt_dir": ckpt_dir,
        "step1": {k: step1.get(k) for k, _ in METRICS if k in step1},
        "peak_step": peak_step,
        "peak_val": peak_val,
        "final_step": final_step,
        "final_val": final_val,
        "collapse_step": collapse_step,
        "n_steps": len(sorted_steps),
    }
    # include ppo_kl / grad_norm at final step too (for the "grad=0 late" claim)
    if sorted_steps:
        last = steps[sorted_steps[-1]]
        out["final_ppo_kl"] = last.get("ppo_kl")
        out["final_grad_norm"] = last.get("grad_norm")
        out["final_resp_len"] = last.get("resp_len")
        out["final_entropy"] = last.get("entropy")
    return out


def fmt(v, d=3):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{d}f}"
    return str(v)


def main():
    results_3b = [summarise(n, d) for n, d in RUNS_3B]
    results_7b = [summarise(n, d) for n, d in RUNS_7B]

    print("=" * 90)
    print("STEP-1 DIAGNOSTICS")
    print("=" * 90)

    for label, results in [("3B runs", results_3b), ("7B runs", results_7b)]:
        print(f"\n--- {label} ---")
        hdr = f"{'run':<22} {'ppo_kl':>8} {'kl_loss':>8} {'entropy':>8} {'grad':>7} {'resp':>6} {'clipf':>6}"
        print(hdr)
        print("-" * len(hdr))
        for r in results:
            if "error" in r:
                print(f"{r['run']:<22}  [{r['error']}]")
                continue
            s = r["step1"]
            print(f"{r['run']:<22} {fmt(s.get('ppo_kl')):>8} {fmt(s.get('kl_loss')):>8} "
                  f"{fmt(s.get('entropy')):>8} {fmt(s.get('grad_norm'),2):>7} "
                  f"{fmt(s.get('resp_len'),1):>6} {fmt(s.get('clipfrac'),3):>6}")

    print("\n" + "=" * 90)
    print("TRAJECTORY SHAPE")
    print("=" * 90)
    for label, results in [("3B runs", results_3b), ("7B runs", results_7b)]:
        print(f"\n--- {label} ---")
        hdr = (f"{'run':<22} {'peak':>12} {'collapse_at':>12} "
               f"{'final':>12} {'final_kl':>9} {'final_grad':>10}")
        print(hdr)
        print("-" * len(hdr))
        for r in results:
            if "error" in r:
                continue
            peak = (f"{r['peak_val']*100:.1f}%@{r['peak_step']}"
                    if r['peak_val'] is not None else "—")
            final = (f"{r['final_val']*100:.1f}%@{r['final_step']}"
                     if r['final_val'] is not None else "—")
            collapse = f"step {r['collapse_step']}" if r['collapse_step'] else "never"
            print(f"{r['run']:<22} {peak:>12} {collapse:>12} {final:>12} "
                  f"{fmt(r.get('final_ppo_kl')):>9} {fmt(r.get('final_grad_norm'),2):>10}")

    # dump JSON for paper figure consumption
    out_json = ROOT / "checkpoints/_eval/first_update_diagnostics.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(
        {"3B": results_3b, "7B": results_7b}, indent=2))
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
