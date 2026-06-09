#!/usr/bin/env python3
"""
  ARPO GSM8K  

 :
  python3 scripts/diagnose_run.py checkpoints/< >/run.log [rollout_dir]

 :
  -  score entropy grad_norm response_len 
  -  
  -   vs   score   rollout_dir 
  -  
"""
import json
import pathlib
import re
import sys


# ───   run.log ─────────────────────────────────────────────────────────────

STEP_PAT = re.compile(
    r"step:(\d+) .*?"
    r"actor/entropy_loss:([\d.\-]+).*?"
    r"actor/kl_loss:([\d.\-]+).*?"
    r"actor/pg_loss:([\d.\-]+).*?"
    r"actor/pg_clipfrac:([\d.\-]+).*?"
    r"actor/grad_norm:([\d.\-]+).*?"
    r"(?:actor/lr:([\d.\-]+).*?)?"
    r"critic/score/mean:([\d.\-]+)"
)

# response_length   step number   pattern  
RESP_LEN_PAT = re.compile(r"training/global_step:([\d.]+).*?response_length/mean:([\d.\-]+)", re.DOTALL)

ROLLOUT_DIFF_PAT = re.compile(r"rollout_probs_diff_mean:([\d.\-]+)")

VAL_PAT = re.compile(
    r"step:(\d+).*?val-core/openai/gsm8k/reward/mean@1:([\d.]+)"
)


ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m|\[[\d]+m")


def parse_log(log_path: pathlib.Path):
    text = log_path.read_text(errors="ignore")
    text = ANSI_ESCAPE.sub("", text)   #   ANSI   tee  
    steps = {}
    for m in STEP_PAT.finditer(text):
        step = int(m.group(1))
        steps[step] = {
            "entropy": float(m.group(2)),
            "kl_loss": float(m.group(3)),
            "pg_loss": float(m.group(4)),
            "clipfrac": float(m.group(5)),
            "grad_norm": float(m.group(6)),
            "lr": float(m.group(7)) if m.group(7) else None,
            "score": float(m.group(8)),
            "resp_len": 0.0,  #   RESP_LEN_PAT  
        }
    # response_length  
    for m in RESP_LEN_PAT.finditer(text):
        step = int(float(m.group(1)))
        if step in steps:
            steps[step]["resp_len"] = float(m.group(2))
    # attach rollout_probs_diff per step (rough: take the value in same line block)
    for m_diff in ROLLOUT_DIFF_PAT.finditer(text):
        pass  # just check if it exists

    val_steps = {}
    for m in VAL_PAT.finditer(text):
        val_steps[int(m.group(1))] = float(m.group(2))

    return steps, val_steps


# ───   rollout dir  ────────────────────────────────────────

def load_gt_map(data_dir: pathlib.Path):
    try:
        import pandas as pd
    except ImportError:
        return {}
    gt_map = {}
    for fname in ["train.parquet", "test.parquet"]:
        p = data_dir / fname
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        for _, row in df.iterrows():
            content = row["prompt"][0]["content"]
            gt = row["reward_model"]["ground_truth"]
            q = re.split("Let's think", content)[0].strip()[:120]
            gt_map[q] = gt
    return gt_map


def true_accuracy(rollout_jsonl: pathlib.Path, gt_map: dict):
    lines = rollout_jsonl.read_text().strip().split("\n")
    correct = wrong = no_fmt = no_gt = 0
    sc_sum = 0.0
    for raw in lines:
        d = json.loads(raw)
        inp = d.get("input", "")
        out = d.get("output", "")
        sc_sum += d.get("score", 0)
        # extract question key
        idx = inp.find("\nuser\n")
        q = (inp[idx + len("\nuser\n"):] if idx != -1 else inp)
        q = re.split("Let's think", q)[0].strip()[:120]
        # extract answer
        matches = re.findall(r"#### (\-?[0-9\.,]+)", out)
        pred = matches[-1].replace(",", "") if matches else None
        gt = gt_map.get(q)
        if gt is None:
            no_gt += 1
        elif pred is None:
            no_fmt += 1
        elif pred == gt:
            correct += 1
        else:
            wrong += 1
    total = len(lines) - no_gt
    true_pct = 100.0 * correct / total if total > 0 else 0.0
    score_pct = 100.0 * sc_sum / len(lines) if lines else 0.0
    return true_pct, score_pct, correct, total


# ───   ───────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print(" : python3 scripts/diagnose_run.py <run.log> [rollout_dir]")
        sys.exit(1)

    log_path = pathlib.Path(sys.argv[1])
    if not log_path.exists():
        print(f" :   {log_path}")
        sys.exit(1)

    rollout_dir = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else log_path.parent / "rollout"
    dataset_dir = log_path.parent.parent.parent / "rl_datasets" / "gsm8k"

    print(f"\n{'='*70}")
    print(f"  : {log_path}")
    print(f"{'='*70}\n")

    steps, val_steps = parse_log(log_path)
    if not steps:
        print("  run.log  ")
        sys.exit(1)

    sorted_steps = sorted(steps.keys())
    print(f" : {len(sorted_steps)}  ({sorted_steps[0]} → {sorted_steps[-1]})\n")

    # ──   ──────────────────────────────────────────────────────────────────
    print(f"{'Step':>5}  {'Score%':>7}  {'Entropy':>8}  {'KL':>7}  {'Grad':>7}  {'RespLen':>8}  {'Val%':>7}")
    print("-" * 62)

    collapse_step = None
    prev_score = None
    issues = []

    for step in sorted_steps:
        d = steps[step]
        score_pct = d["score"] * 100
        val_str = f"{val_steps[step]*100:.1f}%" if step in val_steps else "   -  "


        #  : clipfrac   vLLM(fp16) vs FSDP(bf16)  0.3-0.4 

        flags = ""
        if d["entropy"] < 0.5:
            flags += " [! entropy]"
        if d["grad_norm"] > 10:
            flags += " [! ]"
        if score_pct < 2 and (prev_score is not None and prev_score > 10) and collapse_step is None:
            collapse_step = step
            flags += " ◄  "

        print(
            f"{step:>5}  {score_pct:>6.1f}%  {d['entropy']:>8.3f}  "
            f"{d['kl_loss']:>7.3f}  {d['grad_norm']:>7.2f}  {d['resp_len']:>8.1f}  "
            f"{val_str}{flags}"
        )
        prev_score = score_pct

    # ──   ─────────────────────────────────────────────────────────
    print("\n")
    if rollout_dir.exists() and dataset_dir.exists():
        gt_map = load_gt_map(dataset_dir)
        if gt_map:
            print("──   vs   score  10   ──────────────────────")
            rollout_files = sorted(rollout_dir.glob("*.jsonl"), key=lambda p: int(p.stem))
            sample_files = rollout_files[::10]
            print(f"{'Step':>5}  {' %':>8}  {'Score%':>8}  {' ':>8}  correct/total")
            print("-" * 55)
            score_diff_max = 0
            for rf in sample_files:
                step = int(rf.stem)
                true_pct, score_pct, correct, total = true_accuracy(rf, gt_map)
                diff = true_pct - score_pct
                score_diff_max = max(score_diff_max, abs(diff))
                print(f"{step:>5}  {true_pct:>7.1f}%  {score_pct:>7.1f}%  {diff:>+7.1f}%  {correct}/{total}")
            print()
            if score_diff_max > 5:
                issues.append(f"⚠   true_accuracy vs score   = {score_diff_max:.1f}% ")
            else:
                print("✓   score   <5% ")

    # ──   ───────────────────────────────────────────────────────────────
    print("──   ──────────────────────────────────────────────────────────")

    first_d = steps[sorted_steps[0]]
    last_d = steps[sorted_steps[-1]]

    entropy_drop = first_d["entropy"] - last_d["entropy"]
    if entropy_drop > 1.0:
        issues.append(f"✗ Entropy  : {first_d['entropy']:.3f} → {last_d['entropy']:.3f}  {entropy_drop:.2f} ")

    max_grad = max(d["grad_norm"] for d in steps.values())
    if max_grad > 8:
        issues.append(f"✗  : max grad_norm = {max_grad:.2f} >8  ")

    first_clipfrac = first_d["clipfrac"]
    if first_clipfrac > 0.5:
        issues.append(f"✗   PPO clipfrac = {first_clipfrac:.3f} >0.5 rollout  ")

    if collapse_step:
        issues.append(f"✗   step {collapse_step}  0%")

    lr_vals = [d["lr"] for d in steps.values() if d["lr"] is not None]
    if lr_vals and all(v == 0.0 for v in lr_vals):
        issues.append("⚠ actor/lr   0.000 get_last_lr   scheduler.step   lr  ")

    max_score = max(d["score"] for d in steps.values()) * 100
    issues.append(f"ℹ   score = {max_score:.1f}%")

    if val_steps:
        best_val_step = max(val_steps, key=val_steps.get)
        best_val = val_steps[best_val_step] * 100
        issues.append(f"ℹ   = {best_val:.1f}% step {best_val_step} ")

    for issue in issues:
        print(f"  {issue}")

    # ──   ───────────────────────────────────────────────────────────────
    print("\n──   ─────────────────────────────────────────────────────────")
    if first_clipfrac > 0.5:
        print("""
  [ ] clipfrac  >0.5  rollout  
""")
    elif not collapse_step and val_steps and max(val_steps.values()) > 0.6:
        best_val_s = max(val_steps, key=val_steps.get)
        print(f"""
  ✓   {val_steps[best_val_s]*100:.1f}% (step {best_val_s}) 
    entropy  grad_norm <8  
    entropy < 0.8   score   kl_loss_coef   lr 
   : python3 scripts/gsm8k_log_best_val_step.py <run.log>   checkpoint 
""")
    elif collapse_step and collapse_step > 50:
        print(f"""
  [ ] step {collapse_step}   KL   LR  
  [ ]
    export ARPO_KL_LOSS_COEF=0.003
    export ARPO_ACTOR_LR=5e-7
    export ARPO_EXPERIMENT_NAME=stable_highkl
    bash scripts/ARPO_3B_GSM8K_1node.sh
""")
    else:
        print("    entropy   grad_norm ")

    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
