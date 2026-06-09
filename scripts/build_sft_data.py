#!/usr/bin/env python3
"""
SFT   GTPO   RL   rollout jsonl  
  verl MultiTurnSFTDataset   parquet  

 messages (list[dict]), score, num_turns, answer, source_step

 
  python build_sft_data.py \
    --rollout_dir  ../checkpoints/gtpo_3b_deepsearch_v3/rollout \
    --output_dir   ../rl_datasets/sft \
    --score_thresh 0.3 \
    --dedup         \
    --val_ratio    0.05
"""

import argparse
import glob
import json
import os
import re
import random

import pandas as pd


# ──   rollout input   messages ─────────────────────────────────────

_ROLE_SPLIT_RE = re.compile(r"\n(?=system\n|user\n|assistant\n|tool\n)")


def parse_input_to_messages(input_str: str) -> list[dict]:
    """  Qwen chat-template   input   messages  

     
        system\\nYou are...\\nuser\\nWhat is 2+2?\\nassistant\\n

     
        [{"role": "system", "content": "You are..."},
         {"role": "user",   "content": "What is 2+2?"}]
      assistant  
    """
    segments = _ROLE_SPLIT_RE.split(input_str)
    messages = []
    for seg in segments:
        if "\n" not in seg:
            continue
        role, content = seg.split("\n", 1)
        role = role.strip()
        content = content.strip()
        if role in ("system", "user", "assistant", "tool") and content:
            messages.append({"role": role, "content": content})
    return messages


# ──   ────────────────────────────────────────────────────────────────

def load_positives(rollout_dir: str, score_thresh: float) -> list[dict]:
    """  step jsonl  score >= score_thresh  """
    pattern = os.path.join(rollout_dir, "*.jsonl")
    files = sorted(glob.glob(pattern), key=lambda p: int(os.path.basename(p).replace(".jsonl", "")))

    records = []
    for fp in files:
        step = int(os.path.basename(fp).replace(".jsonl", ""))
        with open(fp) as f:
            for line in f:
                obj = json.loads(line)
                score = float(obj.get("score", 0.0))
                if score < score_thresh:
                    continue
                messages = parse_input_to_messages(obj["input"])
                if not messages:
                    continue
                #   assistant  
                output = obj.get("output", "").strip()
                if not output:
                    continue
                messages.append({"role": "assistant", "content": output})
                records.append(
                    {
                        "messages": messages,
                        "score": score,
                        "num_turns": obj.get("num_turns", 1),
                        "answer": obj.get("answer", ""),
                        "source_step": step,
                    }
                )
    return records


# ──   user  ──────────────────────────────────────────────────

def dedup_by_question(records: list[dict]) -> list[dict]:
    """  score  """
    best: dict[str, dict] = {}
    for r in records:
        #   user   content   key
        key = ""
        for m in r["messages"]:
            if m["role"] == "user":
                key = m["content"]
                break
        if key not in best or r["score"] > best[key]["score"]:
            best[key] = r
    return list(best.values())


# ──   ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout_dir", default="../checkpoints/gtpo_3b_deepsearch_v3/rollout",
                        help="  step jsonl   rollout  ")
    parser.add_argument("--output_dir", default="../rl_datasets/sft",
                        help="  parquet  ")
    parser.add_argument("--score_thresh", type=float, default=0.3,
                        help="  0.3 0.5 ")
    parser.add_argument("--dedup", action="store_true",
                        help=" ")
    parser.add_argument("--val_ratio", type=float, default=0.05,
                        help=" 0= ")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # 1.  
    print(f"[1/4]   rollout  {args.rollout_dir}")
    records = load_positives(args.rollout_dir, args.score_thresh)
    print(f"       {len(records)} (score >= {args.score_thresh})")

    # 2.  
    if args.dedup:
        records = dedup_by_question(records)
        print(f"[2/4]  {len(records)}   user  ")
    else:
        print(f"[2/4]  ")

    if len(records) == 0:
        print("❌   --score_thresh   rollout_dir ")
        return

    # 3.   train / val
    random.shuffle(records)
    if args.val_ratio > 0:
        n_val = max(1, int(len(records) * args.val_ratio))
        val_records = records[:n_val]
        train_records = records[n_val:]
    else:
        train_records = records
        val_records = []
    print(f"[3/4] train={len(train_records)}, val={len(val_records)}")

    # 4.   parquet
    os.makedirs(args.output_dir, exist_ok=True)

    def save(recs, name):
        if not recs:
            return
        df = pd.DataFrame(recs)
        path = os.path.join(args.output_dir, f"{name}.parquet")
        df.to_parquet(path, index=False)
        print(f"        {name}.parquet → {path}  ({len(df)}  )")
        print(f"      score  : mean={df['score'].mean():.3f}  "
              f"min={df['score'].min():.3f}  max={df['score'].max():.3f}")

    print(f"[4/4]   parquet  ")
    save(train_records, "train")
    save(val_records, "val")


    ex = train_records[0]
    print("\n──   ──────────────────────────────────────────────────────────")
    for m in ex["messages"]:
        prefix = f"[{m['role']:9s}]"
        body = m["content"][:120].replace("\n", " ")
        print(f"  {prefix} {body}")
    print(f"  score={ex['score']:.3f}  num_turns={ex['num_turns']}  step={ex['source_step']}")
    print("──────────────────────────────────────────────────────────────────")
    print("✅  ")


if __name__ == "__main__":
    main()
