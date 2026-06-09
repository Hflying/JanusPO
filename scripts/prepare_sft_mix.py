#!/usr/bin/env python3
"""
SFT   ThornZ/Search-R1-SFT   rollout  

 
  ThornZ  
    system  → "You are a helpful and harmless assistant."
    user    → "...instruction... Question: {question}"
    assistant → "<thinking>...</thinking>\n<search>query</search>"
    user    → "<information>[results]</information>"
    assistant → "<thinking>...</thinking>\n<answer>answer</answer>"

  ↓   assistant  
    system    → ARPO   <think>/<search>/<result>/<answer>  
    user      → {question} 
    assistant → <think>...</think><search>query</search><result>[results]</result>
                <think>...</think><answer>answer</answer>

  GTPO rollout  SFT   RL  

 
  python prepare_sft_mix.py \
    [--thornz_path  /path/to/qwen2.5-7b-instruct-sft.json] \
    [--rollout_sft  ../rl_datasets/sft/train.parquet]       \
    [--output_dir   ../rl_datasets/sft_mix]                 \
    [--max_external 3000]       #   ThornZ  
    [--seed 42]

  #   ThornZ 
  python prepare_sft_mix.py --download
"""

import argparse
import json
import os
import random
import re

import pandas as pd


# ── ARPO DeepSearch   GTPO rollout  ──────────────────────
ARPO_SYSTEM_PROMPT = (
    "You are a helpful assistant that can solve the given question step by step "
    "with the help of the wikipedia search tool and python interpreter tool. "
    "Given a question, you need to first think about the reasoning process in the "
    "mind and then provide the answer. During thinking, you can invoke the wikipedia "
    "search tool to search and python interpreter tool to calculate the math problem "
    "for fact information about specific topics if needed. "
    "The reasoning process and answer are enclosed within <think> </think> and "
    "<answer> </answer> tags respectively, and the search query and result are "
    "enclosed within <search> </search> and <result> </result> tags respectively. "
    "For example, <think> This is the reasoning process. </think> "
    "<search> search query here </search> "
    "<result> search result here </result> "
    "<think> This is the reasoning process. </think> "
    "<answer> The final answer is \\[ \\boxed{answer here} \\] </answer>. "
    "In the last part of the answer, the final exact answer is enclosed within "
    "\\boxed{} with latex format."
)

# ── ThornZ   ───────────────────────────────────────────────────────
THORNZ_CACHE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "rl_datasets", "hf_cache"
)


# ──   ThornZ   ──────────────────────────────────────────────────────────

def download_thornz(cache_dir: str, variant: str = "qwen2.5-7b-instruct-sft.json") -> str:
    """  ThornZ/Search-R1-SFT  """
    from huggingface_hub import hf_hub_download
    os.makedirs(cache_dir, exist_ok=True)
    print(f"[ ] ThornZ/Search-R1-SFT ({variant})  → {cache_dir}")
    path = hf_hub_download(
        repo_id="ThornZ/Search-R1-SFT",
        filename=variant,
        repo_type="dataset",
        cache_dir=cache_dir,
    )
    print(f"       ✅  {path}")
    return path


# ── ThornZ   ───────────────────────────────────────────────────────────

_QUESTION_RE = re.compile(r"Question:\s*(.+)$", re.DOTALL)


def _tag(s: str) -> str:
    """ thinking→think, information→result """
    s = s.replace("<thinking>", "<think>").replace("</thinking>", "</think>")
    s = s.replace("<information>", "<result>").replace("</information>", "</result>")
    return s


def convert_thornz_row(messages: list[dict]) -> dict | None:
    """
      ThornZ   ARPO  

      {'messages': [...]}   None 
    """
    if len(messages) < 3:
        return None  #   system / user / assistant

    # 1.   user   "Question: ..."  
    user_raw = messages[1]["content"]
    m = _QUESTION_RE.search(user_raw)
    if not m:
        # fallback  user  
        question = user_raw.strip()
    else:
        question = m.group(1).strip()
    if not question:
        return None

    # 2.   assistant  inline  
    merged_parts = []
    i = 2
    while i < len(messages):
        msg = messages[i]
        if msg["role"] == "assistant":
            merged_parts.append(_tag(msg["content"]))
        elif msg["role"] == "user":
            content = msg["content"].strip()
            if content:
                # information   → <result> </result>  <result>  
                if "<result>" not in content and "<information>" not in content:
                    content = f"<result>\n{content}\n</result>"
                merged_parts.append(_tag(content))
        i += 1

    if not merged_parts:
        return None

    #   <answer>
    full_response = " ".join(merged_parts)
    if "<answer>" not in full_response:
        return None

    return {
        "messages": [
            {"role": "system",    "content": ARPO_SYSTEM_PROMPT},
            {"role": "user",      "content": question},
            {"role": "assistant", "content": full_response},
        ]
    }


# ──   ThornZ   ────────────────────────────────────────────────────────

def load_thornz(json_path: str, max_samples: int, seed: int) -> list[dict]:
    print(f"[ThornZ]   {json_path}  ")
    with open(json_path) as f:
        raw = json.load(f)

    #   max_samples * 2  
    rng = random.Random(seed)
    rng.shuffle(raw)
    raw = raw[: max_samples * 2]

    converted, skipped = [], 0
    for row in raw:
        result = convert_thornz_row(row["messages"])
        if result is None:
            skipped += 1
        else:
            converted.append(result)
        if len(converted) >= max_samples:
            break

    print(f"          {len(converted)} {skipped}")
    return converted


# ──   rollout   ───────────────────────────────────────────────────

def load_rollout_sft(parquet_path: str) -> list[dict]:
    if not os.path.isfile(parquet_path):
        print(f"⚠️    rollout SFT  {parquet_path} ")
        return []
    df = pd.read_parquet(parquet_path)
    records = df.to_dict("records")
    #   messages  
    out = [{"messages": r["messages"]} for r in records]
    print(f"[Rollout]   {len(out)}   {parquet_path} ")
    return out


# ──   ────────────────────────────────────────────────────────────────────

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--thornz_path",
        default=None,
        help="ThornZ json  ",
    )
    parser.add_argument(
        "--thornz_variant",
        default="qwen2.5-7b-instruct-sft.json",
        choices=["qwen2.5-7b-instruct-sft.json", "qwen3-4b-instruct-sft.json"],
        help="ThornZ  ",
    )
    parser.add_argument(
        "--rollout_sft",
        default=os.path.join(parent_dir, "rl_datasets", "sft", "train.parquet"),
        help="rollout   SFT parquet build_sft_data.py  ",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(parent_dir, "rl_datasets", "sft_mix"),
        help=" ",
    )
    parser.add_argument(
        "--max_external", type=int, default=3000,
        help="  ThornZ  ",
    )
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--download", action="store_true",
        help=" ",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # 1.   ThornZ json  
    thornz_path = args.thornz_path
    if thornz_path is None or args.download:

        hf_cache = os.path.join(parent_dir, "rl_datasets", "hf_cache")
        found = []
        for root, dirs, files in os.walk(hf_cache):
            for fn in files:
                if fn == args.thornz_variant:
                    found.append(os.path.join(root, fn))
        if found and not args.download:
            thornz_path = found[0]
            print(f"[ ]  {thornz_path}")
        else:
            try:
                thornz_path = download_thornz(hf_cache, args.thornz_variant)
            except Exception as e:
                print(f"❌  {e}")
                print("     --thornz_path  ")
                return

    # 2.   &   ThornZ
    thornz_records = load_thornz(thornz_path, args.max_external, args.seed)

    # 3.   rollout  
    rollout_records = load_rollout_sft(args.rollout_sft)

    # 4.  
    all_records = thornz_records + rollout_records
    random.shuffle(all_records)
    print(f"\n {len(all_records)}"
          f"   ThornZ={len(thornz_records)}, rollout={len(rollout_records)} ")

    # 5.   train / val
    n_val = max(1, int(len(all_records) * args.val_ratio))
    val_records  = all_records[:n_val]
    train_records = all_records[n_val:]
    print(f"train={len(train_records)}, val={len(n_val if False else val_records)}")

    # 6.   parquet
    def save(recs, name):
        if not recs:
            return
        df = pd.DataFrame(recs)
        path = os.path.join(args.output_dir, f"{name}.parquet")
        df.to_parquet(path, index=False)
        print(f"✅   {name}.parquet → {path}  ({len(df)}  )")

    save(train_records, "train")
    save(val_records,   "val")

    # 7.  
    ex = train_records[0]
    print("\n──  ────────────────────────────────────────────────────────")
    for m in ex["messages"]:
        body = m["content"][:200].replace("\n", " ")
        print(f"  [{m['role']:9s}] {body}")
    print("──────────────────────────────────────────────────────────────────────────")
    print("\n ")
    print(f"  export ARPO_SFT_DATA_DIR={args.output_dir}")
    print("  bash ARPO_3B_DeepSearch_SFT.sh")


if __name__ == "__main__":
    main()
