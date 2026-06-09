#!/usr/bin/env python3
"""Download HuggingFaceH4/MATH-500 and convert to verl-compatible parquet.

Output schema matches rl_datasets/gsm8k/test.parquet so we can feed it to
verl's val-only pipeline without any code change.

  data_source : 'DigitalLearningGmbH/MATH-lighteval'   -> routes to
                  verl.utils.reward_score.math.compute_score, which matches
                  the pred's \\boxed{...} against ground_truth.
  prompt      : [{'role': 'user', 'content': problem + instr}]
  reward_model: {'style': 'rule', 'ground_truth': <answer>}

Usage:
  cd ${ARPO_ROOT}
  python rl_datasets/download_math500.py
  # -> rl_datasets/math500/test.parquet (500 rows)
"""
from __future__ import annotations

import argparse
import os

import datasets


INSTRUCTION = "Let's think step by step and output the final answer within \\boxed{}."
DATA_SOURCE = "DigitalLearningGmbH/MATH-lighteval"


def build_row(example: dict, idx: int) -> dict:
    problem = example["problem"]
    answer = example["answer"]
    return {
        "data_source": DATA_SOURCE,
        "prompt": [{"role": "user", "content": f"{problem} {INSTRUCTION}"}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": str(answer)},
        "extra_info": {
            "split": "test",
            "index": idx,
            "subject": example.get("subject", ""),
            "level": example.get("level", -1),
            "unique_id": example.get("unique_id", ""),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        default=os.path.join(os.path.dirname(__file__), "math500"),
        help="output directory; a test.parquet will be written inside",
    )
    parser.add_argument(
        "--hf_repo",
        default="HuggingFaceH4/MATH-500",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading {args.hf_repo} ...", flush=True)
    ds = datasets.load_dataset(args.hf_repo, split="test")
    print(f"  got {len(ds)} rows; first keys: {list(ds.features.keys())}")

    ds = ds.map(
        build_row,
        with_indices=True,
        remove_columns=ds.column_names,
    )

    out_path = os.path.join(args.out_dir, "test.parquet")
    ds.to_parquet(out_path)
    print(f"wrote {out_path}   ({len(ds)} rows)")


if __name__ == "__main__":
    main()
