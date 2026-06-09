#!/usr/bin/env python3
"""
Convert AIME 2024/2025 jsonl files into verl-compatible parquet datasets,
using the same schema as MATH-500 (columns: data_source, prompt, ability,
reward_model, extra_info) so they can be dropped directly into
`data.val_files=...` for a verl pass@k evaluation.

Inputs  : evaluation/data/aime{24,25}/test.jsonl
Outputs : ARPO/rl_datasets/aime/aime24_test.parquet  (30 items)
          ARPO/rl_datasets/aime/aime25_test.parquet  (30 items)
          ARPO/rl_datasets/aime/aime_combined_test.parquet  (60 items)
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

ROOT = Path("")
OUT_DIR = ROOT / "ARPO/rl_datasets/aime"
OUT_DIR.mkdir(parents=True, exist_ok=True)

AIME_PROMPT_TEMPLATE = (
    "Solve the following math problem step-by-step.  Put your final answer "
    "inside \\boxed{{}}.\n\nProblem: {problem}\n\nSolution: "
)

def convert(jsonl_path: Path, year: int) -> list[dict]:
    rows = []
    for idx, line in enumerate(jsonl_path.read_text().splitlines()):
        if not line.strip():
            continue
        item = json.loads(line)
        problem = item["problem"]
        ans = str(item["answer"]).strip()
        user_content = AIME_PROMPT_TEMPLATE.format(problem=problem)
        rows.append({
            "data_source": f"aime-{year}",
            "prompt": [{"role": "user", "content": user_content}],
            "ability": "math",
            "reward_model": {"ground_truth": ans, "style": "rule"},
            "extra_info": {
                "index": str(item.get("id", idx)),
                "split": "test",
                "subject": "AIME",
                "unique_id": f"aime{year}/problem/{item.get('id', idx)}",
                "year": str(year),
            },
        })
    return rows


def main():
    combined = []
    for year, jsonl in [
        (2024, ROOT / "evaluation/data/aime24/test.jsonl"),
        (2025, ROOT / "evaluation/data/aime25/test.jsonl"),
    ]:
        rows = convert(jsonl, year)
        out = OUT_DIR / f"aime{year-2000:02d}_test.parquet"
        pd.DataFrame(rows).to_parquet(out, index=False)
        combined.extend(rows)
        print(f"wrote {out}  ({len(rows)} items)")

    out = OUT_DIR / "aime_combined_test.parquet"
    pd.DataFrame(combined).to_parquet(out, index=False)
    print(f"wrote {out}  ({len(combined)} items)")

    # Print a sanity sample
    df = pd.read_parquet(out)
    print("\n=== sample row ===")
    row = df.iloc[0]
    for k in df.columns:
        s = str(row[k])
        print(f"  {k}: {s[:150]}")


if __name__ == "__main__":
    main()
