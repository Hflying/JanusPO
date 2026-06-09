#!/usr/bin/env python3
"""  run.log   val-core/openai/gsm8k/reward/mean@1  global_step """
import re
import sys


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "run.log"
    best_v, best_s = -1.0, None
    pat = re.compile(
        r"step:(\d+).*val-core/openai/gsm8k/reward/mean@1:([\d.]+)"
    )
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = pat.search(line)
                if not m:
                    continue
                s, v = int(m.group(1)), float(m.group(2))
                if v > best_v:
                    best_v, best_s = v, s
    except OSError as e:
        print(f" : {path} ({e})", file=sys.stderr)
        sys.exit(1)
    if best_s is None:
        print(" val-core/openai/gsm8k/reward/mean@1 ")
        sys.exit(2)
    pct = 100.0 * best_v
    print(f"  val mean@1 = {best_v:.4f} ({pct:.2f}%)   global_step_{best_s}")
    print(f"  checkpoint: .../global_step_{best_s}/actor/")


if __name__ == "__main__":
    main()
