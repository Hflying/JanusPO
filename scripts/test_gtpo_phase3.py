"""
GTPO Phase-3  w_t(n)  
 
    conda activate arpo
    cd 
    python3 scripts/test_gtpo_phase3.py
"""

import math
import sys
import os

import torch

# ──   verl   PYTHONPATH ──────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "verl_arpo_entropy"))

from verl.workers.reward_manager.gtpo import _find_turn_end_positions, _compute_weights

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
results = []


def check(name, cond, detail=""):
    status = PASS if cond else FAIL
    results.append(cond)
    msg = f"  [{status}] {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)


# ════════════════════════════════════════════════════════════════════════════
# T1-T4: _find_turn_end_positions
# ════════════════════════════════════════════════════════════════════════════
print("\n── T1-T4  _find_turn_end_positions ────────────────────────────────")

# T1: single turn (all ones)
lm = torch.tensor([1, 1, 1, 1, 1], dtype=torch.long)
ends = _find_turn_end_positions(lm)
check("T1   →   4", ends == [4], f"got {ends}")

# T2: two turns separated by tool response (zeros in the middle)
lm = torch.tensor([1, 1, 0, 0, 1, 1, 1], dtype=torch.long)
ends = _find_turn_end_positions(lm)
check("T2   → ends=[1,6]", ends == [1, 6], f"got {ends}")

# T3: three turns
lm = torch.tensor([1, 0, 0, 1, 1, 0, 1], dtype=torch.long)
ends = _find_turn_end_positions(lm)
check("T3   → ends=[0,4,6]", ends == [0, 4, 6], f"got {ends}")

# T4: all zeros (no assistant turn → empty)
lm = torch.tensor([0, 0, 0], dtype=torch.long)
ends = _find_turn_end_positions(lm)
check("T4   → []", ends == [], f"got {ends}")

# ════════════════════════════════════════════════════════════════════════════
# T5-T9: _compute_weights
# ════════════════════════════════════════════════════════════════════════════
print("\n── T5-T9  _compute_weights ─────────────────────────────────────────")

# T5: uniform mode sums to 1
w = _compute_weights(4, "uniform")
check("T5 uniform  =4", len(w) == 4)
check("T5 uniform  ", all(abs(x - 0.25) < 1e-9 for x in w), f"got {w}")

# T6: exp_decay sums to 1 and later turns have higher weight
w = _compute_weights(4, "exp_decay", beta=0.9)
check("T6 exp_decay  ", abs(sum(w) - 1.0) < 1e-9, f"sum={sum(w):.6f}")
check("T6 exp_decay   ( )", w[-1] > w[0], f"w={[f'{x:.4f}' for x in w]}")

# T7: last_only puts all weight on final turn
w = _compute_weights(3, "last_only")
check("T7 last_only  ", w == [0.0, 0.0, 1.0], f"got {w}")

# T8: pos_sensitive sums to 1 and peaks near middle
w = _compute_weights(5, "pos_sensitive", alpha=0.5)
check("T8 pos_sensitive  ", abs(sum(w) - 1.0) < 1e-9, f"sum={sum(w):.6f}")
check("T8 pos_sensitive   (idx=2)", w[2] == max(w), f"w={[f'{x:.4f}' for x in w]}")

# T9: single turn always returns [1.0]
for mode in ("exp_decay", "uniform", "pos_sensitive", "last_only"):
    w = _compute_weights(1, mode)
    check(f"T9   mode={mode} → [1.0]", w == [1.0], f"got {w}")

# ════════════════════════════════════════════════════════════════════════════
# T10-T12: end-to-end reward distribution via GTPoRewardManager
# ════════════════════════════════════════════════════════════════════════════
print("\n── T10-T12  GTPoRewardManager   ──────────────────────")

from unittest.mock import MagicMock
from verl.workers.reward_manager.gtpo import GTPoRewardManager

def _make_mock_data(loss_mask_1d, r_final=1.0, response_len=None):
    """Build a minimal DataProto-like mock for one sample."""
    if response_len is None:
        response_len = len(loss_mask_1d)
    prompt_len = 4
    total_len = prompt_len + response_len

    batch = {
        "prompts":        torch.zeros(prompt_len, dtype=torch.long),
        "responses":      torch.zeros(response_len, dtype=torch.long),
        "attention_mask": torch.ones(total_len, dtype=torch.long),
        "loss_mask":      torch.cat([torch.zeros(prompt_len, dtype=torch.long),
                                     torch.tensor(loss_mask_1d, dtype=torch.long)]),
    }

    non_tensor_batch = {
        "reward_model": {"ground_truth": "42"},
        "data_source":  "gsm8k",
    }

    mock_item = MagicMock()
    mock_item.batch = batch
    mock_item.non_tensor_batch = non_tensor_batch
    return mock_item


def _make_dataset_mock(items, r_final=1.0):
    """Wrap items into a DataProto-like mock."""
    mock_ds = MagicMock()
    mock_ds.__len__.return_value = len(items)
    mock_ds.__iter__.return_value = iter(items)
    mock_ds.batch = {
        "responses": torch.zeros(len(items), items[0].batch["responses"].shape[0], dtype=torch.long),
    }
    mock_ds.__getitem__ = lambda self, i: items[i]
    return mock_ds


def _make_manager(mode):
    tokenizer = MagicMock()
    tokenizer.decode.return_value = "mock response"
    mgr = GTPoRewardManager(
        tokenizer=tokenizer,
        num_examine=0,
        compute_score=lambda **kwargs: r_final_value,
        w_t_mode=mode,
    )
    return mgr


r_final_value = 1.0

# T10: single-turn → reward at last valid token, weight=1.0
loss_mask = [1, 1, 1]
item = _make_mock_data(loss_mask, r_final=1.0)
ds = _make_dataset_mock([item])
mgr = _make_manager("uniform")
reward_tensor = mgr(ds)
check("T10   reward_tensor[0, 2] == 1.0", abs(reward_tensor[0, 2].item() - 1.0) < 1e-6,
      f"tensor={reward_tensor[0].tolist()}")

# T11: two-turn uniform → reward split 0.5 at end of each turn
loss_mask = [1, 1, 0, 0, 1, 1]
item = _make_mock_data(loss_mask, r_final=1.0)
ds = _make_dataset_mock([item])
mgr = _make_manager("uniform")
reward_tensor = mgr(ds)
check("T11   uniform → ends=[1,5],   0.5",
      abs(reward_tensor[0, 1].item() - 0.5) < 1e-6 and abs(reward_tensor[0, 5].item() - 0.5) < 1e-6,
      f"tensor={reward_tensor[0].tolist()}")

# T12: two-turn last_only → all reward at final turn end
mgr2 = _make_manager("last_only")
reward_tensor2 = mgr2(ds)
check("T12   last_only →   pos 5",
      abs(reward_tensor2[0, 1].item() - 0.0) < 1e-6 and abs(reward_tensor2[0, 5].item() - 1.0) < 1e-6,
      f"tensor={reward_tensor2[0].tolist()}")

# ════════════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════════════
n_pass = sum(results)
n_total = len(results)
print(f"\n{'='*60}")
print(f"   : {n_pass}/{n_total}  ")
if n_pass == n_total:
    print("  ✅  Phase 3 w_t(n)  ")
else:
    print(f"  ❌ {n_total - n_pass}  ")
print(f"{'='*60}\n")
sys.exit(0 if n_pass == n_total else 1)
