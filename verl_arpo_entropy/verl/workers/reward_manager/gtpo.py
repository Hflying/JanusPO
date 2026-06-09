# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
GTPO Phase-3: Turn-weighted reward distribution w_t(n).

Key idea: decompose final scalar reward R_final into per-turn rewards
    reward_at_turn_t = w_t(n) * R_final

Three weighting modes (controlled by w_t_mode):
  - 'exp_decay':        w_t(n) ∝ β^(n-t), later turns weighted more by default (β<1)
  - 'pos_sensitive':    Gaussian bell centred at the middle turn
  - 'uniform':          equal weight 1/n for every turn
  - 'last_only':        all weight on the final turn (= standard NaiveRewardManager)

Turn detection: uses loss_mask from multi-turn batches.
  - Consecutive 1-runs in loss_mask → each run is one assistant turn.
  - Falls back gracefully for single-turn (n=1): w_1(1)=1.
"""

import math
from collections import defaultdict
from typing import List, Optional

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score


def _find_turn_end_positions(loss_mask: torch.Tensor) -> List[int]:
    """Return indices of the last token in each assistant turn.

    An assistant turn is a maximal run of 1s in loss_mask.
    Example: [1,1,0,0,1,1,1,0] → turns end at positions 1 and 6.
    """
    ends = []
    n = loss_mask.shape[0]
    in_turn = False
    for i in range(n):
        if loss_mask[i].item() == 1:
            in_turn = True
        elif in_turn:
            ends.append(i - 1)
            in_turn = False
    if in_turn:
        ends.append(n - 1)
    return ends


def _compute_weights(num_turns: int, mode: str, beta: float = 0.9, alpha: float = 0.5) -> List[float]:
    """Compute w_t(n) weights for turns 1..n.

    Args:
        num_turns: total number of turns n
        mode:      weighting mode (see module docstring)
        beta:      decay factor for 'exp_decay' (default 0.9, <1 → later turns higher)
        alpha:     Gaussian width factor for 'pos_sensitive'

    Returns:
        List of length num_turns, summing to 1.0.
    """
    if num_turns == 0:
        return []
    if mode == "last_only" or num_turns == 1:
        return [1.0] * (num_turns - 1) + [1.0] if num_turns == 1 else [0.0] * (num_turns - 1) + [1.0]
    if mode == "uniform":
        return [1.0 / num_turns] * num_turns
    if mode == "exp_decay":
        # w_t ∝ beta^(n-t),  t=1..n  →  last turn has highest weight when beta<1
        raw = [beta ** (num_turns - t) for t in range(1, num_turns + 1)]
        total = sum(raw)
        return [w / total for w in raw]
    if mode == "pos_sensitive":
        # Gaussian centred at middle turn
        mid = (num_turns + 1) / 2.0
        raw = [math.exp(-alpha * (t - mid) ** 2) for t in range(1, num_turns + 1)]
        total = sum(raw)
        return [w / total for w in raw]
    raise ValueError(f"Unknown w_t_mode: {mode!r}. Choose from: exp_decay, pos_sensitive, uniform, last_only")


class GTPoRewardManager:
    """GTPO Phase-3 reward manager with turn-weighted reward distribution.

    Drop-in replacement for NaiveRewardManager that redistributes R_final
    across assistant turns according to w_t(n).

    Args:
        tokenizer:    HuggingFace tokenizer (used for decoding + score computation).
        num_examine:  Number of samples to print to console each call.
        compute_score: Custom scoring function; defaults to default_compute_score.
        reward_fn_key: Key in non_tensor_batch for data source selection.
        w_t_mode:     Turn weighting mode ('exp_decay', 'pos_sensitive',
                      'uniform', 'last_only').
        w_t_beta:     Decay factor β for 'exp_decay' mode (default 0.9).
        w_t_alpha:    Gaussian α for 'pos_sensitive' mode (default 0.5).
    """

    def __init__(
        self,
        tokenizer,
        num_examine: int,
        compute_score=None,
        reward_fn_key: str = "data_source",
        w_t_mode: str = "exp_decay",
        w_t_beta: float = 0.9,
        w_t_alpha: float = 0.5,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.w_t_mode = w_t_mode
        self.w_t_beta = w_t_beta
        self.w_t_alpha = w_t_alpha

    def __call__(self, data: DataProto, return_dict: bool = False):
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources: dict = {}

        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            attention_mask = data_item.batch["attention_mask"]

            valid_prompt_length = attention_mask[:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = attention_mask[prompt_length:].sum().item()
            valid_response_ids = response_ids[:valid_response_length]

            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", None)
            if extra_info is None or extra_info.get("tokenizer") is None:
                extra_info = {"tokenizer": self.tokenizer}

            result = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            if isinstance(result, dict):
                r_final = float(result["score"])
                for key, value in result.items():
                    reward_extra_info[key].append(value)
            else:
                r_final = float(result)

            # ── Turn-weighted reward distribution ─────────────────────────────
            # Detect turn ends from loss_mask when available; fall back to
            # single-turn (put reward at last valid response token).
            response_mask_full = attention_mask[prompt_length:]  # length = response_length
            valid_response_mask = response_mask_full[:valid_response_length]

            if "loss_mask" in data_item.batch:
                loss_mask_full = data_item.batch["loss_mask"]
                # loss_mask covers the full (prompt+response) or response only;
                # keep only the response part.
                if loss_mask_full.shape[0] == attention_mask.shape[0]:
                    loss_mask_resp = loss_mask_full[prompt_length:prompt_length + valid_response_length]
                else:
                    loss_mask_resp = loss_mask_full[:valid_response_length]
                turn_ends = _find_turn_end_positions(loss_mask_resp)
            else:
                # Single-turn fallback: reward at last valid token
                turn_ends = [valid_response_length - 1]

            num_turns = len(turn_ends)
            weights = _compute_weights(num_turns, self.w_t_mode, self.w_t_beta, self.w_t_alpha)

            for t_idx, (end_pos, wt) in enumerate(zip(turn_ends, weights)):
                reward_tensor[i, end_pos] = wt * r_final

            reward_extra_info["num_turns"].append(num_turns)
            reward_extra_info["w_t_mode"].append(self.w_t_mode)

            # ── Console logging ───────────────────────────────────────────────
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                print(f"[gtpo_reward] r_final={r_final:.4f}, num_turns={num_turns}, "
                      f"mode={self.w_t_mode}, weights={[f'{w:.3f}' for w in weights]}")

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        return reward_tensor
