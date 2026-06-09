# JanusPO

**Asymmetric Clip and KL for Entropy-Stable Group Policy Optimization in Tool-Augmented LLM Reasoning**

JanusPO is a reinforcement-learning algorithm built on top of Group Relative Policy Optimization (GRPO) for training tool-augmented LLM agents. It introduces three intertwined modifications that together yield stable, entropy-controlled training in both single-turn and multi-turn settings:

1. **Asymmetric Clipping** — separate `ε⁺` / `ε⁻` ranges for positive- and negative-advantage tokens.
2. **Asymmetric KL Penalty** — `β_KL` switches between low (positive advantage, encourage exploitation) and high (negative advantage, suppress drift) values.
3. **Turn-Weighted Reward Decomposition `w_t(n)`** — distributes the trajectory-level reward `R_final` across the last token of each assistant turn so that mid-trajectory turns receive non-zero credit.

Together with a small entropy bonus and an optional entropy upper bound, JanusPO improves GSM8K validation accuracy from 74.8 % (clean GRPO) to **84.5 %** on Qwen2.5-3B-Instruct (+9.7 pp) without entropy collapse, and remains stable for >200 training steps on a single 8×GH200 node.

---

## Repository layout

```
JanusPO/
├── verl_arpo_entropy/         # Modified verl framework (slimmed: no docs/examples/tests)
│   ├── verl/
│   │   ├── workers/
│   │   │   ├── actor/dp_actor.py            # Asymmetric clip / KL / entropy regularizer
│   │   │   └── reward_manager/gtpo.py       # Turn-weighted reward manager (w_t(n))
│   │   ├── trainer/ppo/
│   │   │   ├── core_algos.py                # GSPO ratio + clip_ratio_pos/neg
│   │   │   └── reward.py                    # 'gtpo' reward-manager registration
│   │   └── utils/reward_score/
│   │       ├── gsm8k.py                     # Robust ####-based answer extraction
│   │       └── deep_research.py             # compute_score / compute_score_permissive
│   ├── recipe/, scripts/, docker/, setup.py # verl auxiliary code, retained
│   └── requirements*.txt
├── scripts/                   # All JanusPO experiment / SFT / eval shell scripts
│   ├── ARPO_3B_GSM8K_grpo_clean.sh                   # baseline (74.8 %)
│   ├── ARPO_3B_GSM8K_gspo_asymclip.sh                # phase 1 (asymmetric clip)
│   ├── ARPO_3B_GSM8K_phase3.sh                       # phase 3 (w_t(n))
│   ├── ARPO_3B_GSM8K_plan_c.sh                       # entropy regularizer
│   ├── ARPO_3B_GSM8K_plan_c_asymkl.sh                # + asymmetric β_KL
│   ├── ARPO_3B_GSM8K_plan_c_asymkl_v2.sh             # ★ best 3B run (84.5 %)
│   ├── ARPO_7B_GSM8K_plan_c_asymkl_v2.sh             # 7B generalization
│   ├── ARPO_3B_DeepSearch_GTPO.sh                    # multi-turn DeepSearch RL
│   ├── ARPO_3B_DeepSearch_SFT.sh                     # SFT warm-up
│   ├── build_sft_data.py / prepare_sft_mix.py        # SFT data prep
│   ├── test_gtpo_phase{1,2,3}.py                     # Unit tests (40 cases)
│   └── config/ppo_trainer{,_dr}.yaml                 # PPO training configs
├── merge_ckpt/                # verl → HuggingFace ckpt converter
├── rl_datasets/               # Compact RL parquet datasets + download scripts
│   ├── train_10k.parquet, valid.parquet, hard_search_1k.parquet
│   ├── gaia_test.parquet, hle_test.parquet
│   └── download_*.sh / download_*.py
├── search_cache/              # Cached search-API responses for reproducibility
└── GTPO_progress.md           # Internal experiment log (full ablation history)
```

---

## Key algorithmic ingredients (one-line each)

| Component | Implementation | Effect |
|---|---|---|
| Asymmetric clip `ε⁺ ≠ ε⁻` | `verl/trainer/ppo/core_algos.py` (`clip_ratio_pos / clip_ratio_neg`) | Allows aggressive exploitation while restricting drift |
| Asymmetric KL `β_low / β_high` | `verl/workers/actor/dp_actor.py` (`kl_beta_pos / kl_beta_neg`) | Doubles as a free entropy regulator (β_neg = 1.5 ⇒ entropy stays in 1.0–2.3) |
| Entropy bonus `λ·H` | `dp_actor.py` (`entropy_coeff`) | Rescues mode-collapse around step 50 |
| Entropy upper bound | `dp_actor.py` (`entropy_upper_bound`) | Optional ceiling; rarely triggered when AsymKL is on |
| Turn-weighted reward `w_t(n)` | `verl/workers/reward_manager/gtpo.py` | Distributes `R_final` across assistant turns |

---

## Quick start

### 1. Install (single-node)

```bash
cd verl_arpo_entropy
pip install -e .
pip install -r requirements.txt
```

### 2. Reproduce the 3B GSM8K best result (84.5 %)

```bash
cd scripts
bash ARPO_3B_GSM8K_plan_c_asymkl_v2.sh
```

Trains Qwen2.5-3B-Instruct for 200+ steps on 8×GH200 (≈ 14 hours). Peak validation `84.5 %` is reached around `global_step_200`.

### 3. Run unit tests (40 cases, all should pass)

```bash
cd scripts
python test_gtpo_phase1.py     # 13 cases (asymmetric clip)
python test_gtpo_phase2.py     # 9 cases (asymmetric β_KL)
python test_gtpo_phase3.py     # 18 cases (turn-weighted reward)
```

### 4. Multi-turn DeepSearch (requires SFT warm-up first)

```bash
# Step 1: SFT warm-up on sft_mix (≈ 1.8 h on 1 GPU)
bash scripts/ARPO_3B_DeepSearch_SFT.sh
# Step 2: GTPO RL on top of SFT checkpoint
bash scripts/ARPO_3B_DeepSearch_GTPO.sh
```

---

## Headline results (Qwen2.5-3B-Instruct on GSM8K)

| Configuration | Peak val | Δ vs baseline |
|---|---|---|
| Clean GRPO baseline | 74.8 % | — |
| + Asymmetric clip | 81.1 % | +6.3 pp |
| + Entropy bonus (Plan C) | 84.0 % | +9.2 pp |
| + Asymmetric β_KL (Plan C + AsymKL) | 84.4 % | +9.6 pp |
| **+ Entropy upper-bound 2.5 (final)** | **84.5 %** | **+9.7 pp** |

Multi-seed (n = 2) standard deviation: **± 0.2 pp**.

7B Qwen2.5-7B-Instruct reaches **92.9 %** validation in only 40 steps under the same recipe (5× faster convergence than 3B).

Full ablation (14 runs, 7 phases) is in `GTPO_progress.md`.

---

## Citation

If you use this code, please cite:

```bibtex
@article{janus2026,
  title  = {{JanusPO}: Asymmetric Clip and {KL} for Entropy-Stable Group Policy Optimization in Tool-Augmented {LLM} Reasoning},
  author = {Anonymous},
  year   = {2026}
}
```

---

## Acknowledgements

- Built on top of **verl** (Volcengine Reinforcement Learning) and the public **ARPO** code release.
- DeepSearch SFT data partially derived from **ThornZ Search-R1-SFT**.
- The `verl_arpo_entropy/` directory inherits the original verl Apache-2.0 license; modifications are clearly localized in the files listed above.

---

## License

Apache 2.0 (inherited from upstream verl). See `verl_arpo_entropy/LICENSE`.
