"""
GTPO Phase-2  
  β_KL  

 : python3 scripts/test_gtpo_phase2.py
"""
import sys
import os

_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "verl_arpo_entropy")
sys.path.insert(0, _root)

import torch
from verl.trainer.ppo.core_algos import kl_penalty, agg_loss

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
_all_passed = True


def check(name, cond, msg=""):
    global _all_passed
    tag = PASS if cond else FAIL
    print(f"  {tag}  {name}" + (f"  [{msg}]" if msg else ""))
    if not cond:
        _all_passed = False


def _apply_asym_kl(log_prob, ref_log_prob, advantages, response_mask,
                   kl_loss_coef, kl_beta_pos=None, kl_beta_neg=None,
                   loss_agg_mode="token-mean"):
    """  dp_actor.py   β_KL   kl_loss scalar """
    kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty="low_var_kl")
    if kl_beta_pos is not None and kl_beta_neg is not None:
        beta_t = torch.where(
            advantages > 0,
            advantages.new_full(advantages.shape, kl_beta_pos),
            advantages.new_full(advantages.shape, kl_beta_neg),
        )
        kl_loss = agg_loss(loss_mat=kld * beta_t, loss_mask=response_mask,
                           loss_agg_mode=loss_agg_mode)
    else:
        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask,
                           loss_agg_mode=loss_agg_mode)
    return kl_loss * kl_loss_coef, kld


# ─────────────────────────────────────────────────────────────────────────────
print("\n═══ D.   β_KL ═══")

torch.manual_seed(0)
bs, seq = 4, 12
log_prob = torch.randn(bs, seq) * 0.2
ref_lp  = torch.randn(bs, seq) * 0.2
adv     = torch.randn(bs, seq)        #  
mask    = torch.ones(bs, seq)
mask[:, -2:] = 0

kl_coef = 0.08
b_pos, b_neg = 0.5, 1.5

# D1:   KL 
loss_sym, kld = _apply_asym_kl(log_prob, ref_lp, adv, mask, kl_coef)
check("D1   KL   scalar", loss_sym.ndim == 0)

# D2:   KL   scalar
loss_asym, _ = _apply_asym_kl(log_prob, ref_lp, adv, mask, kl_coef, b_pos, b_neg)
check("D2   KL   scalar", loss_asym.ndim == 0)
check("D2   KL  ", torch.isfinite(loss_asym))

# D3:   β_t  
pos_mask = adv > 0
beta_t = torch.where(pos_mask,
                     adv.new_full(adv.shape, b_pos),
                     adv.new_full(adv.shape, b_neg))
expected = agg_loss(loss_mat=kld * beta_t, loss_mask=mask,
                    loss_agg_mode="token-mean") * kl_coef
check("D3   KL  ",
      torch.allclose(loss_asym, expected, atol=1e-6),
      f"asym={loss_asym:.6f}  manual={expected:.6f}")

# D4:   β_pos=β_neg=1.0   KL ==   KL
loss_eq, _ = _apply_asym_kl(log_prob, ref_lp, adv, mask, kl_coef, 1.0, 1.0)
check("D4 β_pos=β_neg=1.0   KL",
      torch.allclose(loss_sym, loss_eq, atol=1e-6),
      f"sym={loss_sym:.6f}  eq={loss_eq:.6f}")

# D5:   token   KL <   KL 

adv_pos_all = adv.abs()          #   token  
loss_sym_pos, _ = _apply_asym_kl(log_prob, ref_lp, adv_pos_all, mask, kl_coef)
loss_asym_pos, _ = _apply_asym_kl(log_prob, ref_lp, adv_pos_all, mask, kl_coef, b_pos, b_neg)
check("D5   KL <   KL β_pos<1  ",
      loss_asym_pos.item() < loss_sym_pos.item(),
      f"asym={loss_asym_pos:.4f}  sym={loss_sym_pos:.4f}")

# D6:   token   KL >   KL 
adv_neg_all = -adv.abs()         #   token  
loss_sym_neg, _ = _apply_asym_kl(log_prob, ref_lp, adv_neg_all, mask, kl_coef)
loss_asym_neg, _ = _apply_asym_kl(log_prob, ref_lp, adv_neg_all, mask, kl_coef, b_pos, b_neg)
check("D6   KL >   KL β_neg>1  ",
      loss_asym_neg.item() > loss_sym_neg.item(),
      f"asym={loss_asym_neg:.4f}  sym={loss_sym_neg:.4f}")

# D7: 50/50  β=(0.5,1.5)  =1.0  ≈   KL
#   50/50 mask
adv_balanced = torch.cat([adv.abs()[:, :seq//2], -adv.abs()[:, seq//2:]], dim=1)
mask_full = torch.ones(bs, seq)
loss_sym_bal, _ = _apply_asym_kl(log_prob, ref_lp, adv_balanced, mask_full, kl_coef)
loss_asym_bal, _ = _apply_asym_kl(log_prob, ref_lp, adv_balanced, mask_full, kl_coef, b_pos, b_neg)
#  : (0.5 * kld_pos_mean + 1.5 * kld_neg_mean) ≈ kld_mean when kld_pos≈kld_neg
#   < 20% log_prob   ref_lp  kld  
ratio = abs(loss_asym_bal.item() - loss_sym_bal.item()) / (loss_sym_bal.item() + 1e-8)
check("D7 50/50   <30% β  ≈1 ",
      ratio < 0.30,
      f"asym={loss_asym_bal:.4f}  sym={loss_sym_bal:.4f}  diff_ratio={ratio:.2%}")

# D8: kl_loss_coef  
loss_half, _ = _apply_asym_kl(log_prob, ref_lp, adv, mask, kl_coef / 2, b_pos, b_neg)
check("D8 kl_loss_coef   loss  ",
      torch.allclose(loss_half * 2, loss_asym, atol=1e-5),
      f"half*2={loss_half.item()*2:.6f}  full={loss_asym.item():.6f}")

# D9:   —   Clip  
print("\n═══ E.   Clip +   β_KL   ═══")
from verl.trainer.ppo.core_algos import compute_policy_loss

loss_pg, _, _, _ = compute_policy_loss(
    log_prob, log_prob + 0.1, adv, mask,
    cliprange=0.2, clip_ratio_pos=0.20, clip_ratio_neg=0.30)
loss_kl, _ = _apply_asym_kl(log_prob, ref_lp, adv, mask, kl_coef, b_pos, b_neg)
combined = loss_pg + loss_kl
check("E1   loss  ",
      combined.ndim == 0 and torch.isfinite(combined),
      f"pg={loss_pg:.4f}  kl={loss_kl:.4f}  total={combined:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 50)
if _all_passed:
    print(f"{PASS}   ")
else:
    print(f"{FAIL}   ")
    sys.exit(1)
