"""
GTPO Phase-1  
 
  ① use_gspo_ratio  ——  
  ② clip_ratio_pos / clip_ratio_neg ——   Clip
 : python3 scripts/test_gtpo_phase1.py
"""
import sys
import os

#   verl   PYTHONPATH
_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "verl_arpo_entropy")
sys.path.insert(0, _root)

import torch
from verl.trainer.ppo.core_algos import compute_policy_loss, agg_loss

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
_all_passed = True


def check(name, cond, msg=""):
    global _all_passed
    tag = PASS if cond else FAIL
    print(f"  {tag}  {name}" + (f"  [{msg}]" if msg else ""))
    if not cond:
        _all_passed = False


# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
def make_batch(bs=4, seq=10):
    torch.manual_seed(42)
    old_lp = torch.randn(bs, seq) * 0.1       #   log_prob
    lp = old_lp + torch.randn(bs, seq) * 0.3  #   log_prob 
    adv = torch.randn(bs, seq)                  # advantages 
    mask = torch.ones(bs, seq)
    mask[:, -2:] = 0                            #   2   padding
    return old_lp, lp, adv, mask


# ─────────────────────────────────────────────────────────────────────────────
#   A GSPO  
# ─────────────────────────────────────────────────────────────────────────────
print("\n═══ A. GSPO   ═══")

old_lp, lp, adv, mask = make_batch(bs=4, seq=10)

# A1: ratio  
loss_std, *_ = compute_policy_loss(old_lp, lp, adv, mask, cliprange=0.2,
                                    use_gspo_ratio=False)
loss_gspo, *_ = compute_policy_loss(old_lp, lp, adv, mask, cliprange=0.2,
                                     use_gspo_ratio=True)
check("A1   scalar loss",
      loss_std.ndim == 0 and loss_gspo.ndim == 0)

# A2: GSPO ratio   exp(mean(Δlog))
#   ratio
log_ratio = lp - old_lp                                     # (bs, seq)
seq_log_r = (log_ratio * mask).sum(-1) / mask.sum(-1)       # (bs,)
expected_ratio = seq_log_r.exp().unsqueeze(-1).expand_as(lp)  # (bs, seq)
#   expected_ratio   clip loss   compute_policy_loss  
cliprange = 0.2
pg1 = -adv * expected_ratio
pg2 = -adv * torch.clamp(expected_ratio, 1 - cliprange, 1 + cliprange)
pg_clip = torch.maximum(pg1, pg2)
expected_loss = agg_loss(pg_clip, mask, "token-mean")
check("A2 GSPO loss  ",
      torch.allclose(loss_gspo, expected_loss, atol=1e-5),
      f"gspo={loss_gspo:.6f}  manual={expected_loss:.6f}")

# A3:   1   token  GSPO ratio = token ratio
old_lp1, lp1, adv1, mask1 = (t[:, :1] for t in make_batch(bs=2, seq=1))
mask1 = torch.ones(2, 1)
_, _, _, _ = old_lp1, lp1, adv1, mask1  # just bind
loss_tok, *_ = compute_policy_loss(old_lp1, lp1, adv1, mask1, cliprange=0.2,
                                    use_gspo_ratio=False)
loss_gs1, *_ = compute_policy_loss(old_lp1, lp1, adv1, mask1, cliprange=0.2,
                                    use_gspo_ratio=True)
check("A3   token   GSPO = token-level loss",
      torch.allclose(loss_tok, loss_gs1, atol=1e-5),
      f"tok={loss_tok:.6f}  gspo={loss_gs1:.6f}")

# A4:   token log_prob  GSPO loss = std loss
old_eq = torch.zeros(3, 8)
lp_eq = old_eq + 0.1                    #  
adv_eq = torch.randn(3, 8)
mask_eq = torch.ones(3, 8)
l_std_eq, *_ = compute_policy_loss(old_eq, lp_eq, adv_eq, mask_eq, cliprange=0.2,
                                    use_gspo_ratio=False)
l_gs_eq, *_ = compute_policy_loss(old_eq, lp_eq, adv_eq, mask_eq, cliprange=0.2,
                                   use_gspo_ratio=True)
check("A4   GSPO = standard loss",
      torch.allclose(l_std_eq, l_gs_eq, atol=1e-5),
      f"std={l_std_eq:.6f}  gspo={l_gs_eq:.6f}")


# ─────────────────────────────────────────────────────────────────────────────
#   B  Clip
# ─────────────────────────────────────────────────────────────────────────────
print("\n═══ B.   Clip ═══")

old_lp, lp, adv, mask = make_batch(bs=4, seq=10)
eps_pos, eps_neg = 0.20, 0.30

loss_asym, clipfrac_asym, kl_asym, _ = compute_policy_loss(
    old_lp, lp, adv, mask, cliprange=0.2,
    clip_ratio_pos=eps_pos, clip_ratio_neg=eps_neg)
loss_sym, clipfrac_sym, kl_sym, _ = compute_policy_loss(
    old_lp, lp, adv, mask, cliprange=0.2)

check("B1  / ", loss_asym.ndim == 0 and loss_sym.ndim == 0)

# B2:   clip  
ratio = torch.exp(lp - old_lp)
pos_mask = adv > 0
clipped = torch.where(pos_mask,
                       torch.clamp(ratio, max=1.0 + eps_pos),
                       torch.clamp(ratio, min=1.0 - eps_neg))
pg1_b = -adv * ratio
pg2_b = -adv * clipped
clip_pg = torch.maximum(pg1_b, pg2_b)
cliprange_c = 3.0
pg3_b = -adv * cliprange_c
clip_pg2 = torch.min(pg3_b, clip_pg)
pg_losses = torch.where(adv < 0, clip_pg2, clip_pg)
expected_asym_loss = agg_loss(pg_losses, mask, "token-mean")
check("B2   loss  ",
      torch.allclose(loss_asym, expected_asym_loss, atol=1e-5),
      f"asym={loss_asym:.6f}  manual={expected_asym_loss:.6f}")

# B3: ε_+ = ε_- = ε   clip !=   clip 
#  Â>0  Â<0  
loss_same_eps, *_ = compute_policy_loss(old_lp, lp, adv, mask, cliprange=0.2,
                                         clip_ratio_pos=0.2, clip_ratio_neg=0.2)

check("B3 ε_+=ε_-=0.2   clip  ",
      torch.isfinite(loss_same_eps))

# B4:   token   1+ε_+  
#   ratio >> 1+ε_+  
old_high = torch.zeros(1, 1)
lp_high = old_high + 2.0                 # ratio = exp(2) ≈ 7.4 >> 1 + 0.2
adv_pos = torch.tensor([[1.0]])          #  
mask1t = torch.ones(1, 1)
loss_high_pos, *_ = compute_policy_loss(old_high, lp_high, adv_pos, mask1t,
                                         cliprange=0.2,
                                         clip_ratio_pos=0.2, clip_ratio_neg=0.2)
#   loss = -1.0 * 1.2 = -1.2 → pg_loss = -(-1.2) = 1.2
#   agg_loss   mean(pg_losses)  pg_losses = max(-r*A, -clip_r*A)
# r=7.4>1.2, A=1>0: pg1=-7.4, pg2=-1.2, max=pg2=-1.2; loss=-(-1.2)... wait, pg_loss = agg(pg_losses)
# pg_losses = max(-r*A, -clip*A) = max(-7.4, -1.2) = -1.2  (  loss_mat)
# pg_loss = mean(pg_losses * mask) = -1.2  ( )
expected_clipped = -1.2
check("B4   ratio   1+ε_+",
      abs(loss_high_pos.item() - expected_clipped) < 1e-4,
      f"got={loss_high_pos.item():.4f}  expected={expected_clipped:.4f}")

# B5:   token ratio < 1-ε_-  
lp_low = old_high - 2.0                 # ratio = exp(-2) ≈ 0.135 < 1 - 0.3 = 0.7
adv_neg = torch.tensor([[-1.0]])        #  
loss_low_neg, *_ = compute_policy_loss(old_high, lp_low, adv_neg, mask1t,
                                        cliprange=0.2,
                                        clip_ratio_pos=0.2, clip_ratio_neg=0.3)
# r=0.135 < 0.7, A=-1:
# pg1 = -(-1)*0.135 = 0.135
# pg2 = -(-1)*0.7 = 0.7  (clamp from below at 0.7)
# max(0.135, 0.7) = 0.7
# dual-clip: pg3 = -(-1)*3.0 = 3.0; min(3.0, 0.7) = 0.7
# pg_losses where A<0: clip_pg2=0.7
expected_neg_clipped = 0.7
check("B5   ratio   1-ε_-",
      abs(loss_low_neg.item() - expected_neg_clipped) < 1e-4,
      f"got={loss_low_neg.item():.4f}  expected={expected_neg_clipped:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
#   C  GSPO +   Clip
# ─────────────────────────────────────────────────────────────────────────────
print("\n═══ C. GSPO +   Clip   ═══")

old_lp, lp, adv, mask = make_batch(bs=6, seq=12)
loss_combo, clipfrac_c, kl_c, _ = compute_policy_loss(
    old_lp, lp, adv, mask, cliprange=0.2,
    use_gspo_ratio=True,
    clip_ratio_pos=0.20, clip_ratio_neg=0.30)

check("C1   scalar", loss_combo.ndim == 0)
check("C2   loss  ", torch.isfinite(loss_combo))
check("C3 clipfrac   [0,1]  ", 0.0 <= clipfrac_c.item() <= 1.0,
      f"clipfrac={clipfrac_c.item():.3f}")
# ppo_kl = mean(old_lp - lp)   KL  
check("C4 KL  <5 ", abs(kl_c.item()) < 5.0,
      f"kl={kl_c.item():.5f}")


# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 50)
if _all_passed:
    print(f"{PASS}   ")
else:
    print(f"{FAIL}   ")
    sys.exit(1)
