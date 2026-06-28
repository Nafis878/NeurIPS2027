"""Experiment 2: influence / Jacobian proxy vs. position.

Cheap influence proxy:
  inputs_embeds = embed(ids); inputs_embeds.requires_grad_(True)
  objective = sum_t log p(answer_t | prefix)        (teacher-forced)
  grad = d objective / d inputs_embeds
  per-token influence = ||grad_token||_2  (L2 over the embedding dim)

We aggregate per-token influence by normalized context position, and for each
position bucket report: mean influence at the planted-answer span, mean influence at
surrounding tokens, peak-to-trough ratio, and middle influence mass.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

import numpy as np


def _token_influence(model, ex, device):
    """Return (influence[np.array over full seq], context_len) for one example."""
    import torch
    import torch.nn.functional as F

    full = ex["input_ids"] + ex["answer_ids"]
    ids = torch.tensor([full], device=device)
    embed_layer = model.get_input_embeddings()

    with torch.enable_grad():
        inputs_embeds = embed_layer(ids).detach().clone().requires_grad_(True)
        logits = model(inputs_embeds=inputs_embeds).logits[0]  # [T, V]
        logprobs = F.log_softmax(logits.float(), dim=-1)
        start = len(ex["input_ids"])
        n_ans = len(ex["answer_ids"])
        obj = 0.0
        for k in range(n_ans):
            pos = start + k - 1
            tgt = full[start + k]
            obj = obj + logprobs[pos, tgt]
        (grad,) = torch.autograd.grad(obj, inputs_embeds)
    influence = grad[0].detach().float().norm(dim=-1).cpu().numpy()  # [T]
    return influence, ex["context_token_len"]


def _bin_profile(influence: np.ndarray, ctx_len: int, n_bins: int):
    """Mean influence per uniform position bin over the context region [0,1]."""
    ctx = influence[:ctx_len]
    pos = np.arange(ctx_len) / max(1, ctx_len - 1)
    bin_idx = np.minimum((pos * n_bins).astype(int), n_bins - 1)
    sums = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    np.add.at(sums, bin_idx, ctx)
    np.add.at(counts, bin_idx, 1.0)
    counts[counts == 0] = 1.0
    return sums / counts


def run_influence(model, tokenizer, examples, device, cfg, verbose=True) -> Dict[str, Any]:
    n_bins = int(cfg["influence"]["n_bins"])
    mid_lo = float(cfg["influence"]["middle_low"])
    mid_hi = float(cfg["influence"]["middle_high"])
    cap = cfg["influence"].get("max_examples")
    if cap is not None and int(cap) < len(examples):
        # Stratified subsample: spread evenly across the (bucket-ordered) examples so
        # every position bucket is represented rather than just the first few.
        idx = np.linspace(0, len(examples) - 1, int(cap)).round().astype(int)
        idx = sorted(set(int(i) for i in idx))
        examples = [examples[i] for i in idx]

    model.eval()
    bin_centers = (np.arange(n_bins) + 0.5) / n_bins
    middle_bins = (bin_centers >= mid_lo) & (bin_centers <= mid_hi)

    per_example: List[Dict[str, Any]] = []
    profiles_by_bucket: Dict[Any, List[np.ndarray]] = defaultdict(list)
    all_profiles: List[np.ndarray] = []

    for i, ex in enumerate(examples):
        influence, ctx_len = _token_influence(model, ex, device)
        profile = _bin_profile(influence, ctx_len, n_bins)
        all_profiles.append(profile)
        profiles_by_bucket[ex["position_bucket"]].append(profile)

        c0, c1 = ex["fact_code_span"]
        ans_infl = float(influence[c0:c1].mean())
        W = max(8, ctx_len // 32)
        lo = max(0, c0 - W)
        hi = min(ctx_len, c1 + W)
        surround_idx = list(range(lo, c0)) + list(range(c1, hi))
        surround = float(influence[surround_idx].mean()) if surround_idx else float("nan")

        per_example.append({
            "id": ex["id"],
            "position_bucket": ex["position_bucket"],
            "norm_pos": ex["norm_pos"],
            "context_length": ex["context_length"],
            "influence_at_answer": ans_infl,
            "influence_surrounding": surround,
        })
        if verbose and (i + 1) % 10 == 0:
            print(f"[influence] {i + 1}/{len(examples)} examples")

    # Per-bucket aggregation.
    by_bucket: List[Dict[str, Any]] = []
    for bucket in sorted(profiles_by_bucket):
        profs = np.stack(profiles_by_bucket[bucket], axis=0)
        mean_prof = profs.mean(axis=0)
        ex_rows = [r for r in per_example if r["position_bucket"] == bucket]
        denom = mean_prof.sum() if mean_prof.sum() != 0 else 1.0
        trough = mean_prof[mean_prof > 0].min() if (mean_prof > 0).any() else 1e-12
        by_bucket.append({
            "position_bucket": bucket,
            "n": len(ex_rows),
            "mean_norm_pos": float(np.mean([r["norm_pos"] for r in ex_rows])),
            "influence_at_answer": float(np.mean([r["influence_at_answer"] for r in ex_rows])),
            "influence_surrounding": float(np.nanmean([r["influence_surrounding"] for r in ex_rows])),
            "peak_to_trough": float(mean_prof.max() / trough),
            "middle_influence_mass": float(mean_prof[middle_bins].sum() / denom),
            "binned_profile": mean_prof.tolist(),
        })

    global_profile = np.stack(all_profiles, axis=0).mean(axis=0)
    return {
        "per_example": per_example,
        "by_bucket": by_bucket,
        "bin_centers": bin_centers.tolist(),
        "global_profile": global_profile.tolist(),
        "global_peak_to_trough": float(global_profile.max() / max(global_profile[global_profile > 0].min(), 1e-12)),
        "global_middle_mass": float(global_profile[middle_bins].sum() / max(global_profile.sum(), 1e-12)),
    }


def valley_metrics(bin_centers, profile, mid_lo: float = 0.4, mid_hi: float = 0.6) -> Dict[str, float]:
    """Quantify the depth of the lost-in-the-middle influence valley for one profile.

    Engages the paper's claim directly: a deep valley = a strong architectural position
    bias. We report the relative valley depth (1 - middle/edge), the recency-anchored
    middle floor (paper's Figure 3 normalization), peak-to-trough, and middle mass.
    """
    bc = np.asarray(bin_centers, dtype=float)
    p = np.asarray(profile, dtype=float)
    middle_mask = (bc >= mid_lo) & (bc <= mid_hi)
    edge = max(float(p[0]), float(p[-1]))           # higher of primacy / recency anchor
    middle = float(p[middle_mask].mean()) if middle_mask.any() else float("nan")
    recency = float(p[-1])
    trough = float(p[p > 0].min()) if (p > 0).any() else 1e-12
    return {
        "edge_peak": edge,
        "middle_floor": middle,
        "valley_depth": float(1.0 - middle / edge) if edge > 0 else float("nan"),
        "middle_over_recency": float(middle / recency) if recency > 0 else float("nan"),
        "peak_to_trough": float(p.max() / trough),
        "middle_mass": float(p[middle_mask].sum() / p.sum()) if p.sum() > 0 else float("nan"),
    }


def spearman(xs: List[float], ys: List[float]) -> Dict[str, float]:
    """Spearman correlation; returns rho + p (NaN-safe for tiny/degenerate inputs)."""
    from scipy.stats import spearmanr

    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if len(xs) < 3 or np.all(xs == xs[0]) or np.all(ys == ys[0]):
        return {"rho": float("nan"), "p": float("nan"), "n": int(len(xs))}
    rho, p = spearmanr(xs, ys)
    return {"rho": float(rho), "p": float(p), "n": int(len(xs))}
