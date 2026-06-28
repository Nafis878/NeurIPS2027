"""Position-aware training interventions.

`position_weight` up-weights the loss of examples whose answer-fact sits near the
middle of the context, which is exactly where causal-residual transformers tend to
under-retrieve. `inverse_influence_weights` is the optional reweighting that uses the
baseline influence profile (weight ~ 1 / baseline influence at that position).
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional


def position_weight(x: float, beta: float = 2.0, sigma: float = 0.18) -> float:
    """Gaussian bump centered at the middle (x=0.5) of the normalized context."""
    return 1.0 + beta * math.exp(-((x - 0.5) ** 2) / (2 * sigma ** 2))


def middle_weight_fn(cfg: Dict[str, Any]) -> Callable[[Dict[str, Any]], float]:
    beta = float(cfg["intervention"]["beta"])
    sigma = float(cfg["intervention"]["sigma"])
    return lambda ex: position_weight(ex["norm_pos"], beta=beta, sigma=sigma)


def inverse_influence_weights(
    baseline_by_bucket: List[Dict[str, Any]], eps: float = 1e-6
) -> Callable[[Dict[str, Any]], float]:
    """Build a weight fn ~ 1 / (baseline influence at the example's bucket), normalized to mean 1."""
    infl = {row["position_bucket"]: row["influence_at_answer"] for row in baseline_by_bucket}
    raw = {b: 1.0 / (v + eps) for b, v in infl.items()}
    mean = sum(raw.values()) / max(1, len(raw))
    norm = {b: v / mean for b, v in raw.items()}

    def _fn(ex: Dict[str, Any]) -> float:
        return norm.get(ex["position_bucket"], 1.0)

    return _fn


def describe_weights(examples: List[Dict[str, Any]],
                     weight_fn: Optional[Callable[[Dict[str, Any]], float]]) -> Dict[str, float]:
    if weight_fn is None:
        return {"min": 1.0, "max": 1.0, "mean": 1.0}
    ws = [weight_fn(ex) for ex in examples]
    return {"min": min(ws), "max": max(ws), "mean": sum(ws) / len(ws)}
