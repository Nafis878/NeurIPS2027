"""Claim 2 — "Curable at birth": reprogram the initialization prior.

The target paper proves the lost-in-the-middle U-shape is an immutable geometric birthright
of causal masking + residual connections, modeling each layer as N = (1-alpha) I + alpha M.
We treat that residual mixing as an EDITABLE knob and provide two init-time cures:

  (A) residual-alpha reshaping -- per-layer scaling of the residual branch via forward hooks,
      so block l computes  h <- h_in + s_l * (h_out - h_in).  Increasing s_l amplifies the
      causal-mixing (Cesaro) path relative to the identity/recency anchor, redistributing
      influence toward the starved middle. The schedule is chosen to MINIMIZE the Step-0
      valley depth (the fingerprint as a design target).

  (B) distributed anchor registers -- see data.insert_anchors: inject anchor tokens at
      intervals so every region gets a local readout anchor (the mechanism that rescues the end).

These are architectural / data interventions, NOT loss weighting -- the point the paper's
"future work" did not consider.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


def get_blocks(model):
    """Return the list of transformer blocks, across GPTNeoX (Pythia) and GPT-2."""
    m = model
    if hasattr(m, "gpt_neox"):
        return list(m.gpt_neox.layers)
    if hasattr(m, "transformer") and hasattr(m.transformer, "h"):
        return list(m.transformer.h)
    # generic fallback: find the longest ModuleList of repeated blocks
    import torch.nn as nn
    best = []
    for mod in m.modules():
        if isinstance(mod, nn.ModuleList) and len(mod) > len(best):
            best = list(mod)
    if not best:
        raise RuntimeError("Could not locate transformer blocks for residual scaling.")
    return best


def _residual_hook(scale: float):
    def hook(module, inputs, output):
        h_in = inputs[0]
        if isinstance(output, tuple):
            h_out = output[0]
            new = h_in + scale * (h_out - h_in)
            return (new,) + tuple(output[1:])
        return h_in + scale * (output - h_in)
    return hook


def apply_residual_scaling(model, schedule: List[float]):
    """Register forward hooks implementing h <- h_in + s_l*(h_out - h_in) per block.

    ``schedule`` is a per-layer list of scales (len == #blocks) or a single float broadcast
    to all layers. Returns a list of hook handles; call .remove() on each to undo.
    """
    blocks = get_blocks(model)
    if isinstance(schedule, (int, float)):
        schedule = [float(schedule)] * len(blocks)
    if len(schedule) != len(blocks):
        raise ValueError(f"schedule length {len(schedule)} != #blocks {len(blocks)}")
    handles = []
    for blk, s in zip(blocks, schedule):
        handles.append(blk.register_forward_hook(_residual_hook(float(s))))
    return handles


def remove_hooks(handles):
    for h in handles:
        h.remove()


def make_schedule(blocks_n: int, kind: str = "uniform", value: float = 1.5,
                  ramp_end: Optional[float] = None) -> List[float]:
    """Build a residual-scale schedule.

    kind="uniform" -> all layers = value.
    kind="ramp"    -> linearly from value (layer 0) to ramp_end (last layer).
    """
    if kind == "ramp" and ramp_end is not None:
        return list(np.linspace(value, ramp_end, blocks_n))
    return [float(value)] * blocks_n


def search_residual_schedule(model, examples, device, cfg) -> Dict[str, Any]:
    """Grid-search residual-scale schedules to MINIMIZE the Step-0 influence valley depth.

    The model passed in should be the (random-init) Step-0 model. For each candidate schedule
    we apply hooks, measure the influence profile, compute valley_metrics, then remove hooks.
    Returns {'best': {...}, 'all': [...]} with the schedule that flattens the valley most.
    """
    from src import influence as infl

    blocks_n = len(get_blocks(model))
    grid = cfg["architecture"]["residual_grid"]
    mid_lo = float(cfg["influence"]["middle_low"])
    mid_hi = float(cfg["influence"]["middle_high"])
    # cap examples for the search (cheap)
    search_cap = int(cfg["architecture"].get("search_max_examples", 33))
    sub = examples[:search_cap] if search_cap < len(examples) else examples

    results = []
    # include the unmodified baseline (scale=1.0) as reference
    candidates = [{"name": "none", "kind": "uniform", "value": 1.0, "ramp_end": None}] + list(grid)
    for cand in candidates:
        sched = make_schedule(blocks_n, cand.get("kind", "uniform"),
                              float(cand.get("value", 1.0)), cand.get("ramp_end"))
        handles = apply_residual_scaling(model, sched)
        try:
            res = infl.run_influence(model, None, sub, device, cfg, verbose=False,
                                     metric=cfg["architecture"].get("metric", "jacobian"))
        finally:
            remove_hooks(handles)
        vm = infl.valley_metrics(res["bin_centers"], res["global_profile"], mid_lo, mid_hi)
        row = {"name": cand["name"], "schedule": sched,
               "valley_depth": vm["valley_depth"], "peak_to_trough": vm["peak_to_trough"],
               "middle_mass": vm["middle_mass"]}
        results.append(row)
        print(f"[arch] schedule '{cand['name']}' (s~{sched[0]:.2f}): "
              f"valley_depth={vm['valley_depth']:.3f} peak/trough={vm['peak_to_trough']:.2f}")
    best = min(results, key=lambda r: r["valley_depth"])
    print(f"[arch] best flattening schedule: '{best['name']}' "
          f"(valley_depth={best['valley_depth']:.3f})")
    return {"best": best, "all": results, "n_blocks": blocks_n}
