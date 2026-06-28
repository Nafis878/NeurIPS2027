"""All figures. Uses a non-interactive backend and writes PNGs only."""
from __future__ import annotations

import os
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _ensure(path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def plot_accuracy_vs_position(bucket_rows: List[Dict[str, Any]], out_path: str,
                              title: str, metric: str = "accuracy",
                              ylabel: str = "Exact-match accuracy"):
    _ensure(out_path)
    xs = [r["mean_norm_pos"] for r in bucket_rows]
    ys = [r[metric] for r in bucket_rows]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(xs, ys, "o-", color="#1f77b4")
    ax.axvspan(0.4, 0.6, color="orange", alpha=0.12, label="middle band")
    ax.set_xlabel("Normalized fact position")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_influence_vs_position(bucket_rows: List[Dict[str, Any]], out_path: str, title: str):
    _ensure(out_path)
    xs = [r["mean_norm_pos"] for r in bucket_rows]
    ya = [r["influence_at_answer"] for r in bucket_rows]
    ys = [r.get("influence_surrounding", float("nan")) for r in bucket_rows]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(xs, ya, "o-", color="#d62728", label="influence at answer span")
    ax.plot(xs, ys, "s--", color="#7f7f7f", alpha=0.7, label="surrounding tokens")
    ax.axvspan(0.4, 0.6, color="orange", alpha=0.12, label="middle band")
    ax.set_xlabel("Normalized fact position")
    ax.set_ylabel("Input-embedding gradient norm")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_global_profile(bin_centers: List[float], profile: List[float], out_path: str, title: str):
    _ensure(out_path)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(bin_centers, profile, "o-", color="#9467bd")
    ax.axvspan(0.4, 0.6, color="orange", alpha=0.12, label="middle band")
    ax.set_xlabel("Normalized context position")
    ax.set_ylabel("Mean input-embedding gradient norm")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_profiles_overlay(profiles: Dict[str, Any], out_path: str, title: str,
                          anchor_recency: bool = False):
    """Overlay several global influence profiles. ``profiles`` maps name -> (bin_centers, profile).

    If ``anchor_recency`` the recency anchor (x=1) of each profile is normalized to 1.0
    (the paper's Figure 3 view), which exposes the *relative* depth of the middle valley.
    """
    _ensure(out_path)
    order = ["step0_init", "baseline", "standard_finetune", "intervention"]
    colors = {"step0_init": "#9467bd", "baseline": "#1f77b4",
              "standard_finetune": "#ff7f0e", "intervention": "#d62728"}
    labels = {"step0_init": "random init (Step 0)", "baseline": "pretrained",
              "standard_finetune": "standard FT", "intervention": "middle-weighted"}
    fig, ax = plt.subplots(figsize=(7.5, 4.7))
    keys = [k for k in order if k in profiles] + [k for k in profiles if k not in order]
    for name in keys:
        bc, prof = profiles[name]
        import numpy as np
        prof = np.asarray(prof, dtype=float)
        if anchor_recency and prof[-1] > 0:
            prof = prof / prof[-1]
        ax.plot(bc, prof, "o-", label=labels.get(name, name), color=colors.get(name))
    ax.axvspan(0.4, 0.6, color="orange", alpha=0.12, label="middle band")
    ax.set_xlabel("Normalized context position")
    ax.set_ylabel("Recency-anchored influence" if anchor_recency else "Mean gradient norm")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_loss_curve(history: List[Dict[str, Any]], out_path: str, title: str):
    _ensure(out_path)
    steps = [h["step"] for h in history]
    loss = [h["loss"] for h in history]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(steps, loss, "-", color="#2ca02c")
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Answer-span cross-entropy")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_fingerprint_scatter(rows: List[Dict[str, Any]], out_path: str, title: str,
                             region_thr: float = 0.7):
    """Scatter: Step-0 fingerprint (x) vs trained accuracy (y), one point per position bucket,
    colored by normalized position. Circles = architectural region (x<=thr, where the birthright
    fingerprint forecasts accuracy); X markers = learned-recency region (x>thr).
    """
    _ensure(out_path)
    fig, ax = plt.subplots(figsize=(6.8, 5))
    arch = [r for r in rows if r["mean_norm_pos"] <= region_thr]
    late = [r for r in rows if r["mean_norm_pos"] > region_thr]
    if arch:
        ax.scatter([r["fingerprint"] for r in arch], [r["trained_accuracy"] for r in arch],
                   c=[r["mean_norm_pos"] for r in arch], cmap="viridis", s=80, marker="o",
                   edgecolor="k", linewidth=0.4, label=f"architectural region (x<={region_thr})")
    if late:
        ax.scatter([r["fingerprint"] for r in late], [r["trained_accuracy"] for r in late],
                   c=[r["mean_norm_pos"] for r in late], cmap="viridis", s=90, marker="X",
                   edgecolor="k", linewidth=0.4, label="learned-recency region (x>thr)")
    for r in rows:
        ax.annotate(f"{r['mean_norm_pos']:.2f}", (r["fingerprint"], r["trained_accuracy"]),
                    fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Step-0 influence fingerprint (random init)")
    ax.set_ylabel("Trained exact-match accuracy")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_depth_trend(summaries: List[Dict[str, Any]], out_path: str, title: str):
    """Dual-axis: Step-0 valley depth and trained middle accuracy vs. model depth (#layers)."""
    _ensure(out_path)
    L = [float(s["model_layers"]) for s in summaries]
    depth = [float(s["step0_valley_depth"]) for s in summaries]
    mid = [float(s["trained_middle_acc"]) for s in summaries]
    fig, ax1 = plt.subplots(figsize=(7, 4.6))
    ax1.plot(L, depth, "o-", color="#9467bd", label="Step-0 valley depth")
    ax1.set_xlabel("Depth (number of layers, H)")
    ax1.set_ylabel("Step-0 valley depth", color="#9467bd")
    ax1.tick_params(axis="y", labelcolor="#9467bd")
    ax2 = ax1.twinx()
    ax2.plot(L, mid, "s--", color="#d62728", label="trained middle accuracy")
    ax2.set_ylabel("Trained middle accuracy", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax1.set_title(title)
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_sweep(rows: List[Dict[str, Any]], out_path: str, title: str,
               control_name: str = "standard"):
    """Grouped bars of middle / average / worst accuracy per sweep config.

    ``rows`` each have: name, middle_acc, avg_acc, worst_acc. The control config is
    marked so the reader can see which variants beat ordinary fine-tuning.
    """
    import numpy as np
    _ensure(out_path)
    names = [r["name"] for r in rows]
    middle = [r["middle_acc"] for r in rows]
    avg = [r["avg_acc"] for r in rows]
    worst = [r["worst_acc"] for r in rows]
    x = np.arange(len(names))
    w = 0.27
    fig, ax = plt.subplots(figsize=(max(7, 1.3 * len(names)), 4.8))
    ax.bar(x - w, middle, w, label="middle acc", color="#d62728")
    ax.bar(x, avg, w, label="avg acc", color="#1f77b4")
    ax.bar(x + w, worst, w, label="worst acc", color="#7f7f7f")
    # control reference lines
    ctrl = next((r for r in rows if r["name"] == control_name), None)
    if ctrl is not None:
        ax.axhline(ctrl["middle_acc"], color="#d62728", ls="--", lw=0.9, alpha=0.6)
        ax.axhline(ctrl["avg_acc"], color="#1f77b4", ls="--", lw=0.9, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Exact-match accuracy")
    ax.set_title(title + f"  (dashed = '{control_name}' control)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_comparison(methods: Dict[str, List[Dict[str, Any]]], out_path: str, title: str,
                    metric: str, ylabel: str):
    _ensure(out_path)
    colors = {"baseline": "#1f77b4", "standard": "#ff7f0e", "intervention": "#d62728"}
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for name, rows in methods.items():
        if not rows:
            continue
        xs = [r["mean_norm_pos"] for r in rows]
        ys = [r[metric] for r in rows]
        ax.plot(xs, ys, "o-", label=name, color=colors.get(name))
    ax.axvspan(0.4, 0.6, color="orange", alpha=0.12, label="middle band")
    ax.set_xlabel("Normalized fact position")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path
