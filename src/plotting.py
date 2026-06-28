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
