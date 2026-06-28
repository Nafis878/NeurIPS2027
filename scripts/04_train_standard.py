"""Experiment 3: ordinary fine-tuning baseline (no position-aware intervention)."""
import argparse

import _bootstrap  # noqa: F401
from _train_common import run_finetune_experiment
from src.utils import load_config, apply_overrides, set_seed, repo_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=repo_path("configs", "default.yaml"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, smoke=args.smoke)
    cfg = apply_overrides(cfg, {
        "model": args.model, "epochs": args.epochs,
        "max_steps": args.max_steps, "lr": args.lr,
    })
    set_seed(cfg["seed"])

    run_finetune_experiment(
        cfg, weight_fn=None, tag="standard_finetune",
        loss_curve_name="loss_curve_standard_finetune.png",
        loss_title="Training loss (standard fine-tune)")


if __name__ == "__main__":
    main()
