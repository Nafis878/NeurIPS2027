"""Experiment 4: middle-weighted training intervention (matched compute)."""
import argparse
import os

import _bootstrap  # noqa: F401
from _train_common import run_finetune_experiment
from src.interventions import middle_weight_fn, inverse_influence_weights
from src.utils import (load_config, apply_overrides, set_seed, repo_path, read_jsonl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=repo_path("configs", "default.yaml"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--beta", type=float, default=None)
    ap.add_argument("--sigma", type=float, default=None)
    ap.add_argument("--inverse-influence", action="store_true",
                    help="Use inverse-baseline-influence reweighting instead of the Gaussian bump.")
    args = ap.parse_args()

    cfg = load_config(args.config, smoke=args.smoke)
    cfg = apply_overrides(cfg, {
        "model": args.model, "epochs": args.epochs,
        "max_steps": args.max_steps, "lr": args.lr,
    })
    if args.beta is not None:
        cfg["intervention"]["beta"] = args.beta
    if args.sigma is not None:
        cfg["intervention"]["sigma"] = args.sigma
    set_seed(cfg["seed"])

    if args.inverse_influence or cfg["intervention"].get("inverse_influence"):
        baseline_csv = os.path.join(repo_path(cfg["influence"]["out_tables"]),
                                    "influence_baseline_by_position.csv")
        if not os.path.exists(baseline_csv):
            raise FileNotFoundError(
                "inverse_influence requested but baseline influence table missing; "
                "run scripts/03_measure_influence.py first.")
        import csv
        with open(baseline_csv, newline="", encoding="utf-8") as fh:
            rows = [{"position_bucket": float(r["position_bucket"]),
                     "influence_at_answer": float(r["influence_at_answer"])}
                    for r in csv.DictReader(fh)]
        weight_fn = inverse_influence_weights(rows)
        print("[05] Using inverse-baseline-influence reweighting.")
    else:
        weight_fn = middle_weight_fn(cfg)
        print(f"[05] Using Gaussian middle weighting beta={cfg['intervention']['beta']} "
              f"sigma={cfg['intervention']['sigma']}")

    run_finetune_experiment(
        cfg, weight_fn=weight_fn, tag="intervention",
        loss_curve_name="loss_curve_intervention.png",
        loss_title="Training loss (middle-weighted intervention)")


if __name__ == "__main__":
    main()
