"""Experiment 5: sweep position-weighting configs to find a NET middle-accuracy win.

Motivation: at beta=2, sigma=0.18, edge_gamma=0 the middle-weighted intervention lifts
middle accuracy but cannibalizes the edges (average stays flat). This sweep searches over
beta / sigma / edge-floor (edge_gamma, edge_sigma) with matched compute and scores each
config on middle accuracy *subject to not regressing* average or worst accuracy vs the
'standard' (uniform) control. The goal is to turn redistribution into a net gain.

For each config: load a FRESH pretrained model (matched seed/data/steps), train with that
config's weighting, evaluate accuracy vs position, and record middle/avg/worst/edge accuracy.
"""
import argparse
import os

import _bootstrap  # noqa: F401
from src import data, evaluate, plotting, train as train_mod
from src.interventions import make_weight_fn, describe_weights
from src.utils import (load_config, apply_overrides, set_seed, pick_device,
                       load_model_tokenizer, read_jsonl, write_csv, write_json,
                       save_run_config, ensure_dir, repo_path)


def _load_dataset(cfg):
    path = os.path.join(repo_path(cfg["data"]["out_dir"]), data.dataset_filename(cfg))
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found: {path}\nRun scripts/01_generate_dataset.py first.")
    return read_jsonl(path)


def _acc_stats(by_bucket, lo=0.4, hi=0.6, edge=0.15):
    """middle / average / worst / edge accuracy from per-bucket eval rows."""
    accs = [(r["mean_norm_pos"], r["accuracy"]) for r in by_bucket]
    mids = [a for p, a in accs if lo <= p <= hi]
    edges = [a for p, a in accs if p <= edge or p >= 1 - edge]
    avg = sum(a for _, a in accs) / len(accs)
    return {
        "middle_acc": sum(mids) / len(mids) if mids else float("nan"),
        "avg_acc": avg,
        "worst_acc": min(a for _, a in accs),
        "edge_acc": sum(edges) / len(edges) if edges else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=repo_path("configs", "default.yaml"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--max-configs", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, smoke=args.smoke)
    cfg = apply_overrides(cfg, {"model": args.model})
    set_seed(cfg["seed"])

    configs = list(cfg["sweep"]["configs"])
    cap = args.max_configs or cfg.get("sweep", {}).get("max_configs")
    if cap is not None:
        configs = configs[: int(cap)]

    examples = _load_dataset(cfg)
    device = pick_device()
    tables = ensure_dir(repo_path(cfg["sweep"]["out_tables"]))
    plots = ensure_dir(repo_path(cfg["sweep"]["out_plots"]))

    rows = []
    per_position = {}
    for i, spec in enumerate(configs):
        name = spec["name"]
        weight_fn = make_weight_fn(spec)
        print(f"\n[08] === config {i+1}/{len(configs)}: {name} "
              f"(beta={spec.get('beta')}, sigma={spec.get('sigma')}, "
              f"edge_gamma={spec.get('edge_gamma')}) ===")
        wstats = describe_weights(examples, weight_fn)
        print(f"[08] weights: min={wstats['min']:.2f} max={wstats['max']:.2f} mean={wstats['mean']:.2f}")

        # Fresh model per config -> matched compute, no contamination.
        model, tokenizer, model_name = load_model_tokenizer(cfg, device, for_training=True)
        train_mod.train(model, tokenizer, examples, device, cfg,
                        weight_fn=weight_fn, seed=cfg["seed"], verbose=False)
        eval_res = evaluate.run_eval(model, tokenizer, examples, device, cfg, verbose=False)
        stats = _acc_stats(eval_res["by_bucket"])
        row = {"name": name, **{k: spec.get(k) for k in ("beta", "sigma", "edge_gamma", "edge_sigma")},
               **{k: round(v, 4) for k, v in stats.items()}}
        rows.append(row)
        per_position[name] = eval_res["by_bucket"]
        print(f"[08] {name}: middle={stats['middle_acc']:.3f} avg={stats['avg_acc']:.3f} "
              f"worst={stats['worst_acc']:.3f} edge={stats['edge_acc']:.3f}")
        del model

    # Score: maximize middle accuracy, penalize regressions vs the 'standard' control.
    ctrl = next((r for r in rows if r["name"] == "standard"), None)
    for r in rows:
        score = r["middle_acc"]
        if ctrl is not None:
            if r["avg_acc"] < ctrl["avg_acc"]:
                score -= (ctrl["avg_acc"] - r["avg_acc"])        # penalize avg regression
            if r["worst_acc"] < ctrl["worst_acc"]:
                score -= (ctrl["worst_acc"] - r["worst_acc"])    # penalize worst regression
        r["score"] = round(score, 4)

    rows_sorted = sorted(rows, key=lambda r: r["score"], reverse=True)
    write_csv(os.path.join(tables, "sweep_intervention.csv"), rows_sorted,
              fieldnames=["name", "beta", "sigma", "edge_gamma", "edge_sigma",
                          "middle_acc", "avg_acc", "worst_acc", "edge_acc", "score"])
    write_json(os.path.join(tables, "sweep_per_position.json"), per_position)
    save_run_config(cfg, tables, "run_config_08_sweep.json")
    plotting.plot_sweep(rows, os.path.join(plots, "sweep_intervention_accuracy.png"),
                        "Position-weighting sweep: accuracy by config")

    best = rows_sorted[0]
    print("\n[08] === Sweep ranking (by score) ===")
    for r in rows_sorted:
        flag = ""
        if ctrl is not None and r["name"] != "standard":
            net = (r["middle_acc"] > ctrl["middle_acc"] and r["avg_acc"] >= ctrl["avg_acc"]
                   and r["worst_acc"] >= ctrl["worst_acc"])
            flag = "  <-- NET WIN" if net else ""
        print(f"   {r['name']:>16}: score={r['score']:.3f} middle={r['middle_acc']:.3f} "
              f"avg={r['avg_acc']:.3f} worst={r['worst_acc']:.3f}{flag}")
    print(f"\n[08] Best config: {best['name']} "
          f"(beta={best['beta']}, sigma={best['sigma']}, edge_gamma={best['edge_gamma']})")
    if ctrl is not None:
        print(f"[08] vs standard control: middle {ctrl['middle_acc']:.3f} -> {best['middle_acc']:.3f}, "
              f"avg {ctrl['avg_acc']:.3f} -> {best['avg_acc']:.3f}, "
              f"worst {ctrl['worst_acc']:.3f} -> {best['worst_acc']:.3f}")


if __name__ == "__main__":
    main()
