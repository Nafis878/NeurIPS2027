"""Claim 2 — "Curable at birth": bake-off of cures for lost-in-the-middle.

At matched compute, compare arms that try to beat the architectural position-bias:
  standard        ordinary fine-tuning (control)
  loss_edgefloor  position-weighted loss, edge-floor winner (Phase-1; training-time fix)
  resid_alpha     INIT-time residual reshaping (search the schedule that flattens the Step-0
                  valley, apply it, then fine-tune)  <- the out-of-the-box cure
  anchors         distributed anchor registers in the data, then fine-tune

Each arm: fresh pretrained model + same data/seed/steps, evaluate accuracy vs position, and
also report the Step-0 valley depth the arm starts from. The headline question: does editing
the initialization prior (resid_alpha / anchors) beat both standard training and loss-weighting
on middle retrieval?
"""
import argparse
import os

import _bootstrap  # noqa: F401
from src import data, evaluate, plotting, train as train_mod, architecture, influence
from src.interventions import make_weight_fn
from src.utils import (load_config, apply_overrides, set_seed, pick_device,
                       load_model_tokenizer, read_jsonl, write_csv, write_json,
                       save_run_config, ensure_dir, repo_path)


def _load_dataset(cfg):
    path = os.path.join(repo_path(cfg["data"]["out_dir"]), data.dataset_filename(cfg))
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}\nRun scripts/01_generate_dataset.py first.")
    return read_jsonl(path)


def _acc_stats(by_bucket, lo=0.4, hi=0.6, edge=0.15):
    accs = [(r["mean_norm_pos"], r["accuracy"]) for r in by_bucket]
    mids = [a for p, a in accs if lo <= p <= hi]
    edges = [a for p, a in accs if p <= edge or p >= 1 - edge]
    return {
        "middle_acc": round(sum(a for _, a in accs if lo <= _ <= hi) / len(mids), 4) if mids else float("nan"),
        "avg_acc": round(sum(a for _, a in accs) / len(accs), 4),
        "worst_acc": round(min(a for _, a in accs), 4),
        "edge_acc": round(sum(edges) / len(edges), 4) if edges else float("nan"),
    }


def _train_eval(cfg, device, examples, weight_fn=None, residual_schedule=None):
    """Fresh model -> (optional init residual reshaping) -> fine-tune -> eval. Returns (stats, by_bucket)."""
    model, tokenizer, _ = load_model_tokenizer(cfg, device, for_training=True)
    handles = []
    if residual_schedule is not None:
        handles = architecture.apply_residual_scaling(model, residual_schedule)
    try:
        train_mod.train(model, tokenizer, examples, device, cfg, weight_fn=weight_fn,
                        seed=cfg["seed"], verbose=False)
        eval_res = evaluate.run_eval(model, tokenizer, examples, device, cfg, verbose=False)
    finally:
        architecture.remove_hooks(handles)
    del model
    return _acc_stats(eval_res["by_bucket"]), eval_res["by_bucket"]


def _run_all_arms(cfg, device, examples, arms, seed, tables):
    """Run every arm once for a given seed; returns {arm: stats}."""
    import copy
    cfg = copy.deepcopy(cfg)
    cfg["seed"] = seed
    set_seed(seed)

    residual_schedule = None
    if "resid_alpha" in arms:
        init_model, _, _ = load_model_tokenizer(cfg, device, for_training=False,
                                                init_random=True, seed=seed)
        search = architecture.search_residual_schedule(init_model, examples, device, cfg)
        residual_schedule = search["best"]["schedule"]
        write_json(os.path.join(tables, f"resid_schedule_search_seed{seed}.json"), search)
        del init_model

    anchored = None
    out = {}
    for arm in arms:
        if arm == "standard":
            stats, _ = _train_eval(cfg, device, examples)
        elif arm == "loss_edgefloor":
            stats, _ = _train_eval(cfg, device, examples, weight_fn=make_weight_fn(dict(cfg["intervention"])))
        elif arm == "resid_alpha":
            stats, _ = _train_eval(cfg, device, examples, residual_schedule=residual_schedule)
        elif arm == "anchors":
            if anchored is None:
                _, tok, _ = load_model_tokenizer(cfg, device, for_training=False)
                anchored = data.insert_anchors(examples, tok, int(cfg["architecture"]["anchor_interval"]))
            stats, _ = _train_eval(cfg, device, anchored)
        else:
            raise ValueError(f"unknown arm '{arm}'")
        out[arm] = stats
        print(f"[10] seed{seed} {arm}: middle={stats['middle_acc']} avg={stats['avg_acc']} "
              f"worst={stats['worst_acc']} edge={stats['edge_acc']}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=repo_path("configs", "default.yaml"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--seeds", type=str, default=None,
                    help="Comma-separated seeds (e.g. 0,1,2) for mean+/-std over arms.")
    args = ap.parse_args()

    cfg = load_config(args.config, smoke=args.smoke)
    cfg = apply_overrides(cfg, {"model": args.model})
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [cfg["seed"]]

    examples = _load_dataset(cfg)
    device = pick_device()
    arms = cfg["architecture"]["bakeoff_arms"]
    tables = ensure_dir(repo_path(cfg["sweep"]["out_tables"]))
    plots = ensure_dir(repo_path(cfg["sweep"]["out_plots"]))

    # seed -> {arm: stats}
    per_seed = {}
    for seed in seeds:
        print(f"\n[10] ===== SEED {seed} =====")
        per_seed[seed] = _run_all_arms(cfg, device, examples, arms, seed, tables)

    # Aggregate mean+/-std per arm across seeds.
    import numpy as np
    metrics = ["middle_acc", "avg_acc", "worst_acc", "edge_acc"]
    rows = []
    for arm in arms:
        row = {"arm": arm, "n_seeds": len(seeds)}
        for mname in metrics:
            vals = np.asarray([per_seed[s][arm][mname] for s in seeds], float)
            row[f"{mname}_mean"] = round(float(vals.mean()), 4)
            row[f"{mname}_std"] = round(float(vals.std(ddof=1)) if len(vals) > 1 else 0.0, 4)
        rows.append(row)

    # per-seed long table for transparency
    seed_rows = [{"seed": s, "arm": arm, **per_seed[s][arm]} for s in seeds for arm in arms]
    write_csv(os.path.join(tables, "cure_bakeoff_seeds.csv"), seed_rows,
              fieldnames=["seed", "arm", "middle_acc", "avg_acc", "worst_acc", "edge_acc"])
    write_csv(os.path.join(tables, "cure_bakeoff.csv"), rows,
              fieldnames=["arm", "n_seeds"] + [f"{m}_{s}" for m in metrics for s in ("mean", "std")])
    save_run_config(cfg, tables, "run_config_10_bakeoff.json")
    plotting.plot_sweep([{"name": r["arm"], "middle_acc": r["middle_acc_mean"],
                          "avg_acc": r["avg_acc_mean"], "worst_acc": r["worst_acc_mean"]} for r in rows],
                        os.path.join(plots, "cure_bakeoff_accuracy.png"),
                        f"Cure bake-off (mean of {len(seeds)} seeds): accuracy by method",
                        control_name="standard")

    ctrl = next((r for r in rows if r["arm"] == "standard"), None)
    print(f"\n[10] === Cure bake-off verdict (mean +/- std over {len(seeds)} seeds) ===")
    for r in rows:
        flag = ""
        if ctrl and r["arm"] != "standard":
            # win = mean beats control mean on middle, and avg/worst not worse than control mean
            if (r["middle_acc_mean"] > ctrl["middle_acc_mean"]
                    and r["avg_acc_mean"] >= ctrl["avg_acc_mean"]
                    and r["worst_acc_mean"] >= ctrl["worst_acc_mean"]):
                flag = "  <-- beats standard on all (mean)"
        init_fix = "  [INIT-TIME CURE]" if r["arm"] in ("resid_alpha", "anchors") else ""
        print(f"   {r['arm']:>16}: middle={r['middle_acc_mean']:.3f}+/-{r['middle_acc_std']:.3f} "
              f"avg={r['avg_acc_mean']:.3f}+/-{r['avg_acc_std']:.3f} "
              f"worst={r['worst_acc_mean']:.3f}{flag}{init_fix}")


if __name__ == "__main__":
    main()
