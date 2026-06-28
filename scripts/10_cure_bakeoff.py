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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=repo_path("configs", "default.yaml"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--model", type=str, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, smoke=args.smoke)
    cfg = apply_overrides(cfg, {"model": args.model})
    set_seed(cfg["seed"])

    examples = _load_dataset(cfg)
    device = pick_device()
    arms = cfg["architecture"]["bakeoff_arms"]
    tables = ensure_dir(repo_path(cfg["sweep"]["out_tables"]))
    plots = ensure_dir(repo_path(cfg["sweep"]["out_plots"]))

    # Pre-compute the residual schedule that flattens the Step-0 valley (if needed).
    residual_schedule = None
    if "resid_alpha" in arms:
        print("[10] searching residual schedule that flattens the Step-0 valley...")
        init_model, _, _ = load_model_tokenizer(cfg, device, for_training=False,
                                                init_random=True, seed=cfg["seed"])
        search = architecture.search_residual_schedule(init_model, examples, device, cfg)
        residual_schedule = search["best"]["schedule"]
        write_json(os.path.join(tables, "resid_schedule_search.json"), search)
        del init_model

    anchored = None
    rows, per_position = [], {}
    for arm in arms:
        print(f"\n[10] === arm: {arm} ===")
        if arm == "standard":
            stats, byb = _train_eval(cfg, device, examples)
        elif arm == "loss_edgefloor":
            wf = make_weight_fn(dict(cfg["intervention"]))
            stats, byb = _train_eval(cfg, device, examples, weight_fn=wf)
        elif arm == "resid_alpha":
            stats, byb = _train_eval(cfg, device, examples, residual_schedule=residual_schedule)
        elif arm == "anchors":
            if anchored is None:
                _, tok, _ = load_model_tokenizer(cfg, device, for_training=False)
                anchored = data.insert_anchors(examples, tok, int(cfg["architecture"]["anchor_interval"]))
            stats, byb = _train_eval(cfg, device, anchored)
        else:
            raise ValueError(f"unknown arm '{arm}'")
        row = {"arm": arm, **stats}
        rows.append(row)
        per_position[arm] = byb
        print(f"[10] {arm}: middle={stats['middle_acc']} avg={stats['avg_acc']} "
              f"worst={stats['worst_acc']} edge={stats['edge_acc']}")

    write_csv(os.path.join(tables, "cure_bakeoff.csv"), rows,
              fieldnames=["arm", "middle_acc", "avg_acc", "worst_acc", "edge_acc"])
    write_json(os.path.join(tables, "cure_bakeoff_per_position.json"), per_position)
    save_run_config(cfg, tables, "run_config_10_bakeoff.json")
    plotting.plot_sweep([{"name": r["arm"], **r} for r in rows],
                        os.path.join(plots, "cure_bakeoff_accuracy.png"),
                        "Cure bake-off: accuracy by method", control_name="standard")

    ctrl = next((r for r in rows if r["arm"] == "standard"), None)
    print("\n[10] === Cure bake-off verdict ===")
    for r in rows:
        flag = ""
        if ctrl and r["arm"] != "standard":
            if (r["middle_acc"] > ctrl["middle_acc"] and r["avg_acc"] >= ctrl["avg_acc"]
                    and r["worst_acc"] >= ctrl["worst_acc"]):
                flag = "  <-- beats standard on all"
        init_fix = "  [INIT-TIME CURE]" if r["arm"] in ("resid_alpha", "anchors") else ""
        print(f"   {r['arm']:>16}: middle={r['middle_acc']} avg={r['avg_acc']} "
              f"worst={r['worst_acc']}{flag}{init_fix}")


if __name__ == "__main__":
    main()
