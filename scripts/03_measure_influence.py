"""Experiment 2: influence proxy vs. position + Spearman vs. accuracy."""
import argparse
import os

import _bootstrap  # noqa: F401
from src import data, influence, plotting
from src.utils import (load_config, apply_overrides, set_seed, pick_device,
                       load_model_tokenizer, read_jsonl, write_jsonl, write_csv,
                       write_json, save_run_config, ensure_dir, repo_path)


def _load_dataset(cfg):
    path = os.path.join(repo_path(cfg["data"]["out_dir"]), data.dataset_filename(cfg))
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found: {path}\nRun scripts/01_generate_dataset.py first.")
    return read_jsonl(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=repo_path("configs", "default.yaml"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--tag", type=str, default="baseline")
    ap.add_argument("--checkpoint", type=str, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, smoke=args.smoke)
    cfg = apply_overrides(cfg, {"model": args.model})
    set_seed(cfg["seed"])

    examples = _load_dataset(cfg)
    device = pick_device()
    if args.checkpoint:
        cfg = dict(cfg)
        cfg["model"] = dict(cfg["model"], name=args.checkpoint, fallback=None)
    model, tokenizer, model_name = load_model_tokenizer(cfg, device, for_training=False)

    result = influence.run_influence(model, tokenizer, examples, device, cfg)

    tables = ensure_dir(repo_path(cfg["influence"]["out_tables"]))
    plots = ensure_dir(repo_path(cfg["influence"]["out_plots"]))
    write_jsonl(os.path.join(tables, f"influence_{args.tag}_per_example.jsonl"),
                result["per_example"])
    # strip large per-bucket profiles out of the CSV (kept in JSON)
    csv_rows = [{k: v for k, v in r.items() if k != "binned_profile"} for r in result["by_bucket"]]
    write_csv(os.path.join(tables, f"influence_{args.tag}_by_position.csv"), csv_rows)
    write_json(os.path.join(tables, f"influence_{args.tag}_full.json"), result)
    save_run_config(cfg, tables, f"run_config_03_{args.tag}.json")

    # Spearman vs. accuracy (and vs. logprob) if the eval table exists.
    eval_csv = os.path.join(repo_path(cfg["eval"]["out_tables"]), f"eval_{args.tag}_by_position.csv")
    corr = {}
    if os.path.exists(eval_csv) and os.path.getsize(eval_csv) > 0:
        import csv as _csv
        with open(eval_csv, newline="", encoding="utf-8") as fh:
            eval_rows = list(_csv.DictReader(fh))
        eval_by_bucket = {float(r["position_bucket"]): r for r in eval_rows}
        infl_x, acc_y, lp_y = [], [], []
        for r in result["by_bucket"]:
            b = r["position_bucket"]
            if b in eval_by_bucket:
                infl_x.append(r["influence_at_answer"])
                acc_y.append(float(eval_by_bucket[b]["accuracy"]))
                lp_y.append(float(eval_by_bucket[b]["mean_logprob"]))
        corr = {
            "spearman_influence_vs_accuracy": influence.spearman(infl_x, acc_y),
            "spearman_influence_vs_logprob": influence.spearman(infl_x, lp_y),
        }
        write_json(os.path.join(tables, f"spearman_{args.tag}.json"), corr)
    else:
        print(f"[03] (no eval table at {eval_csv}; run 02 to enable Spearman)")

    title_map = {
        "baseline": "Influence vs. position (baseline)",
        "standard_finetune": "Influence vs. position (standard fine-tune)",
        "intervention": "Influence vs. position (middle-weighted)",
    }
    fname_map = {
        "baseline": "influence_vs_position_baseline.png",
        "standard_finetune": "influence_vs_position_standard_finetune.png",
        "intervention": "influence_vs_position_intervention.png",
    }
    plotting.plot_influence_vs_position(
        result["by_bucket"],
        os.path.join(plots, fname_map.get(args.tag, f"influence_vs_position_{args.tag}.png")),
        title_map.get(args.tag, f"Influence vs position ({args.tag})"))
    plotting.plot_global_profile(
        result["bin_centers"], result["global_profile"],
        os.path.join(plots, f"influence_global_profile_{args.tag}.png"),
        f"Global per-token influence profile ({args.tag})")

    print(f"[03] model={model_name} global peak/trough={result['global_peak_to_trough']:.2f} "
          f"global middle mass={result['global_middle_mass']:.3f}")
    if corr:
        s = corr["spearman_influence_vs_accuracy"]
        sl = corr["spearman_influence_vs_logprob"]
        print(f"[03] Spearman influence vs accuracy: rho={s['rho']:.3f} p={s['p']:.3f} (n={s['n']})")
        print(f"[03] Spearman influence vs logprob:  rho={sl['rho']:.3f} p={sl['p']:.3f}")


if __name__ == "__main__":
    main()
