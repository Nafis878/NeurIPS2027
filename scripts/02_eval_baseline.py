"""Experiment 1: evaluate pretrained model retrieval accuracy vs. position."""
import argparse
import os

import _bootstrap  # noqa: F401
from src import data, evaluate
from src.utils import (load_config, apply_overrides, set_seed, pick_device,
                       load_model_tokenizer, read_jsonl, write_jsonl, write_csv,
                       save_run_config, ensure_dir, repo_path)
from src import plotting


def _load_dataset(cfg):
    path = os.path.join(repo_path(cfg["data"]["out_dir"]), data.dataset_filename(cfg))
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found: {path}\nRun scripts/01_generate_dataset.py first "
            f"(with matching --smoke / --context-length / --examples-per-position).")
    return read_jsonl(path), path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=repo_path("configs", "default.yaml"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--tag", type=str, default="baseline")
    ap.add_argument("--checkpoint", type=str, default=None,
                    help="Optional path to a fine-tuned checkpoint to evaluate instead.")
    args = ap.parse_args()

    cfg = load_config(args.config, smoke=args.smoke)
    cfg = apply_overrides(cfg, {"model": args.model})
    set_seed(cfg["seed"])

    examples, ds_path = _load_dataset(cfg)
    print(f"[02] Loaded {len(examples)} examples from {ds_path}")

    device = pick_device()
    if args.checkpoint:
        cfg = dict(cfg)
        cfg["model"] = dict(cfg["model"], name=args.checkpoint, fallback=None)
    model, tokenizer, model_name = load_model_tokenizer(cfg, device, for_training=False)

    result = evaluate.run_eval(model, tokenizer, examples, device, cfg)

    tables = ensure_dir(repo_path(cfg["eval"]["out_tables"]))
    plots = ensure_dir(repo_path(cfg["eval"]["out_plots"]))
    write_jsonl(os.path.join(tables, f"eval_{args.tag}_per_example.jsonl"), result["per_example"])
    write_csv(os.path.join(tables, f"eval_{args.tag}_by_position.csv"), result["by_bucket"])
    save_run_config(cfg, tables, f"run_config_02_{args.tag}.json")

    title_map = {
        "baseline": "Baseline retrieval accuracy vs. position",
        "standard_finetune": "Accuracy vs. position (standard fine-tune)",
        "intervention": "Accuracy vs. position (middle-weighted)",
    }
    fname_map = {
        "baseline": "accuracy_vs_position_baseline.png",
        "standard_finetune": "accuracy_vs_position_standard_finetune.png",
        "intervention": "accuracy_vs_position_intervention.png",
    }
    plotting.plot_accuracy_vs_position(
        result["by_bucket"],
        os.path.join(plots, fname_map.get(args.tag, f"accuracy_vs_position_{args.tag}.png")),
        title_map.get(args.tag, f"Accuracy vs position ({args.tag})"))

    overall_acc = sum(r["exact_match"] for r in result["per_example"]) / len(result["per_example"])
    print(f"[02] model={model_name} overall EM accuracy={overall_acc:.3f}")
    print("[02] accuracy by bucket:")
    for r in result["by_bucket"]:
        print(f"   pos~{r['mean_norm_pos']:.2f}  acc={r['accuracy']:.3f}  "
              f"logprob={r['mean_logprob']:.3f}  n={r['n']}")


if __name__ == "__main__":
    main()
