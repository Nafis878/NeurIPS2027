"""Generate the synthetic position-controlled retrieval dataset."""
import argparse
import os

import _bootstrap  # noqa: F401
from src import data
from src.utils import (load_config, apply_overrides, set_seed, load_tokenizer,
                       write_jsonl, write_json, save_run_config, ensure_dir, repo_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=repo_path("configs", "default.yaml"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--examples-per-position", type=int, default=None)
    ap.add_argument("--context-length", type=int, default=None)
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, smoke=args.smoke)
    cfg = apply_overrides(cfg, {
        "examples_per_position": args.examples_per_position,
        "context_length": args.context_length,
        "model": args.model,
        "seed": args.seed,
    })
    set_seed(cfg["seed"])

    tokenizer, model_name = load_tokenizer(cfg)
    print(f"[01] Tokenizer: {model_name} | ctx={cfg['data']['context_length']} "
          f"| n/pos={cfg['data']['examples_per_position']} | smoke={args.smoke}")

    examples = data.generate_dataset(cfg, tokenizer, seed=cfg["seed"])
    out_dir = ensure_dir(repo_path(cfg["data"]["out_dir"]))
    fname = data.dataset_filename(cfg)
    path = os.path.join(out_dir, fname)
    write_jsonl(path, examples)

    meta = {
        "model_for_tokenizer": model_name,
        "n_examples": len(examples),
        "examples_per_position": cfg["data"]["examples_per_position"],
        "context_length": cfg["data"]["context_length"],
        "position_buckets": cfg["data"]["position_buckets"],
        "dataset_file": fname,
        "smoke": args.smoke,
    }
    write_json(os.path.join(out_dir, "dataset_meta.json"), meta)
    save_run_config(cfg, out_dir, "run_config_01.json")

    # quick sanity print of realized positions
    by_bucket = {}
    for ex in examples:
        by_bucket.setdefault(ex["position_bucket"], []).append(ex["norm_pos"])
    print("[01] realized mean normalized positions per bucket:")
    for b in sorted(by_bucket):
        ps = by_bucket[b]
        print(f"   bucket {b:>5}: mean_norm_pos={sum(ps)/len(ps):.3f}  n={len(ps)}")
    print(f"[01] Wrote {len(examples)} examples -> {path}")


if __name__ == "__main__":
    main()
