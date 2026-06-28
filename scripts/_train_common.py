"""Shared logic for the two fine-tuning experiments (standard and middle-weighted)."""
import os

import _bootstrap  # noqa: F401
from src import data, evaluate, influence, plotting, train as train_mod
from src.interventions import describe_weights
from src.utils import (pick_device, load_model_tokenizer, read_jsonl, write_jsonl,
                       write_csv, write_json, save_run_config, ensure_dir, repo_path)


def _load_dataset(cfg):
    path = os.path.join(repo_path(cfg["data"]["out_dir"]), data.dataset_filename(cfg))
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found: {path}\nRun scripts/01_generate_dataset.py first.")
    return read_jsonl(path)


def run_finetune_experiment(cfg, weight_fn, tag, loss_curve_name, loss_title):
    """Train, save loss curve + checkpoint, then re-eval + re-measure influence."""
    examples = _load_dataset(cfg)
    device = pick_device()
    model, tokenizer, model_name = load_model_tokenizer(cfg, device, for_training=True)

    wstats = describe_weights(examples, weight_fn)
    print(f"[{tag}] example weights: min={wstats['min']:.3f} max={wstats['max']:.3f} "
          f"mean={wstats['mean']:.3f}")

    history = train_mod.train(model, tokenizer, examples, device, cfg,
                              weight_fn=weight_fn, seed=cfg["seed"])

    tables = ensure_dir(repo_path(cfg["train"]["out_tables"]))
    plots = ensure_dir(repo_path(cfg["train"]["out_plots"]))
    write_csv(os.path.join(tables, f"loss_history_{tag}.csv"), history)
    plotting.plot_loss_curve(history, os.path.join(plots, loss_curve_name), loss_title)

    ckpt = ensure_dir(os.path.join(repo_path(cfg["train"]["ckpt_dir"]), tag))
    train_mod.save_checkpoint(model, tokenizer, ckpt)
    save_run_config(cfg, tables, f"run_config_train_{tag}.json")

    # Re-evaluate the fine-tuned model.
    eval_res = evaluate.run_eval(model, tokenizer, examples, device, cfg)
    etables = ensure_dir(repo_path(cfg["eval"]["out_tables"]))
    write_jsonl(os.path.join(etables, f"eval_{tag}_per_example.jsonl"), eval_res["per_example"])
    write_csv(os.path.join(etables, f"eval_{tag}_by_position.csv"), eval_res["by_bucket"])
    acc_name = ("accuracy_vs_position_standard_finetune.png" if tag == "standard_finetune"
                else "accuracy_vs_position_intervention.png")
    plotting.plot_accuracy_vs_position(
        eval_res["by_bucket"], os.path.join(plots, acc_name),
        f"Accuracy vs. position ({tag})")

    # Re-measure influence on the fine-tuned model.
    infl_res = influence.run_influence(model, tokenizer, examples, device, cfg)
    itables = ensure_dir(repo_path(cfg["influence"]["out_tables"]))
    write_jsonl(os.path.join(itables, f"influence_{tag}_per_example.jsonl"), infl_res["per_example"])
    csv_rows = [{k: v for k, v in r.items() if k != "binned_profile"} for r in infl_res["by_bucket"]]
    write_csv(os.path.join(itables, f"influence_{tag}_by_position.csv"), csv_rows)
    write_json(os.path.join(itables, f"influence_{tag}_full.json"), infl_res)
    infl_name = ("influence_vs_position_standard_finetune.png" if tag == "standard_finetune"
                 else "influence_vs_position_intervention.png")
    plotting.plot_influence_vs_position(
        infl_res["by_bucket"], os.path.join(plots, infl_name),
        f"Influence vs. position ({tag})")

    # Spearman influence vs accuracy for this method.
    infl_x = [r["influence_at_answer"] for r in infl_res["by_bucket"]]
    acc_by = {r["position_bucket"]: r["accuracy"] for r in eval_res["by_bucket"]}
    lp_by = {r["position_bucket"]: r["mean_logprob"] for r in eval_res["by_bucket"]}
    acc_y = [acc_by[r["position_bucket"]] for r in infl_res["by_bucket"]]
    lp_y = [lp_by[r["position_bucket"]] for r in infl_res["by_bucket"]]
    corr = {
        "spearman_influence_vs_accuracy": influence.spearman(infl_x, acc_y),
        "spearman_influence_vs_logprob": influence.spearman(infl_x, lp_y),
    }
    write_json(os.path.join(itables, f"spearman_{tag}.json"), corr)

    # Console summary with the headline numbers.
    accs = eval_res["by_bucket"]
    overall = sum(r["exact_match"] for r in eval_res["per_example"]) / len(eval_res["per_example"])
    middle = [r["accuracy"] for r in accs if 0.4 <= r["mean_norm_pos"] <= 0.6]
    worst = min(r["accuracy"] for r in accs)
    print(f"[{tag}] model={model_name} overall_acc={overall:.3f} "
          f"middle_acc={(sum(middle)/len(middle) if middle else float('nan')):.3f} "
          f"worst_acc={worst:.3f}")
    ptr = [r["peak_to_trough"] for r in infl_res["by_bucket"]]
    print(f"[{tag}] mean per-bucket peak/trough={sum(ptr)/len(ptr):.2f} "
          f"global middle mass={infl_res['global_middle_mass']:.3f}")
    return {"history": history, "eval": eval_res, "influence": infl_res, "corr": corr}
