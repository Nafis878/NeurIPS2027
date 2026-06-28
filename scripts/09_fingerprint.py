"""Claim 1 — "Legible at birth": the random-init influence fingerprint predicts the
TRAINED model's per-position retrieval accuracy.

This is the result the target paper explicitly disclaims ("we do not claim the Jacobian
norm measures retrieval accuracy"). We show the stronger thing: the Step-0 (untrained,
random-weight) influence profile rank-correlates with where the *fully fine-tuned* model
succeeds and fails at retrieval -- i.e. lost-in-the-middle is forecastable before training.

Fingerprint definition: the Step-0 GLOBAL per-token influence profile (U-shaped: primacy
tail + recency anchor), interpolated at each evaluation bucket's normalized position. (The
fact-location `influence_at_answer` is primacy-only and is reported too, for contrast.)

Modes:
  default            compute the fingerprint for the current tables, write fingerprint_<label>.csv
  --depth-trend      aggregate all fingerprint_*.csv into depth_trend.csv/png (the depth law)
  --seeds 0,1,2      re-run Step-0 + standard FT + fingerprint for each seed and report the
                     architectural-region Spearman as mean +/- std (statistical robustness)
"""
import argparse
import csv
import glob
import os

import numpy as np

import _bootstrap  # noqa: F401
from src import data, evaluate, influence, plotting, train as train_mod
from src.utils import (load_config, apply_overrides, set_seed, pick_device,
                       load_model_tokenizer, read_jsonl, repo_path, ensure_dir,
                       read_json, write_csv)


def _read_eval_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return [{k: (float(v) if v not in ("", None) else v) for k, v in r.items()}
                for r in csv.DictReader(fh)]


def _num_layers(model_name):
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_name)
        return int(getattr(cfg, "num_hidden_layers", getattr(cfg, "n_layer", -1)))
    except Exception:  # noqa: BLE001
        return -1


def _pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def run_single(cfg, label, step0_tag, trained_tag):
    itables = repo_path(cfg["influence"]["out_tables"])
    etables = repo_path(cfg["eval"]["out_tables"])
    plots = ensure_dir(repo_path(cfg["eval"]["out_plots"]))

    step0_path = os.path.join(itables, f"influence_{step0_tag}_full.json")
    eval_path = os.path.join(etables, f"eval_{trained_tag}_by_position.csv")
    for p in (step0_path, eval_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing {p}. Run `03 --init-random` (Step-0) and `04` (standard FT) first.")

    step0 = read_json(step0_path)
    bin_centers = np.asarray(step0["bin_centers"], float)
    global_profile = np.asarray(step0["global_profile"], float)
    # also the fact-location (primacy-only) influence per bucket, for contrast
    s0_by_bucket = {r["position_bucket"]: r for r in _read_csv_rows(
        os.path.join(itables, f"influence_{step0_tag}_by_position.csv"))}

    eval_rows = _read_eval_csv(eval_path)
    rows = []
    for r in eval_rows:
        x = float(r["mean_norm_pos"])
        fp = float(np.interp(x, bin_centers, global_profile))  # fingerprint at this position
        s0b = s0_by_bucket.get(r["position_bucket"], {})
        rows.append({
            "position_bucket": r["position_bucket"],
            "mean_norm_pos": round(x, 4),
            "fingerprint": round(fp, 6),
            "step0_influence_at_answer": round(float(s0b.get("influence_at_answer", float("nan"))), 6),
            "trained_accuracy": round(float(r["accuracy"]), 4),
            "trained_logprob": round(float(r["mean_logprob"]), 4),
        })

    fp_x = [r["fingerprint"] for r in rows]
    acc_y = [r["trained_accuracy"] for r in rows]
    lp_y = [r["trained_logprob"] for r in rows]
    pos = [r["mean_norm_pos"] for r in rows]

    # Decomposition: the Step-0 fingerprint is primacy-only (no distance-decay at random init),
    # so it forecasts the ARCHITECTURAL half (primacy + middle collapse, x <= region_thr).
    # The recency recovery at x -> 1 is a LEARNED effect, absent at birth.
    region_thr = float(cfg.get("fingerprint", {}).get("region_threshold", 0.7))
    arch = [(f, a) for f, a, p in zip(fp_x, acc_y, pos) if p <= region_thr]
    arch_fp = [f for f, _ in arch]
    arch_acc = [a for _, a in arch]
    s_full = influence.spearman(fp_x, acc_y)
    s_arch = influence.spearman(arch_fp, arch_acc)
    # recency emergence: did accuracy recover at x>0.8 where the fingerprint stays flat-low?
    late_acc = [a for a, p in zip(acc_y, pos) if p >= 0.8]
    mid_acc = [a for a, p in zip(acc_y, pos) if 0.4 <= p <= 0.6]
    recency_gap = (round(sum(late_acc) / len(late_acc) - sum(mid_acc) / len(mid_acc), 4)
                   if late_acc and mid_acc else float("nan"))

    corr = {
        "label": label,
        "model_layers": _num_layers(_model_name_from_runcfg(itables, step0_tag)),
        "metric": step0.get("metric", "answer_grad"),
        "spearman_arch_region": s_arch["rho"],      # headline: birthright (x<=thr)
        "p_arch_region": s_arch["p"],
        "spearman_full_range": s_full["rho"],        # diluted by learned recency
        "spearman_fingerprint_vs_logprob": influence.spearman(fp_x, lp_y)["rho"],
        "pearson_arch_region": _pearson(arch_fp, arch_acc),
        "recency_gap_late_minus_mid": recency_gap,   # >0 => learned recency wall
        "step0_valley_depth": influence.valley_metrics(
            bin_centers, global_profile,
            float(cfg["influence"]["middle_low"]), float(cfg["influence"]["middle_high"]))["valley_depth"],
        "trained_middle_acc": _middle(acc_y, pos),
        "n_buckets": len(rows),
        "region_threshold": region_thr,
    }

    tables = ensure_dir(etables)
    write_csv(os.path.join(tables, f"fingerprint_{label}.csv"), rows)
    write_csv(os.path.join(tables, f"fingerprint_{label}_summary.csv"), [corr])
    plotting.plot_fingerprint_scatter(
        rows, os.path.join(plots, f"fingerprint_scatter_{label}.png"),
        f"Step-0 fingerprint vs. trained accuracy ({label})", region_thr=region_thr)

    print(f"[09] {label}: ARCHITECTURAL region (x<={region_thr}) "
          f"Spearman(fingerprint, trained acc) rho={s_arch['rho']:.3f} p={s_arch['p']:.3f}  "
          f"<-- headline (birthright predicts trained failure)")
    print(f"[09] {label}: full-range rho={s_full['rho']:.3f} (diluted by learned recency); "
          f"recency_gap(late-mid)={recency_gap} (>0 => recency is learned, not at birth)")
    print(f"[09] {label}: step0 valley_depth={corr['step0_valley_depth']:.3f} "
          f"trained middle acc={corr['trained_middle_acc']:.3f} layers={corr['model_layers']}")
    return corr


def _read_csv_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return [{k: (float(v) if _isnum(v) else v) for k, v in r.items()}
                for r in csv.DictReader(fh)]


def _isnum(v):
    try:
        float(v); return True
    except (TypeError, ValueError):
        return False


def _model_name_from_runcfg(itables, step0_tag):
    p = os.path.join(itables, f"run_config_03_{step0_tag}.json")
    if os.path.exists(p):
        return read_json(p)["config"]["model"]["name"]
    return "EleutherAI/pythia-70m"


def _middle(acc, pos, lo=0.4, hi=0.6):
    vals = [a for a, p in zip(acc, pos) if lo <= p <= hi]
    return round(sum(vals) / len(vals), 4) if vals else float("nan")


def _fingerprint_one_seed(cfg, device, examples, seed, region_thr, metric):
    """Re-run Step-0 influence + standard FT + eval for one seed and return the fingerprint stats."""
    set_seed(seed)
    cfg = dict(cfg)
    cfg["seed"] = seed

    # Step-0 (random-init) influence -> global profile (the fingerprint source)
    m0, _, _ = load_model_tokenizer(cfg, device, for_training=False, init_random=True, seed=seed)
    inf0 = influence.run_influence(m0, None, examples, device, cfg, verbose=False, metric=metric)
    del m0
    bc = np.asarray(inf0["bin_centers"], float)
    gp = np.asarray(inf0["global_profile"], float)

    # Standard fine-tune -> trained accuracy by position
    m, tok, _ = load_model_tokenizer(cfg, device, for_training=True)
    train_mod.train(m, tok, examples, device, cfg, weight_fn=None, seed=seed, verbose=False)
    ev = evaluate.run_eval(m, tok, examples, device, cfg, verbose=False)
    del m

    pos = [r["mean_norm_pos"] for r in ev["by_bucket"]]
    acc = [r["accuracy"] for r in ev["by_bucket"]]
    fp = [float(np.interp(p, bc, gp)) for p in pos]
    arch = [(f, a) for f, a, p in zip(fp, acc, pos) if p <= region_thr]
    s_arch = influence.spearman([f for f, _ in arch], [a for _, a in arch])
    s_full = influence.spearman(fp, acc)
    late = [a for a, p in zip(acc, pos) if p >= 0.8]
    mid = [a for a, p in zip(acc, pos) if 0.4 <= p <= 0.6]
    recency = (sum(late) / len(late) - sum(mid) / len(mid)) if late and mid else float("nan")
    vd = influence.valley_metrics(bc, gp, float(cfg["influence"]["middle_low"]),
                                  float(cfg["influence"]["middle_high"]))["valley_depth"]
    return {"seed": seed, "spearman_arch_region": round(s_arch["rho"], 4),
            "p_arch_region": round(s_arch["p"], 4), "spearman_full_range": round(s_full["rho"], 4),
            "recency_gap": round(recency, 4), "step0_valley_depth": round(vd, 4),
            "trained_middle_acc": round(_middle(acc, pos), 4)}


def run_seeds(cfg, device, examples, seeds, label, region_thr, metric):
    rows = [_fingerprint_one_seed(cfg, device, examples, s, region_thr, metric) for s in seeds]
    tables = ensure_dir(repo_path(cfg["eval"]["out_tables"]))
    write_csv(os.path.join(tables, f"fingerprint_seeds_{label}.csv"), rows)

    def _ms(key):
        vals = [r[key] for r in rows if r[key] == r[key]]  # drop NaN
        arr = np.asarray(vals, float)
        return (round(float(arr.mean()), 4), round(float(arr.std(ddof=1)), 4) if len(arr) > 1 else 0.0)

    arch_m, arch_s = _ms("spearman_arch_region")
    full_m, full_s = _ms("spearman_full_range")
    rec_m, rec_s = _ms("recency_gap")
    summary = [{"label": label, "n_seeds": len(seeds), "metric": metric,
                "spearman_arch_mean": arch_m, "spearman_arch_std": arch_s,
                "spearman_full_mean": full_m, "spearman_full_std": full_s,
                "recency_gap_mean": rec_m, "recency_gap_std": rec_s,
                "win_rate_arch_gt_0": round(np.mean([r["spearman_arch_region"] > 0 for r in rows]), 3)}]
    write_csv(os.path.join(tables, f"fingerprint_multiseed_{label}.csv"), summary)
    print(f"\n[09] MULTI-SEED fingerprint ({label}, n={len(seeds)}):")
    for r in rows:
        print(f"   seed {r['seed']}: arch_rho={r['spearman_arch_region']:.3f} "
              f"(p={r['p_arch_region']:.3f}) full={r['spearman_full_range']:.3f} "
              f"recency_gap={r['recency_gap']:.3f}")
    print(f"[09] arch-region Spearman = {arch_m:.3f} +/- {arch_s:.3f}  "
          f"(full-range {full_m:.3f} +/- {full_s:.3f}); recency_gap {rec_m:.3f} +/- {rec_s:.3f}")
    print(f"[09] => birthright legible: {summary[0]['win_rate_arch_gt_0']*100:.0f}% of seeds have arch_rho>0")
    return summary


def run_depth_trend(cfg):
    tables = repo_path(cfg["eval"]["out_tables"])
    plots = ensure_dir(repo_path(cfg["eval"]["out_plots"]))
    summaries = []
    for p in sorted(glob.glob(os.path.join(tables, "fingerprint_*_summary.csv"))):
        rows = _read_csv_rows(p)
        if rows:
            summaries.append(rows[0])
    if not summaries:
        raise FileNotFoundError("No fingerprint_*_summary.csv found; run the per-model fingerprints first.")
    summaries.sort(key=lambda r: float(r.get("model_layers", 0)))
    write_csv(os.path.join(tables, "depth_trend.csv"), summaries,
              fieldnames=["label", "model_layers", "step0_valley_depth", "trained_middle_acc",
                          "spearman_arch_region", "recency_gap_late_minus_mid"])
    plotting.plot_depth_trend(summaries, os.path.join(plots, "depth_trend.png"),
                              "Depth law: birth-valley & trained middle accuracy vs. depth")
    print("[09] depth trend:")
    for s in summaries:
        print(f"   {s['label']:>8} (L={int(float(s['model_layers']))}): "
              f"valley_depth={float(s['step0_valley_depth']):.3f} "
              f"middle_acc={float(s['trained_middle_acc']):.3f} "
              f"arch_rho={float(s['spearman_arch_region']):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=repo_path("configs", "default.yaml"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--label", type=str, default="pythia-70m",
                    help="Short label for this model's fingerprint outputs (e.g. 70m/160m/410m).")
    ap.add_argument("--step0-tag", type=str, default="step0_init")
    ap.add_argument("--trained-tag", type=str, default="standard_finetune")
    ap.add_argument("--depth-trend", action="store_true",
                    help="Aggregate all fingerprint_*_summary.csv into the depth-law trend.")
    ap.add_argument("--seeds", type=str, default=None,
                    help="Comma-separated seeds (e.g. 0,1,2) to compute the fingerprint as "
                         "mean+/-std by re-running Step-0 + standard FT per seed.")
    ap.add_argument("--model", type=str, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, smoke=args.smoke)
    cfg = apply_overrides(cfg, {"model": args.model})
    if args.depth_trend:
        run_depth_trend(cfg)
    elif args.seeds:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
        device = pick_device()
        examples = read_jsonl(os.path.join(repo_path(cfg["data"]["out_dir"]),
                                           data.dataset_filename(cfg)))
        region_thr = float(cfg.get("fingerprint", {}).get("region_threshold", 0.7))
        metric = cfg["influence"].get("metric", "answer_grad")
        run_seeds(cfg, device, examples, seeds, args.label, region_thr, metric)
    else:
        run_single(cfg, args.label, args.step0_tag, args.trained_tag)


if __name__ == "__main__":
    main()
