"""Aggregate all metrics into comparison plots + outputs/results_summary.md."""
import argparse
import csv
import json
import os

import _bootstrap  # noqa: F401
from src import plotting
from src.utils import load_config, repo_path, ensure_dir


def _read_csv(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for r in rows:
        out.append({k: (float(v) if v not in (None, "") and _isnum(v) else v)
                    for k, v in r.items()})
    return out


def _isnum(v):
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def _read_json(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _fmt(x, nd=3):
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=repo_path("configs", "default.yaml"))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config, smoke=args.smoke)
    tables = repo_path(cfg["eval"]["out_tables"])
    itables = repo_path(cfg["influence"]["out_tables"])
    plots = ensure_dir(repo_path(cfg["eval"]["out_plots"]))

    tags = ["baseline", "standard_finetune", "intervention"]
    short = {"baseline": "baseline", "standard_finetune": "standard", "intervention": "intervention"}

    acc = {t: _read_csv(os.path.join(tables, f"eval_{t}_by_position.csv")) for t in tags}
    infl = {t: _read_csv(os.path.join(itables, f"influence_{t}_by_position.csv")) for t in tags}
    spear = {t: _read_json(os.path.join(itables, f"spearman_{t}.json")) for t in tags}
    if spear["baseline"] is None:
        spear["baseline"] = _read_json(os.path.join(tables, "spearman_baseline.json"))

    # Comparison plots (only methods that exist).
    acc_methods = {short[t]: acc[t] for t in tags if acc[t]}
    infl_methods = {short[t]: infl[t] for t in tags if infl[t]}
    if acc_methods:
        plotting.plot_comparison(acc_methods, os.path.join(plots, "comparison_accuracy_all_methods.png"),
                                 "Accuracy vs. position: all methods", "accuracy", "Exact-match accuracy")
    if infl_methods:
        plotting.plot_comparison(infl_methods, os.path.join(plots, "comparison_influence_all_methods.png"),
                                 "Influence vs. position: all methods", "influence_at_answer",
                                 "Influence at answer span")

    # Build the markdown report.
    lines = []
    lines.append("# Results Summary\n")
    lines.append(f"- Config: `{cfg['_meta']['config_path']}` (smoke={cfg['_meta']['smoke']})")
    lines.append(f"- Model: `{cfg['model']['name']}` (fallback `{cfg['model'].get('fallback')}`)")
    lines.append(f"- Context length: {cfg['data']['context_length']} | "
                 f"examples/position: {cfg['data']['examples_per_position']} | "
                 f"position buckets: {len(cfg['data']['position_buckets'])}\n")

    def _overall(rows, key):
        vals = [r[key] for r in rows if _isnum(r.get(key))]
        return sum(vals) / len(vals) if vals else float("nan")

    def _middle(rows, key):
        vals = [r[key] for r in rows if 0.4 <= float(r["mean_norm_pos"]) <= 0.6]
        return sum(vals) / len(vals) if vals else float("nan")

    def _worst(rows, key):
        vals = [r[key] for r in rows if _isnum(r.get(key))]
        return min(vals) if vals else float("nan")

    lines.append("## Headline metrics\n")
    lines.append("| Method | Avg acc | Middle acc | Worst acc | Avg influence@answer | Spearman(infl,acc) rho |")
    lines.append("|---|---|---|---|---|---|")
    for t in tags:
        if not acc[t]:
            continue
        a = acc[t]
        avg = _overall(a, "accuracy")
        mid = _middle(a, "accuracy")
        wrs = _worst(a, "accuracy")
        infl_avg = _overall(infl[t], "influence_at_answer") if infl[t] else float("nan")
        rho = spear[t]["spearman_influence_vs_accuracy"]["rho"] if spear[t] else float("nan")
        lines.append(f"| {t} | {_fmt(avg)} | {_fmt(mid)} | {_fmt(wrs)} | "
                     f"{_fmt(infl_avg,5)} | {_fmt(rho)} |")
    lines.append("")

    # Spearman detail (Experiment 2 core question).
    lines.append("## Experiment 2 — does influence predict accuracy?\n")
    for t in tags:
        if spear[t]:
            sa = spear[t]["spearman_influence_vs_accuracy"]
            sl = spear[t]["spearman_influence_vs_logprob"]
            lines.append(f"- **{t}**: Spearman(influence, EM accuracy) rho={_fmt(sa['rho'])} "
                         f"p={_fmt(sa['p'])} (n={sa['n']}); "
                         f"vs log-prob rho={_fmt(sl['rho'])} p={_fmt(sl['p'])}")
    lines.append("")

    # Per-position tables.
    for t in tags:
        if not acc[t]:
            continue
        lines.append(f"### {t}: accuracy & influence by position\n")
        lines.append("| norm_pos | n | EM acc | mean logprob | influence@answer | peak/trough | middle mass |")
        lines.append("|---|---|---|---|---|---|---|")
        infl_by = {r["position_bucket"]: r for r in (infl[t] or [])}
        for r in acc[t]:
            ir = infl_by.get(r["position_bucket"], {})
            lines.append(f"| {_fmt(r['mean_norm_pos'],2)} | {int(r['n'])} | {_fmt(r['accuracy'])} | "
                         f"{_fmt(r['mean_logprob'])} | {_fmt(ir.get('influence_at_answer'),5)} | "
                         f"{_fmt(ir.get('peak_to_trough'),2)} | {_fmt(ir.get('middle_influence_mass'))} |")
        lines.append("")

    # Plots index.
    lines.append("## Figures (outputs/plots/)\n")
    for fn in ["accuracy_vs_position_baseline.png", "influence_vs_position_baseline.png",
               "loss_curve_standard_finetune.png", "accuracy_vs_position_standard_finetune.png",
               "influence_vs_position_standard_finetune.png",
               "accuracy_vs_position_intervention.png", "influence_vs_position_intervention.png",
               "comparison_accuracy_all_methods.png", "comparison_influence_all_methods.png"]:
        exists = os.path.exists(os.path.join(plots, fn))
        lines.append(f"- {'[x]' if exists else '[ ]'} `{fn}`")
    lines.append("")

    # Automated verdict.
    lines.append("## Automated verdict\n")
    base_rho = (spear["baseline"]["spearman_influence_vs_accuracy"]["rho"]
                if spear["baseline"] else None)
    verdict = []
    if base_rho is not None and base_rho == base_rho:
        if base_rho >= 0.4:
            verdict.append(f"Influence positively predicts accuracy (baseline rho={_fmt(base_rho)}).")
        elif base_rho <= -0.4:
            verdict.append(f"Influence negatively predicts accuracy (baseline rho={_fmt(base_rho)}).")
        else:
            verdict.append(f"Weak/unclear influence-accuracy link (baseline rho={_fmt(base_rho)}).")
    if acc["intervention"] and acc["standard_finetune"]:
        mid_i = _middle(acc["intervention"], "accuracy")
        mid_s = _middle(acc["standard_finetune"], "accuracy")
        verdict.append(f"Middle accuracy: intervention={_fmt(mid_i)} vs standard={_fmt(mid_s)} "
                       f"(delta={_fmt(mid_i - mid_s)}).")
    if not verdict:
        verdict.append("Insufficient stages completed for a verdict; run experiments 1-4.")
    for v in verdict:
        lines.append(f"- {v}")
    lines.append("")

    out_path = repo_path("outputs", "results_summary.md")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"[06] Wrote {out_path}")
    for v in verdict:
        print(f"   verdict: {v}")


if __name__ == "__main__":
    main()
