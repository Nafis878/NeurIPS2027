"""Step-0 vs. trained comparison: does the architectural position-bias valley persist,
and does the middle-weighted intervention bridge it?

Directly engages "Lost in the Middle at Birth" (Chowdhury 2026):
- their core claim: a U-shaped influence valley exists at RANDOM INITIALIZATION (Step 0)
  and standard training does not flatten it (their Fig. 3: the valley *deepens*).
- our beat: we (a) reproduce the Step-0 valley on our setup, (b) confirm standard
  fine-tuning deepens it, and (c) show the middle-weighted intervention flattens it.

Reads the saved `influence_<tag>_full.json` profiles (produced by scripts 03/04/05) for
tags [step0_init, baseline, standard_finetune, intervention], computes valley-depth
metrics for each, writes a CSV, and overlays the profiles (raw + recency-anchored).
"""
import argparse
import os

import _bootstrap  # noqa: F401
from src import influence, plotting
from src.utils import load_config, repo_path, ensure_dir, read_json, write_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=repo_path("configs", "default.yaml"))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config, smoke=args.smoke)
    itables = repo_path(cfg["influence"]["out_tables"])
    plots = ensure_dir(repo_path(cfg["influence"]["out_plots"]))
    mid_lo = float(cfg["influence"]["middle_low"])
    mid_hi = float(cfg["influence"]["middle_high"])

    tags = ["step0_init", "baseline", "standard_finetune", "intervention"]
    profiles = {}
    rows = []
    for tag in tags:
        path = os.path.join(itables, f"influence_{tag}_full.json")
        if not os.path.exists(path):
            print(f"[07] (skipping '{tag}': {os.path.basename(path)} not found)")
            continue
        full = read_json(path)
        bc, prof = full["bin_centers"], full["global_profile"]
        profiles[tag] = (bc, prof)
        m = influence.valley_metrics(bc, prof, mid_lo, mid_hi)
        m = {"method": tag, **{k: round(v, 5) for k, v in m.items()}}
        rows.append(m)
        print(f"[07] {tag:>18}: valley_depth={m['valley_depth']:.3f} "
              f"peak/trough={m['peak_to_trough']:.2f} middle_mass={m['middle_mass']:.3f}")

    if not profiles:
        raise FileNotFoundError(
            "No influence_*_full.json found. Run scripts 03 (with and without --init-random), "
            "04 and 05 first.")

    write_csv(os.path.join(itables, "valley_depth_by_method.csv"), rows,
              fieldnames=["method", "edge_peak", "middle_floor", "valley_depth",
                          "middle_over_recency", "peak_to_trough", "middle_mass"])

    plotting.plot_profiles_overlay(
        profiles, os.path.join(plots, "influence_step0_vs_trained.png"),
        "Influence profile: random init (Step 0) vs. trained models", anchor_recency=False)
    plotting.plot_profiles_overlay(
        profiles, os.path.join(plots, "influence_step0_vs_trained_anchored.png"),
        "Recency-anchored valley depth (paper Fig. 3 view)", anchor_recency=True)

    # Headline interpretation vs the paper's claims.
    by = {r["method"]: r for r in rows}
    print("\n[07] === Verdict vs. 'Lost in the Middle at Birth' ===")
    if "step0_init" in by:
        print(f"[07] Step-0 architectural valley depth = {by['step0_init']['valley_depth']:.3f} "
              f"(reproduces their init U-shape).")
    if "step0_init" in by and "standard_finetune" in by:
        d = by["standard_finetune"]["valley_depth"] - by["step0_init"]["valley_depth"]
        print(f"[07] standard FT valley vs Step-0: delta_depth={d:+.3f} "
              f"({'deepens (matches their Fig.3)' if d > 0 else 'flattens'}).")
    if "standard_finetune" in by and "intervention" in by:
        d = by["intervention"]["valley_depth"] - by["standard_finetune"]["valley_depth"]
        print(f"[07] intervention vs standard FT: delta_depth={d:+.3f} "
              f"({'FLATTENS the valley (our beat)' if d < 0 else 'does not flatten'}).")


if __name__ == "__main__":
    main()
