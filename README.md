# Position-Bias / Middle-Context Retrieval (NeurIPS 2027 — Phase 1)

Controlled experiment testing whether the **initialization-time geometric position-bias
prior** in causal-residual transformers constrains **middle-context retrieval**, and whether
a simple **middle-weighted training intervention** improves middle-position accuracy.

**Phase-1 question:** *Does position-wise influence predict retrieval accuracy, and can a
simple intervention improve middle-position retrieval?*

The whole pipeline is small, seeded, and reproducible. It is designed to be **run on a Google
Colab T4 GPU** (recommended) but degrades gracefully to CPU for smoke testing.

---

## What it does

A synthetic, *digit-free* distractor context has exactly one planted fact
(`The access code for Project Vega is 73914.`) inserted at a controlled normalized token
position, followed by a question and the short exact answer. Because the only number in the
whole context is the answer code, exact-match retrieval is a clean signal.

| Stage | Script | Output |
|---|---|---|
| 1. Generate dataset | `01_generate_dataset.py` | `data/synthetic/dataset_ctx{L}_n{N}.jsonl` |
| 2. Baseline accuracy vs position | `02_eval_baseline.py` | `accuracy_vs_position_baseline.png` |
| 3. Influence proxy vs position + Spearman | `03_measure_influence.py` | `influence_vs_position_baseline.png` |
| 4. Standard fine-tune | `04_train_standard.py` | `loss_curve_standard_finetune.png`, `*_standard_finetune.png` |
| 5. Middle-weighted intervention | `05_train_middle_weighted.py` | `*_intervention.png` |
| 6. Report + comparisons | `06_make_report.py` | `comparison_*.png`, `outputs/results_summary.md` |
| 7. Step-0 vs trained valley | `07_step0_compare.py` | `influence_step0_vs_trained*.png`, `valley_depth_by_method.csv` |
| 8. Weighting sweep (β/σ/edge-floor) | `08_sweep_intervention.py` | `sweep_intervention.csv`, `sweep_intervention_accuracy.png` |
| 9. Fingerprint (Claim 1) | `09_fingerprint.py` | `fingerprint_<label>.csv`, `fingerprint_scatter_*.png`, `depth_trend.*` |
| 10. Cure bake-off (Claim 2) | `10_cure_bakeoff.py` | `cure_bakeoff.csv`, `cure_bakeoff_accuracy.png` |

**NeurIPS novelty program — "Born Lost, Born Legible, Born Curable":** stages 9–10 go beyond the
paper's stated future work. **N1 (legible):** the random-init influence *fingerprint* forecasts the
trained model's per-position accuracy over the architectural region (Pythia-70m: Spearman 0.85,
p=0.016), and the recency half is shown to be *learned*, not present at birth. **N2 (curable):** edit
the initialization prior (residual-α reshaping and/or distributed anchor registers) and test whether
it beats training-time fixes. See [docs/BEATING_THE_PAPER.md](docs/BEATING_THE_PAPER.md).

This project is positioned to **beat** "Lost in the Middle at Birth" (Chowdhury, arXiv:2603.10123)
by answering its stated open question — see [docs/BEATING_THE_PAPER.md](docs/BEATING_THE_PAPER.md).
Stage 7 reproduces the **Step-0 architectural valley** on a randomly-initialized model and shows
that standard fine-tuning *deepens* it (their Fig. 3) while our middle-weighted intervention
*flattens* it.

Metrics: **exact-match (EM)** is primary; we also log a continuous sensitivity signal
(**answer-token mean log-prob and rank**) so the influence↔accuracy Spearman stays informative
even when a 70M model scores ~0 EM.

---

## Run on Google Colab (T4 GPU) — recommended

1. **Runtime → Change runtime type → T4 GPU.**
2. Get the code onto Colab (zip-upload + unzip, or `git clone <your-repo>`), then `cd` into it.
3. Install deps (Colab already ships torch+CUDA):
   ```
   !pip install -q transformers scipy matplotlib pyyaml
   ```
4. Run the full pipeline (first real run: 50 examples/position at context length 1024):
   ```
   !python scripts/01_generate_dataset.py    --config configs/default.yaml --examples-per-position 50 --context-length 1024
   !python scripts/02_eval_baseline.py       --config configs/default.yaml
   !python scripts/03_measure_influence.py   --config configs/default.yaml
   !python scripts/04_train_standard.py      --config configs/default.yaml
   !python scripts/05_train_middle_weighted.py --config configs/default.yaml
   !python scripts/06_make_report.py         --config configs/default.yaml
   # Step-0 architectural-prior experiment (beats the paper's open question):
   !python scripts/03_measure_influence.py   --config configs/default.yaml --init-random
   !python scripts/07_step0_compare.py       --config configs/default.yaml
   # Weighting sweep (beta / sigma / edge-floor) to find a NET middle-accuracy win:
   !python scripts/08_sweep_intervention.py  --config configs/default.yaml
   # Claim 1 (fingerprint): random-init influence predicts trained accuracy
   !python scripts/09_fingerprint.py         --config configs/default.yaml --label pythia-70m
   # Claim 2 (cure bake-off): init-time reshaping vs loss-weighting vs standard
   !python scripts/10_cure_bakeoff.py        --config configs/default.yaml
   ```

   **Depth law (Claim 1, causal):** Pythia models share the GPT-NeoX tokenizer, so the dataset is
   reused across sizes. For each size, re-measure Step-0 + standard FT + fingerprint, then aggregate:
   ```python
   for M, L in [("EleutherAI/pythia-70m","70m"), ("EleutherAI/pythia-160m","160m"), ("EleutherAI/pythia-410m","410m")]:
       !python scripts/03_measure_influence.py --config configs/default.yaml --init-random --model $M
       !python scripts/04_train_standard.py    --config configs/default.yaml --model $M
       !python scripts/09_fingerprint.py       --config configs/default.yaml --label $L
   !python scripts/09_fingerprint.py --config configs/default.yaml --depth-trend
   ```
   (410m is optional on a T4 — use a small `train.batch_size` if memory is tight.)
   `--init-random` measures the influence valley on a randomly-initialized (untrained) model at
   the **same context length** as the trained runs, so stage 7 compares all four profiles
   (Step-0 / pretrained / standard-FT / intervention) apples-to-apples.
5. Inspect `outputs/plots/*.png` and `outputs/results_summary.md`.

**Scale up** (better statistics / longer context):
```
!python scripts/01_generate_dataset.py --config configs/default.yaml --examples-per-position 100 --context-length 2048
# then re-run 02..06
```
A one-click notebook is provided at `notebooks/colab_run.ipynb`.

> Note: scripts 02–06 locate the dataset by `context_length` + `examples_per_position` from the
> config, so if you override those on script 01 you must pass the **same** overrides (or edit
> `configs/default.yaml`) when running 02–06. Easiest: set them once in the config.

---

## Smoke test (any machine, no GPU needed)

Runs the whole pipeline at tiny scale (5 ex/pos, ctx 256, a few train steps) in minutes:
```
python scripts/01_generate_dataset.py --smoke
python scripts/02_eval_baseline.py    --smoke
python scripts/03_measure_influence.py --smoke
python scripts/04_train_standard.py   --smoke
python scripts/05_train_middle_weighted.py --smoke
python scripts/06_make_report.py      --smoke
```
Run unit tests: `python -m pytest tests/ -q`.

---

## Configuration

All knobs live in `configs/default.yaml`. The `smoke:` block is deep-merged on top when
`--smoke` is passed. Key CLI overrides: `--model`, `--examples-per-position`, `--context-length`,
`--epochs`, `--max-steps`, `--lr`, and (script 05) `--beta`, `--sigma`, `--inverse-influence`.

Model: `EleutherAI/pythia-70m` (auto-fallback to `gpt2`). Intervention weighting:
`position_weight(x) = 1 + beta * exp(-((x-0.5)^2)/(2*sigma^2))`, default `beta=2.0, sigma=0.18`.

## Outputs

- `outputs/plots/` — all PNG figures.
- `outputs/tables/` — per-example JSONL + per-position CSV for eval & influence, loss histories,
  Spearman JSONs, saved run configs.
- `outputs/checkpoints/` — fine-tuned model checkpoints.
- `outputs/results_summary.md` — headline metrics, Spearman, per-position tables, verdict.

## Reproducibility & robustness

- Global seed for python/numpy/torch; each stage saves its resolved config.
- Device auto-detected (CUDA → GPU, else CPU, logged).
- CUDA-OOM during eval halves the batch and retries; training logs an OOM hint.
- No silent failures — missing datasets / load errors raise with actionable messages.
