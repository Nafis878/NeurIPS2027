# Beating "Lost in the Middle at Birth" (Chowdhury, arXiv:2603.10123, Mar 2026)

## What that paper establishes (and deliberately stops at)

It is a **pure theory + Step-0 measurement** paper. Its single precise claim: the
"lost in the middle" U-shape is present **at initialization**, before any training or
positional encoding, as a geometric property of the causal decoder with residuals.

- Models multi-layer causal attention as iterated powers of the **Cesàro matrix**; derives
  a closed-form influence density: a logarithmic **Primacy Tail** (causal masking), an O(1)
  **Recency Anchor** (residual connections), and an **O(1/(H−1)!) factorial "dead zone"** in
  the middle (H = depth).
- Validates the closed-form theory against the empirical Step-0 input→output **Jacobian norm**
  on untrained **Qwen2-0.5B** and **GPT-2** (Spearman ρ=0.99, Wasserstein 0.02); shows the
  shape is identical with/without RoPE (ρ=0.99).
- Shows standard pretraining **deepens** the relative valley (their Figure 3).

### Their explicit disclaimers = our open doors
1. *"We do **not** claim the Jacobian norm directly measures retrieval accuracy."*
2. *"This paper derives the architectural baseline; evaluating **interventions to overcome it
   is future work**."*
3. Future-work list names our method verbatim: *"targeted loss weighting … middle-context
   curriculum … over-sampling needle-in-a-haystack data."*
4. Open question: *"the upper bound of the Score Pathway's ability to override the topological
   baseline under aggressive, position-targeted fine-tuning remains an open empirical question."*

## NeurIPS novelty thesis: **"Born Lost, Born Legible, Born Curable"**

Executing their suggested future work (loss weighting) is *not novel enough* — it is literally
their suggestion. The novel program attacks the two things they treat as **fixed**: the unmeasured
link to accuracy, and the *immutable* birthright. Two flagship claims:

**N1 — Legible at birth (a new predictive law + a correction of their U-shape).** The random-init
(Step-0) influence fingerprint **forecasts the trained model's per-position retrieval accuracy** —
before any training or data. Crucially we find a **decomposition they miss**: at random init the
attention is uniform (Cesàro), so there is *no distance-decay* and the fingerprint is **primacy-only**
(monotonic), **not** the full U. So lost-in-the-middle is *two* phenomena:
- a **birthright** primacy-tail + middle-collapse, **forecastable at init**, and
- a **learned recency wall** that emerges during training (absent at birth).

  *Real result (Pythia-70m, ctx 1024, 550 ex):* over the architectural region (x≤0.7),
  **Spearman(Step-0 fingerprint, trained accuracy) = 0.85, p=0.016**; full-range ρ falls to 0.38
  precisely because of the learned recency recovery (recency_gap = +0.067). This *sharpens* the
  paper, whose Step-0 "U-shape" conflates the final-token residual anchor with retrieval recency —
  we show the recency benefit for non-final facts is **not** a birthright. (`scripts/09_fingerprint.py`)
  Made causal by a **depth sweep** (Pythia 70m/160m/410m = 6/12/24 layers): their O(1/(H−1)!) predicts
  deeper → deeper birth-valley → worse trained middle; the fingerprint should track it. (`--depth-trend`)

**N2 — Curable at birth (intervene on the prior, not the loss).** They model each layer as
`N=(1−α)I+αM` and treat it as fixed. We make it an **editable knob**: (a) **residual-α reshaping** —
forward hooks scaling each block's residual branch, with the schedule chosen to **minimize the Step-0
valley** (the fingerprint as a *design target*); and (b) **distributed anchor registers** — inject
anchor tokens through the context so every region gets a local readout anchor (the mechanism that
rescues the end). Bake-off vs standard FT and the loss-weighting winner, at matched compute, asks:
**does editing the initialization prior beat training-time fixes on middle retrieval?**
(`src/architecture.py`, `scripts/10_cure_bakeoff.py`)

**Unifying idea (the out-of-the-box bit):** the Step-0 fingerprint is *both a diagnostic and a design
target* — it predicts failure (N1) and we minimize it at init to cure failure (N2). No prior work
predicts trained retrieval from random weights, nor edits the initialization prior to fix
lost-in-the-middle.

### Real T4 results — UPDATED with 3-seed runs (the single-seed numbers were noise)

**Honest status: at Pythia-70m / 50-per-position, neither N1 (predictive) nor N2 (cure) is
statistically robust.** The single-seed wins (N1 ρ=0.85; loss-weighting/anchors beating standard)
do **not** survive 3 seeds. Do not quote the single-seed numbers.

- **N1 — does NOT replicate as a strong claim.** arch-region Spearman = **0.25 ± 0.36** over 3 seeds
  (per-seed −0.15 / +0.53 / +0.38; one seed negative; win-rate 2/3). `recency_gap = −0.02 ± 0.03`
  (the earlier +0.067 decomposition signal vanishes). `fingerprint_multiseed_pythia-70m.csv`.
- **N2 — no cure robustly beats standard.** Middle accuracy (mean ± std, 3 seeds):
  standard 0.113 ± 0.076; loss_edgefloor 0.153 ± **0.232**; resid_alpha 0.040 ± 0.040;
  anchors 0.093 ± 0.092. Per-seed is wild (seed-2 loss_edgefloor = 0.42, others ≈ 0–0.04). The
  variance dwarfs every difference. `resid_alpha` is *consistently worse*; `anchors` ≈ standard.
  `cure_bakeoff.csv`, `cure_bakeoff_seeds.csv`.
- **Root cause:** at 50/pos the trained middle accuracy is 1–3 correct out of 50, so seed noise
  flips bucket accuracy by ±0.02–0.04 — which swamps both the fingerprint correlation and the cure
  deltas. The experiment is *underpowered*, not (yet) informative about the hypotheses.

**What IS robust (and still worth a claim):**
- The **Step-0 architectural prior is highly reproducible**: valley_depth = 0.7886 / 0.7837 / 0.7862
  across seeds (std ≈ 0.002). The birthright *measurement* replicates cleanly (consistent with the
  paper); it is the link to *trained accuracy* that is noise-dominated at this scale.
- The prior's **shape is editable**: a depth-decreasing residual schedule (`ramp_2to0.5`) flattens
  the Step-0 valley (peak/trough 5.85→3.61); uniform scaling is ratio-invariant. (Mechanistic, seed-
  stable — but flattening it did not improve trained retrieval here.)
- **Methodological finding:** small-scale lost-in-the-middle interventions are seed-noise-dominated;
  honest evaluation requires many more examples/position, more seeds, and ideally a model whose
  trained accuracy is not floored. This is itself a useful caution for the subfield.

**Path to a real result (needed before any submission):** scale to ≥200–500 examples/position (so
bucket accuracy is estimated to ±0.01), ≥5 seeds, and a larger model (Pythia-410m/1.4b) so middle
accuracy isn't floored near 0; or switch the predictive target to a less granular signal
(answer log-prob / rank) that is far less noisy than 0/1 EM at small n.

## Foundation (Phase-1, done): "Found in the Middle by Design"

We take their architectural-prior baseline as given and answer their open question empirically.
Three claims, each mapped to a disclaimer above:

| # | Our claim | Beats their gap | Evidence in this repo |
|---|---|---|---|
| C1 | The position-wise **influence profile predicts per-position retrieval accuracy** | their disclaimer #1 | `spearman_*` (influence ↔ EM & log-prob); post-FT ρ(infl,logprob)=−0.99, p≈4e-9 |
| C2 | **Standard training deepens** the init valley; **position-targeted training flattens it** | their Fig. 3 + disclaimer #2/#4 | `07_step0_compare.py` → `valley_depth_by_method.csv`, `influence_step0_vs_trained*.png` |
| C3 | A position-targeted objective gives a **net** accuracy gain over standard FT | their disclaimer #2/#3 | **edge-floor** config (β=2, σ=0.18, edge_γ=1) beats standard FT on middle (0.14>0.12), avg (0.231>0.162), worst (0.14>0.04) and edge (0.237>0.19) — `sweep_intervention.csv` |

The chain C1→C2→C3 is exactly the causal story their paper sets up but declines to test:
init prior (theirs) → predicts retrieval failure (C1) → standard training can't escape it (C2) →
a targeted objective bridges the O(1/(H−1)!) dead zone and recovers middle accuracy (C2+C3).

## Experiments and how they engage the paper

- **Step-0 / random-init influence** (`03_measure_influence.py --init-random`): reproduce the
  init U-shape on our setup → directly engages their central result, on our model/metric.
- **Valley-depth comparison** (`07_step0_compare.py`): Step-0 vs pretrained vs standard-FT vs
  intervention; reports `valley_depth`, `peak_to_trough`, `middle_mass`. This is the decisive
  "standard deepens / intervention flattens" figure (their Fig. 3, reversed by us).
- **Influence↔accuracy link** (Spearman in 03 / `_train_common`): closes their disclaimed gap.
- **Middle-weighted intervention** (`05_train_middle_weighted.py`): the `position_weight`
  Gaussian; matched compute vs standard FT.

## Where we must be careful (reviewer-proofing)

- **Distinct metric.** Their influence = input→output Jacobian norm of the *final hidden state*.
  Ours = gradient of the *answer log-prob* w.r.t. input embeddings — a **task-grounded** signal,
  which is *why* we can correlate it with retrieval accuracy. State this as a feature, and
  (optional next step) add their exact Jacobian metric for a head-to-head.
- **Scale honesty.** We use Pythia-70m on a T4; they use Qwen2-0.5B / GPT-2. Our claims are
  about the *mechanism and the intervention*, validated at small scale. A depth-scaling study
  (Pythia 70m/160m/410m = 6/12/24 layers) would test their O(1/(H−1)!) law and is the natural
  follow-up.
- **The edge-floor is the real finding.** Naïve middle-only weighting (the paper's literal
  suggestion, "targeted loss weighting") *collapses* accuracy in our T4 sweep (avg ~0.03 vs
  standard 0.16). Only the **edge-floor** variant — which protects primacy/recency while boosting
  the middle — yields a net win. The headline is therefore not "loss weighting works" but
  "loss weighting works *iff* you don't starve the geometric anchors the architecture relies on."
- **Single-seed caveat.** The sweep is one seed at 50/pos with coarse EM; the *ranking* is
  internally consistent (one session) but absolute numbers vary run-to-run on GPU. Confirm the
  edge-floor net win across ≥3 seeds (and ideally 100/pos) before submission.

## Status (real T4 results in)

- **C1 ✓** influence ↔ accuracy: post-FT Spearman ρ(infl,logprob)=−0.99.
- **C2 ✓** Step-0 valley reproduced (peak/trough 5.29) and standard training deepens it
  (→39.58), matching their Fig. 3; `valley_depth_by_method.csv`.
- **C3 ✓ (net win)** edge-floor config beats standard FT on middle/avg/worst/edge accuracy;
  middle-only weighting backfires. `sweep_intervention.csv`, `sweep_intervention_accuracy.png`.
  Default config now set to the winning edge-floor weighting.
- Next: multi-seed confirmation of the edge-floor win; depth-scaling (Pythia 70m/160m/410m) to
  test their O(1/(H−1)!) law; optional head-to-head with their exact final-hidden-state Jacobian.
