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

## Our thesis: **"Found in the Middle by Design"**

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
