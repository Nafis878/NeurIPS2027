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
| C3 | Flattening the influence valley **lifts middle-position accuracy** | their disclaimer #2/#3 | middle EM 0.04→0.20 (pos 0.5), 0.02→0.18 (pos 0.35) — `comparison_accuracy_all_methods.png` |

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
- **Don't over-claim the intervention.** At 50/pos it currently *redistributes* accuracy toward
  the middle (right-edge cost), average flat. The honest framing: it **bridges the middle dead
  zone**; making it net-positive (edge-floor term, β/σ sweep, larger model) is the headline
  experiment to nail before submission.

## Status

- Built & smoke-verified: Step-0 random-init measurement, valley-depth metrics, overlay plots,
  comparison script. Real T4 run pending for apples-to-apples (all four profiles at ctx 1024).
- Next: run Step-0 at ctx 1024 on T4; then β/σ sweep + edge-floor to turn redistribution into a
  net middle gain; then depth-scaling.
