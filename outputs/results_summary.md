# Results Summary

- Config: `C:\Users\UseR\Downloads\NeurIPS 2027\configs\default.yaml` (smoke=True)
- Model: `EleutherAI/pythia-70m` (fallback `gpt2`)
- Context length: 256 | examples/position: 5 | position buckets: 11

## Headline metrics

| Method | Avg acc | Middle acc | Worst acc | Avg influence@answer | Spearman(infl,acc) rho |
|---|---|---|---|---|---|
| baseline | 0.000 | 0.000 | 0.000 | 5.75594 | nan |
| standard_finetune | 0.709 | 1.000 | 0.400 | 0.21512 | 0.165 |
| intervention | 0.545 | 0.800 | 0.200 | 0.67533 | 0.337 |

## Experiment 2 — does influence predict accuracy?

- **baseline**: Spearman(influence, EM accuracy) rho=nan p=nan (n=11); vs log-prob rho=0.173 p=0.612
- **standard_finetune**: Spearman(influence, EM accuracy) rho=0.165 p=0.627 (n=11); vs log-prob rho=-0.082 p=0.811
- **intervention**: Spearman(influence, EM accuracy) rho=0.337 p=0.311 (n=11); vs log-prob rho=-0.627 p=0.039

### baseline: accuracy & influence by position

| norm_pos | n | EM acc | mean logprob | influence@answer | peak/trough | middle mass |
|---|---|---|---|---|---|---|
| 0.05 | 5 | 0.000 | -3.549 | 5.51341 | 20.30 | 0.090 |
| 0.08 | 5 | 0.000 | -3.464 | 8.93484 | 20.72 | 0.092 |
| 0.13 | 5 | 0.000 | -3.862 | 3.66116 | 10.73 | 0.098 |
| 0.22 | 5 | 0.000 | -5.214 | 3.72785 | 11.98 | 0.095 |
| 0.37 | 5 | 0.000 | -4.507 | 3.96651 | 8.70 | 0.096 |
| 0.51 | 5 | 0.000 | -5.333 | 3.37083 | 12.80 | 0.427 |
| 0.65 | 5 | 0.000 | -4.306 | 5.77254 | 13.09 | 0.160 |
| 0.80 | 5 | 0.000 | -5.575 | 10.50817 | 19.89 | 0.100 |
| 0.89 | 5 | 0.000 | -4.971 | 3.17448 | 11.76 | 0.105 |
| 0.94 | 5 | 0.000 | -4.464 | 5.64091 | 19.25 | 0.090 |
| 0.97 | 5 | 0.000 | -4.502 | 9.04466 | 20.41 | 0.089 |

### standard_finetune: accuracy & influence by position

| norm_pos | n | EM acc | mean logprob | influence@answer | peak/trough | middle mass |
|---|---|---|---|---|---|---|
| 0.05 | 5 | 0.400 | -0.005 | 0.00818 | 29.80 | 0.120 |
| 0.08 | 5 | 0.600 | -0.001 | 1.66745 | 130.20 | 0.022 |
| 0.13 | 5 | 0.800 | -0.000 | 0.01536 | 44.96 | 0.056 |
| 0.22 | 5 | 0.800 | -0.001 | 0.22894 | 48.40 | 0.064 |
| 0.37 | 5 | 0.400 | -0.001 | 0.12464 | 54.21 | 0.096 |
| 0.51 | 5 | 1.000 | -0.000 | 0.08437 | 36.62 | 0.587 |
| 0.65 | 5 | 0.600 | -0.000 | 0.03159 | 22.00 | 0.148 |
| 0.80 | 5 | 0.600 | -0.001 | 0.01697 | 17.93 | 0.085 |
| 0.89 | 5 | 1.000 | -0.001 | 0.12298 | 42.57 | 0.063 |
| 0.94 | 5 | 0.800 | -0.001 | 0.02036 | 28.73 | 0.111 |
| 0.97 | 5 | 0.800 | -0.000 | 0.04547 | 22.35 | 0.079 |

### intervention: accuracy & influence by position

| norm_pos | n | EM acc | mean logprob | influence@answer | peak/trough | middle mass |
|---|---|---|---|---|---|---|
| 0.05 | 5 | 0.200 | -0.001 | 0.02287 | 32.78 | 0.100 |
| 0.08 | 5 | 0.200 | -0.001 | 0.74945 | 20.66 | 0.087 |
| 0.13 | 5 | 0.800 | -0.000 | 0.02579 | 66.60 | 0.072 |
| 0.22 | 5 | 0.800 | -0.003 | 0.55552 | 34.46 | 0.062 |
| 0.37 | 5 | 0.600 | -0.002 | 0.34180 | 56.14 | 0.028 |
| 0.51 | 5 | 0.800 | -0.022 | 1.54453 | 64.51 | 0.639 |
| 0.65 | 5 | 0.600 | -0.002 | 0.00360 | 35.88 | 0.159 |
| 0.80 | 5 | 0.200 | -0.000 | 0.03029 | 22.84 | 0.086 |
| 0.89 | 5 | 0.800 | -0.012 | 4.09818 | 39.41 | 0.069 |
| 0.94 | 5 | 0.400 | -0.001 | 0.04694 | 29.30 | 0.053 |
| 0.97 | 5 | 0.600 | -0.000 | 0.00963 | 36.74 | 0.072 |

## Figures (outputs/plots/)

- [x] `accuracy_vs_position_baseline.png`
- [x] `influence_vs_position_baseline.png`
- [x] `loss_curve_standard_finetune.png`
- [x] `accuracy_vs_position_standard_finetune.png`
- [x] `influence_vs_position_standard_finetune.png`
- [x] `accuracy_vs_position_intervention.png`
- [x] `influence_vs_position_intervention.png`
- [x] `comparison_accuracy_all_methods.png`
- [x] `comparison_influence_all_methods.png`

## Automated verdict

- Middle accuracy: intervention=0.800 vs standard=1.000 (delta=-0.200).
