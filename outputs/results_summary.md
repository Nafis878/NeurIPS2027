# Results Summary

- Config: `C:\Users\UseR\Downloads\NeurIPS 2027\configs\default.yaml` (smoke=False)
- Model: `EleutherAI/pythia-70m` (fallback `gpt2`)
- Context length: 1024 | examples/position: 50 | position buckets: 11

## Headline metrics

| Method | Avg acc | Middle acc | Worst acc | Avg influence@answer | Spearman(infl,acc) rho |
|---|---|---|---|---|---|
| baseline | 0.005 | 0.040 | 0.000 | 7.36012 | 0.108 |
| standard_finetune | 0.098 | 0.040 | 0.020 | 0.18616 | -0.668 |
| intervention | 0.091 | 0.200 | 0.000 | 1.15756 | -0.447 |

## Experiment 2 — does influence predict accuracy?

- **baseline**: Spearman(influence, EM accuracy) rho=0.108 p=0.752 (n=11); vs log-prob rho=0.355 p=0.285
- **standard_finetune**: Spearman(influence, EM accuracy) rho=-0.668 p=0.025 (n=11); vs log-prob rho=-0.991 p=0.000
- **intervention**: Spearman(influence, EM accuracy) rho=-0.447 p=0.168 (n=11); vs log-prob rho=-0.973 p=0.000

### baseline: accuracy & influence by position

| norm_pos | n | EM acc | mean logprob | influence@answer | peak/trough | middle mass |
|---|---|---|---|---|---|---|
| 0.03 | 50 | 0.000 | -4.691 | 14.61566 | 24.61 | 0.083 |
| 0.06 | 50 | 0.000 | -5.662 | 5.33487 | 11.80 | 0.087 |
| 0.11 | 50 | 0.020 | -4.608 | 10.22600 | 13.84 | 0.080 |
| 0.21 | 50 | 0.000 | -5.275 | 6.76235 | 10.87 | 0.087 |
| 0.35 | 50 | 0.000 | -5.215 | 5.15880 | 9.12 | 0.084 |
| 0.50 | 50 | 0.040 | -5.008 | 5.96522 | 9.26 | 0.321 |
| 0.65 | 50 | 0.000 | -5.297 | 7.80661 | 11.87 | 0.121 |
| 0.80 | 50 | 0.000 | -5.040 | 4.67886 | 12.86 | 0.105 |
| 0.90 | 50 | 0.000 | -4.660 | 7.73341 | 17.01 | 0.095 |
| 0.95 | 50 | 0.000 | -5.046 | 6.33057 | 17.08 | 0.099 |
| 0.98 | 50 | 0.000 | -4.586 | 6.34897 | 24.30 | 0.085 |

### standard_finetune: accuracy & influence by position

| norm_pos | n | EM acc | mean logprob | influence@answer | peak/trough | middle mass |
|---|---|---|---|---|---|---|
| 0.03 | 50 | 0.200 | -0.000 | 0.00219 | 156.99 | 0.047 |
| 0.06 | 50 | 0.160 | -0.000 | 0.00125 | 95.75 | 0.039 |
| 0.11 | 50 | 0.160 | -0.000 | 0.00027 | 55.31 | 0.061 |
| 0.21 | 50 | 0.080 | -0.002 | 0.24557 | 70.82 | 0.097 |
| 0.35 | 50 | 0.020 | -0.000 | 0.02061 | 86.82 | 0.033 |
| 0.50 | 50 | 0.040 | -0.004 | 0.38215 | 67.08 | 0.556 |
| 0.65 | 50 | 0.060 | -0.000 | 0.02634 | 88.18 | 0.111 |
| 0.80 | 50 | 0.040 | -0.000 | 0.00555 | 48.11 | 0.053 |
| 0.90 | 50 | 0.120 | -0.000 | 0.00316 | 32.17 | 0.051 |
| 0.95 | 50 | 0.100 | -0.000 | 0.00296 | 55.49 | 0.050 |
| 0.98 | 50 | 0.100 | -0.042 | 1.35772 | 77.70 | 0.037 |

### intervention: accuracy & influence by position

| norm_pos | n | EM acc | mean logprob | influence@answer | peak/trough | middle mass |
|---|---|---|---|---|---|---|
| 0.03 | 50 | 0.080 | -0.001 | 0.08502 | 149.27 | 0.042 |
| 0.06 | 50 | 0.220 | -0.007 | 0.81605 | 122.76 | 0.028 |
| 0.11 | 50 | 0.100 | -0.002 | 0.31138 | 143.66 | 0.025 |
| 0.21 | 50 | 0.120 | -0.017 | 2.11858 | 137.31 | 0.120 |
| 0.35 | 50 | 0.180 | -0.002 | 0.42411 | 106.30 | 0.045 |
| 0.50 | 50 | 0.200 | -0.002 | 0.30183 | 67.34 | 0.654 |
| 0.65 | 50 | 0.060 | -0.001 | 0.23000 | 62.43 | 0.164 |
| 0.80 | 50 | 0.000 | -0.021 | 1.74641 | 79.07 | 0.085 |
| 0.90 | 50 | 0.020 | -0.022 | 2.67289 | 33.99 | 0.050 |
| 0.95 | 50 | 0.020 | -0.014 | 1.38728 | 104.14 | 0.040 |
| 0.98 | 50 | 0.000 | -0.023 | 2.63963 | 100.73 | 0.050 |

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

- Weak/unclear influence-accuracy link (baseline rho=0.108).
- Middle accuracy: intervention=0.200 vs standard=0.040 (delta=0.160).
