# Mild Aug Mean + ntl_boost2.5 + Old-9 Config

## Setup

- Concept bank: refined vectors + original top10, 40 concepts total.
- Train representation: mild image-level augmentation with mean view aggregation.
- Views: orig, color_bright, color_dark, rotate, crop.
- Loss: class-weighted CE with `ntl_boost2.5`; lambda_cycl = 0.0.
- Old-9 config groups:

| Group | Seeds | Hidden Dim | lambda_align |
|---|---:|---:|---:|
| align015_h128 | 42,43,44 | 128 | 0.15 |
| align012_h128 | 42,43,44 | 128 | 0.12 |
| align005_h256 | 42,43,44 | 256 | 0.05 |

## 9-Model Ensemble Result

| Split | Acc | Macro-F1 | AUROC | HGC F1 | LGC F1 | NTL F1 | NST F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| val | 0.8442 | 0.8313 | 0.9699 | 0.7483 | 0.8487 | 0.8000 | 0.9282 |
| test | 0.5926 | 0.6235 | 0.8299 | 0.5035 | 0.5043 | 0.5417 | 0.9444 |

## Group-Level 3-Model Ensembles

| Group | Split | Acc | Macro-F1 | AUROC | HGC F1 | LGC F1 | NTL F1 | NST F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| align015_h128 | val | 0.8474 | 0.8323 | 0.9696 | 0.7682 | 0.8534 | 0.7843 | 0.9231 |
| align015_h128 | test | 0.5926 | 0.6155 | 0.8328 | 0.5205 | 0.5273 | 0.5000 | 0.9143 |
| align012_h128 | val | 0.8474 | 0.8338 | 0.9696 | 0.7483 | 0.8536 | 0.8000 | 0.9333 |
| align012_h128 | test | 0.5767 | 0.6068 | 0.8227 | 0.4930 | 0.5000 | 0.5200 | 0.9143 |
| align005_h256 | val | 0.8182 | 0.8007 | 0.9657 | 0.7027 | 0.8216 | 0.7556 | 0.9231 |
| align005_h256 | test | 0.5767 | 0.5804 | 0.8269 | 0.5000 | 0.5128 | 0.3500 | 0.9589 |

## Comparison

| Setting | Test Acc | Test Macro-F1 | Test AUROC | HGC F1 | LGC F1 | NTL F1 | NST F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| refined old-9, no aug/no class weight | 0.6614 | 0.6140 | 0.8271 | 0.6923 | 0.5542 | 0.2927 | 0.9167 |
| mild aug mean + ntl_boost2.5, 3-model h128 align0.10 | 0.5979 | 0.6289 | 0.8272 | 0.5333 | 0.4727 | 0.5652 | 0.9444 |
| mild aug mean + ntl_boost2.5, old-9 config | 0.5926 | 0.6235 | 0.8299 | 0.5035 | 0.5043 | 0.5417 | 0.9444 |

## Interpretation

- The old-9 matched version gives test Macro-F1 `0.6235`, slightly higher than refined old-9 without augmentation/class weighting (`0.6140`).
- The main improvement is NTL: F1 rises from `0.2927` to `0.5417`.
- Accuracy drops from `0.6614` to `0.5926`, mainly because class weighting trades some HGC/LGC accuracy for better NTL balance.
- Compared with the simpler 3-model `mild aug mean + ntl_boost2.5` result, the old-9 config is slightly worse in Macro-F1 (`0.6235` vs `0.6289`) and NTL F1 (`0.5417` vs `0.5652`).
- Therefore, old-9 config is useful as a fair comparison, but it is not the new best configuration.

## Output Files

- 9-model metrics: `/home/kunet.ae/100069491/newcode/ebtc_mild_aug_mean_ntlboost25_old9_outputs/old9_ensemble/ensemble_metrics.csv`
- Group metrics: `/home/kunet.ae/100069491/newcode/ebtc_mild_aug_mean_ntlboost25_old9_outputs/old9_ensemble/group_3model_ensemble_metrics.csv`
- Confusion matrix val: `/home/kunet.ae/100069491/newcode/ebtc_mild_aug_mean_ntlboost25_old9_outputs/old9_ensemble/confusion_matrix_val.csv`
- Confusion matrix test: `/home/kunet.ae/100069491/newcode/ebtc_mild_aug_mean_ntlboost25_old9_outputs/old9_ensemble/confusion_matrix_test.csv`
- Manifest: `/home/kunet.ae/100069491/newcode/ebtc_mild_aug_mean_ntlboost25_old9_outputs/old9_ensemble/ensemble_manifest.json`
