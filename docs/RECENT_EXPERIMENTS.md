# Recent EBTC Experiments

This document records the recent experiments that are intentionally kept as
lightweight GitHub-tracked summaries. Large generated outputs, checkpoints,
embeddings, CSV tables, and figures remain local because they are ignored by
`.gitignore`.

## 1. Embedding Refinement Stage

The embedding refinement stage keeps the filtered top300 concept bank fixed,
adds lightweight residual MLP adapters to BioMedCLIP image and text embeddings,
and then reruns the original top10 whitelist CBM.

Main scripts:

- `ebtc_embedding_refinement_stage.py`
- `ebtc_embedding_refinement_ensemble.py`
- `ebtc_ensemble_checkpoints.py`

Core data flow:

```text
cached BioMedCLIP image embedding
  -> image residual adapter
  -> refined image embedding
  -> cosine similarity to refined original top10 concept embeddings
  -> concept activation vector
  -> linear classifier
```

The strongest setting uses:

- concept bank: original discriminative top10 whitelist, 40 concepts total
- vectors: refined image embeddings + refined concept text embeddings
- classifier input: concept activations only
- no hidden image-feature bypass

## 2. Old 9-Model Ensemble Composition

The previous `old 9-model ensemble` is a mixed-configuration ensemble:

| Group | Directory | Seeds | Hidden Dim | lambda_cycl | lambda_align |
|---|---|---:|---:|---:|---:|
| old-A | `ebtc_discriminative_whitelist_cycl_search2_h128_align015` | 42,43,44 | 128 | 0.0 | 0.15 |
| old-B | `ebtc_discriminative_whitelist_cycl_search3_h128_align012` | 42,43,44 | 128 | 0.0 | 0.12 |
| old-C | `ebtc_discriminative_whitelist_cycl_improved_fast_align_only_top10` | 42,43,44 | 256 | 0.0 | 0.05 |

Old 9-model ensemble result:

| Split | Accuracy | Macro-F1 | Macro-AUROC |
|---|---:|---:|---:|
| val | 0.8604 | 0.8624 | 0.9648 |
| test | 0.5450 | 0.5665 | 0.7957 |

Per-class test F1:

| Class | F1 |
|---|---:|
| HGC | 0.5033 |
| LGC | 0.4404 |
| NTL | 0.4400 |
| NST | 0.8824 |

Local result file:

```text
ebtc_discriminative_whitelist_cycl_improvement_analysis/best_val_selected_ensemble_3dirs/ensemble_metrics.csv
```

## 3. Refined Vectors + Original Top10, Same Old-9 Configuration

To compare fairly against the old 9-model ensemble, the refined-vector model
was rerun with the same three ensemble groups:

| Group | Setting | Seeds | Hidden Dim | lambda_cycl | lambda_align |
|---|---|---:|---:|---:|---:|
| refined-A | refined vectors + original top10 | 42,43,44 | 128 | 0.0 | 0.15 |
| refined-B | refined vectors + original top10 | 42,43,44 | 128 | 0.0 | 0.12 |
| refined-C | refined vectors + original top10 | 42,43,44 | 256 | 0.0 | 0.05 |

Result:

| Split | Accuracy | Macro-F1 | Macro-AUROC |
|---|---:|---:|---:|
| val | 0.7792 | 0.7904 | 0.9363 |
| test | 0.6614 | 0.6140 | 0.8271 |

Per-class test F1:

| Class | F1 |
|---|---:|
| HGC | 0.6923 |
| LGC | 0.5542 |
| NTL | 0.2927 |
| NST | 0.9167 |

Interpretation:

- The refined-vector same-configuration ensemble improves test Accuracy,
  Macro-F1, and Macro-AUROC over the old 9-model ensemble.
- HGC, LGC, and NST improve substantially.
- NTL remains the main failure case and drops compared with the old ensemble.

Local result file:

```text
ebtc_embedding_refinement_old9_matched_outputs/refined_orig_top10_old9_config_ensemble/ensemble_metrics.csv
```

## 4. Current Best Strict Same-Level 9-Model Result

A refined-vector original-top10 9-model ensemble with `lambda_align=0.10`
was selected by validation behavior during alignment search.

Result:

| Split | Accuracy | Macro-F1 | Macro-AUROC |
|---|---:|---:|---:|
| val | 0.8701 | 0.8691 | 0.9572 |
| test | 0.5714 | 0.5735 | 0.8209 |

Per-class test F1:

| Class | F1 |
|---|---:|
| HGC | 0.5180 |
| LGC | 0.5042 |
| NTL | 0.3404 |
| NST | 0.9315 |

Local result file:

```text
ebtc_embedding_refinement_align_search_outputs/refined_orig_top10_align01_9seed_ensemble/ensemble_metrics.csv
```

## 5. InfoNCE Formula Check

The reference single-positive InfoNCE formula is:

```text
L_InfoNCE = -log [ exp(sim(q, k+) / tau) /
                   ( exp(sim(q, k+) / tau) + sum_i exp(sim(q, k_i-) / tau) ) ]
```

The implemented CyCL loss is a weighted multi-positive extension:

```text
p_ij = exp(sim(z_i, z_j) / tau) / sum_a exp(sim(z_i, z_a) / tau)

L_i = - [sum_j w_ij log p_ij] / [sum_j w_ij]
```

If `w_ij` is one-hot with a single positive key, this reduces to the standard
single-positive InfoNCE. Therefore, the implementation is not missing the
positive term in the denominator. It is a weighted multi-positive InfoNCE /
supervised-contrastive variant.

Relevant implementation locations:

- `ebtc_discriminative_whitelist_cycl.py::profile_weighted_contrastive_loss`
- `ebtc_embedding_refinement_stage.py::multipositive_nce`
- `ebtc_cbm_cycl_v2_lib.py::weighted_image_image_contrastive_loss_v2`
- `ebtc_cbm_cycl_v2_lib.py::weighted_image_text_contrastive_loss_v2`

## 6. High-Weight CyCL Experiment

The supervisor suggested testing a larger CyCL coefficient between 0.3 and 0.5,
and increasing concept alignment to 0.2.

Tested settings:

- `lambda_cycl = 0.30`, `lambda_align = 0.20`
- `lambda_cycl = 0.50`, `lambda_align = 0.20`

Seed-mean results:

| Setting | Val Macro-F1 | Test Accuracy | Test Macro-F1 | Test Macro-AUROC |
|---|---:|---:|---:|---:|
| cycl0.30 align0.20 | 0.5097 ± 0.0806 | 0.4850 ± 0.0375 | 0.3759 ± 0.0278 | 0.7147 ± 0.0284 |
| cycl0.50 align0.20 | 0.5090 ± 0.0835 | 0.4568 ± 0.0480 | 0.3731 ± 0.0100 | 0.7226 ± 0.0235 |

3-seed ensemble results:

| Setting | Test Accuracy | Test Macro-F1 | Test Macro-AUROC |
|---|---:|---:|---:|
| cycl0.30 align0.20 | 0.5026 | 0.3871 | 0.7333 |
| cycl0.50 align0.20 | 0.5026 | 0.4101 | 0.7376 |

Interpretation:

- Larger CyCL weights substantially degraded performance in this setting.
- The model became biased toward HGC/NST and LGC almost collapsed.
- The issue is not the InfoNCE denominator; the current soft pair weighting is
  too strong when multiplied by `lambda_cycl=0.3-0.5`.

Local output directory:

```text
ebtc_embedding_refinement_high_loss_weight_outputs/
```

## 7. Class Weighting Ablation

The latest meeting suggested testing class weighting to address class
imbalance, especially the minority NTL class. This ablation fixed the current
refined-vector original-top10 CBM pipeline and changed only the CE class
weights.

Script:

```text
ebtc_class_weight_ablation.py
```

Fixed setting:

- vectors: refined BioMedCLIP image/text embeddings
- concept bank: original top10 whitelist, 40 concepts total
- classifier: linear classifier on concept activations
- hidden dim: 128
- `lambda_cycl = 0.0`
- `lambda_align = 0.10`
- seeds: 42, 43, 44

Train split counts and weights:

| Mode | HGC | LGC | NTL | NST |
|---|---:|---:|---:|---:|
| none | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| sqrt_inverse | 1.0085 | 0.8074 | 1.9005 | 0.9106 |
| pow0.75 | 1.0127 | 0.7256 | 2.6201 | 0.8689 |
| inverse | 1.0170 | 0.6520 | 3.6121 | 0.8292 |
| ntl_boost2 | 1.0000 | 1.0000 | 2.0000 | 1.0000 |
| ntl_boost3 | 1.0000 | 1.0000 | 3.0000 | 1.0000 |

Seed-mean results:

| Mode | Val Macro-F1 | Test Accuracy | Test Macro-F1 | Test AUROC | Test NTL F1 |
|---|---:|---:|---:|---:|---:|
| none | 0.6661 ± 0.0556 | 0.5626 ± 0.0261 | 0.5143 ± 0.0155 | 0.7910 ± 0.0180 | 0.1229 |
| sqrt_inverse | 0.5381 ± 0.2297 | 0.5044 ± 0.1324 | 0.4232 ± 0.1675 | 0.7482 ± 0.1113 | 0.0901 |
| pow0.75 | 0.7426 ± 0.0477 | 0.5273 ± 0.0950 | 0.5252 ± 0.0769 | 0.7834 ± 0.0057 | 0.2649 |
| inverse | 0.6773 ± 0.1344 | 0.5926 ± 0.0317 | 0.5469 ± 0.0588 | 0.7779 ± 0.0351 | 0.2400 |
| ntl_boost2 | 0.6941 ± 0.2365 | 0.5062 ± 0.0319 | 0.4814 ± 0.0495 | 0.7949 ± 0.0358 | 0.2246 |
| ntl_boost3 | 0.7809 ± 0.0738 | 0.5432 ± 0.0272 | 0.5599 ± 0.0146 | 0.8009 ± 0.0114 | 0.4369 |

3-seed ensemble test results:

| Mode | Accuracy | Macro-F1 | Macro-AUROC | HGC F1 | LGC F1 | NTL F1 | NST F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| none | 0.5767 | 0.4987 | 0.8084 | 0.5731 | 0.4630 | 0.0000 | 0.9589 |
| sqrt_inverse | 0.5873 | 0.5370 | 0.8317 | 0.5036 | 0.5714 | 0.1379 | 0.9351 |
| pow0.75 | 0.4603 | 0.4776 | 0.7834 | 0.3284 | 0.4179 | 0.2500 | 0.9143 |
| inverse | 0.6032 | 0.5786 | 0.8158 | 0.6036 | 0.4792 | 0.3000 | 0.9315 |
| ntl_boost2 | 0.5344 | 0.5451 | 0.8064 | 0.4196 | 0.4921 | 0.3243 | 0.9444 |
| ntl_boost3 | 0.5556 | 0.5821 | 0.8146 | 0.4857 | 0.4957 | 0.4815 | 0.8657 |

Interpretation:

- Class weighting is necessary: without it, ensemble NTL F1 collapses to 0.0.
- `inverse` gives the best test accuracy and a balanced HGC/NST result.
- `ntl_boost3` gives the best test Macro-F1 and the strongest NTL F1.
- The NTL gain from `ntl_boost3` comes with lower NST F1, so it should be
  treated as the NTL-focused candidate rather than an unconditional final
  setting.

Local output directory:

```text
ebtc_class_weight_ablation_outputs/
```
