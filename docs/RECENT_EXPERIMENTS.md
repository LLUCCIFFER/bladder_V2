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

