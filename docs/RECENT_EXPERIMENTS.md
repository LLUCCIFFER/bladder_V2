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

## 8. Class Weight Fine Search

After the first class-weighting ablation, a finer search was run around the
useful NTL-focused region.

Additional script support:

- `custom_A_B_C_D` weight mode was added to `ebtc_class_weight_ablation.py`.
- The four values correspond to explicit `HGC/LGC/NTL/NST` CE weights.

Search output directory:

```text
ebtc_class_weight_ablation_search2_outputs/
```

Searched modes:

| Mode | HGC | LGC | NTL | NST |
|---|---:|---:|---:|---:|
| ntl_boost2.5 | 1.0 | 1.0 | 2.5 | 1.0 |
| ntl_boost3 | 1.0 | 1.0 | 3.0 | 1.0 |
| ntl_boost3.5 | 1.0 | 1.0 | 3.5 | 1.0 |
| ntl_boost4 | 1.0 | 1.0 | 4.0 | 1.0 |
| ntl_boost5 | 1.0 | 1.0 | 5.0 | 1.0 |
| custom_1_1_2.5_1.1 | 1.0 | 1.0 | 2.5 | 1.1 |
| custom_1_1_3_1.1 | 1.0 | 1.0 | 3.0 | 1.1 |
| custom_1_1_3_1.2 | 1.0 | 1.0 | 3.0 | 1.2 |
| custom_1.1_1_3_1.2 | 1.1 | 1.0 | 3.0 | 1.2 |
| custom_1_0.9_3_1.2 | 1.0 | 0.9 | 3.0 | 1.2 |

Validation-selected best by 3-seed ensemble Macro-F1:

| Mode | Val Acc | Val Macro-F1 | Val AUROC | Test Acc | Test Macro-F1 | Test AUROC |
|---|---:|---:|---:|---:|---:|---:|
| ntl_boost2.5 | 0.8571 | 0.8463 | 0.9688 | 0.5714 | 0.5903 | 0.8090 |

Test per-class F1 for `ntl_boost2.5`:

| HGC | LGC | NTL | NST |
|---:|---:|---:|---:|
| 0.5170 | 0.5088 | 0.4400 | 0.8955 |

Best diagnostic test Macro-F1, not validation-selected:

| Mode | Val Macro-F1 | Test Acc | Test Macro-F1 | Test AUROC |
|---|---:|---:|---:|---:|
| ntl_boost4 | 0.7505 | 0.5873 | 0.6075 | 0.8277 |

Interpretation:

- `ntl_boost2.5` is the strict validation-selected setting and is the current
  recommended class-weight mode for the next-stage mild augmentation run.
- `ntl_boost4` has the best diagnostic test Macro-F1, but its validation
  Macro-F1 is much lower. It should not be promoted as the final choice unless
  repeated validation behavior supports it.
- Compared with no class weighting, `ntl_boost2.5` fixes the NTL collapse:
  ensemble test NTL F1 improves from `0.0000` to `0.4400`.

## 9. Image-Level Mild Augmentation

The meeting also suggested using mild image augmentation to improve image
representation robustness. This stage keeps the concept bank and classifier
fixed, and only changes how train image embeddings are produced.

Script:

```text
ebtc_image_level_mild_aug.py
```

Mild views:

```text
orig,color_bright,color_dark,rotate,crop
```

Two train strategies were tested:

- `expand`: use every augmented view as a separate train row, giving 6285 train
  rows.
- `mean`: encode all views, average them back to one robust embedding per
  original image, giving 1257 train rows.

Validation/test remain clean single-view refined embeddings.

### Expand Strategy

Local output:

```text
ebtc_image_level_mild_aug_outputs/
```

3-seed ensemble test results:

| Weight mode | Test Acc | Test Macro-F1 | Test AUROC | HGC F1 | LGC F1 | NTL F1 | NST F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ntl_boost2.5 | 0.5079 | 0.5261 | 0.7732 | 0.4658 | 0.4746 | 0.3774 | 0.7869 |
| inverse | 0.5238 | 0.5547 | 0.7694 | 0.4722 | 0.4483 | 0.4727 | 0.8254 |
| ntl_boost4 | 0.5132 | 0.5223 | 0.7967 | 0.5161 | 0.4630 | 0.3860 | 0.7241 |

Interpretation:

- Directly expanding train to 5x augmented rows did not help.
- It produced very high validation scores but lower test performance, indicating
  likely overfitting to correlated augmented views or a train/test embedding
  distribution shift.

### Mean Strategy

Local output:

```text
ebtc_image_level_mild_aug_mean_outputs/
```

3-seed ensemble test results:

| Weight mode | Test Acc | Test Macro-F1 | Test AUROC | HGC F1 | LGC F1 | NTL F1 | NST F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ntl_boost2.5 | 0.5979 | 0.6289 | 0.8272 | 0.5333 | 0.4727 | 0.5652 | 0.9444 |
| inverse | 0.5714 | 0.5326 | 0.8028 | 0.6054 | 0.2985 | 0.3077 | 0.9189 |
| ntl_boost4 | 0.6085 | 0.6270 | 0.8517 | 0.5440 | 0.6111 | 0.4872 | 0.8657 |

Comparison to no-augmentation class-weight search:

| Setting | Weight mode | Test Acc | Test Macro-F1 | Test AUROC | NTL F1 |
|---|---|---:|---:|---:|---:|
| no augmentation | ntl_boost2.5 | 0.5714 | 0.5903 | 0.8090 | 0.4400 |
| mild aug mean | ntl_boost2.5 | 0.5979 | 0.6289 | 0.8272 | 0.5652 |
| no augmentation | ntl_boost4 | 0.5873 | 0.6075 | 0.8277 | 0.4746 |
| mild aug mean | ntl_boost4 | 0.6085 | 0.6270 | 0.8517 | 0.4872 |

Interpretation:

- Mild image-level augmentation is useful when view embeddings are averaged per
  image before classifier training.
- `mild aug mean + ntl_boost2.5` is the best validation-consistent choice:
  it improves test Macro-F1 from `0.5903` to `0.6289`, and NTL F1 from
  `0.4400` to `0.5652`.
- `mild aug mean + ntl_boost4` has slightly higher test Accuracy and AUROC, but
  lower NTL F1 and lower NST F1 than `ntl_boost2.5`; keep it as a diagnostic
  alternative rather than the primary recommendation.

Current recommendation after this stage:

```text
refined vectors + original top10
+ image-level mild augmentation with mean view aggregation
+ CE class weight ntl_boost2.5
```

### Augmentation-Only View With None Baseline

A follow-up table added the basic no-class-weight (`none`) baseline so the
augmentation effect can be read without focusing only on class weighting.

Local output:

```text
ebtc_image_level_mild_aug_analysis_outputs/augmentation_ablation_with_none.csv
```

| Weight | Augmentation | Test Acc | Test Macro-F1 | Test AUROC | HGC F1 | LGC F1 | NTL F1 | NST F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| none | no_aug | 0.5767 | 0.4987 | 0.8084 | 0.5731 | 0.4630 | 0.0000 | 0.9589 |
| none | aug_expand | 0.4868 | 0.4931 | 0.7558 | 0.4027 | 0.4500 | 0.2500 | 0.8696 |
| none | aug_mean | 0.5556 | 0.5032 | 0.7874 | 0.5062 | 0.4696 | 0.0769 | 0.9600 |
| inverse | no_aug | 0.6032 | 0.5786 | 0.8158 | 0.6036 | 0.4792 | 0.3000 | 0.9315 |
| inverse | aug_expand | 0.5238 | 0.5547 | 0.7694 | 0.4722 | 0.4483 | 0.4727 | 0.8254 |
| inverse | aug_mean | 0.5714 | 0.5326 | 0.8028 | 0.6054 | 0.2985 | 0.3077 | 0.9189 |
| ntl_boost2.5 | aug_expand | 0.5079 | 0.5261 | 0.7732 | 0.4658 | 0.4746 | 0.3774 | 0.7869 |
| ntl_boost2.5 | aug_mean | 0.5979 | 0.6289 | 0.8272 | 0.5333 | 0.4727 | 0.5652 | 0.9444 |
| ntl_boost4 | aug_expand | 0.5132 | 0.5223 | 0.7967 | 0.5161 | 0.4630 | 0.3860 | 0.7241 |
| ntl_boost4 | aug_mean | 0.6085 | 0.6270 | 0.8517 | 0.5440 | 0.6111 | 0.4872 | 0.8657 |

Additional interpretation:

- Augmentation alone is not enough. With `none`, `aug_mean` gives only a tiny
  Macro-F1 increase over no augmentation and NTL F1 remains very low.
- `aug_expand` is not recommended; it is consistently weak despite high
  validation behavior in some settings.
- The useful recipe is not "augmentation only"; it is `aug_mean` plus an
  appropriate NTL-aware class weight.

## 11. Cosine-Only Top-10 Concept Assignment

This stage implements the requested non-training cosine-similarity
classification checks. It uses the concept activation vector directly and does
not train a CBM/CyCL classifier.

Local output:

```text
ebtc_cosine_topk_assignment_outputs/
```

Two rules were evaluated:

- `majority_vote`: take the global top 10 concepts and predict the class that
  appears most often among those concepts.
- `class_average`: take the global top 10 concepts, average cosine similarity
  within each concept class, and predict the class with the highest average.

Main val/test results:

| Stage | Split | Rule | Acc | Macro-F1 | Macro-AUROC |
|---|---|---|---:|---:|---:|
| refined_vectors_original_top10 | val | majority_vote | 0.5617 | 0.4756 | 0.8080 |
| refined_vectors_original_top10 | val | class_average | 0.5325 | 0.4364 | 0.7649 |
| refined_vectors_original_top10 | test | majority_vote | 0.5291 | 0.4334 | 0.7221 |
| refined_vectors_original_top10 | test | class_average | 0.4974 | 0.3870 | 0.7103 |
| original_vectors_original_top10 | test | majority_vote | 0.5503 | 0.4258 | 0.7262 |
| original_vectors_original_top10 | test | class_average | 0.4656 | 0.3936 | 0.6602 |
| refined_vectors_refined_top10 | test | majority_vote | 0.5397 | 0.4444 | 0.7280 |
| refined_vectors_refined_top10 | test | class_average | 0.5132 | 0.3654 | 0.6448 |

For the current main representation, `refined_vectors_original_top10`,
majority vote is better than class-average on test (`0.4334` vs `0.3870`
Macro-F1), but both are clearly weaker than the trained CBM results.

Current-main test per-class F1:

| Rule | HGC | LGC | NTL | NST |
|---|---:|---:|---:|---:|
| majority_vote | 0.6105 | 0.0000 | 0.3137 | 0.8095 |
| class_average | 0.5744 | 0.0000 | 0.1304 | 0.8434 |

The failure mode is LGC. For true LGC test images, the global top 10 concepts
contain on average `7.23` HGC concepts and only `0.92` LGC concepts, so direct
top-k rules almost never predict LGC. This supports the need for a learned
CBM classifier: the concept vector contains useful signal, but the raw top-k
class labels are too noisy and too affected by shared HGC/LGC structure.

## 12. MLP Clean Concept-Vector Prediction

The latest follow-up tested whether a small predictor can clean the noisy
40-d concept activation vector before top10 rule-based classification.

Script:

```text
ebtc_concept_vector_prediction.py
```

Target construction:

- true-class concept positions keep their image-concept cosine values;
- all other class concept positions are set to `-1`;
- MLP maps refined image embeddings directly to the 40-d target vector.

Two settings were run:

```text
positive_weight = 1.0
positive_weight = 3.0
```

Main test results:

| Source | Rule | Test Acc | Test Macro-F1 | Test AUROC |
|---|---|---:|---:|---:|
| raw cosine activation | majority_vote | 0.5291 | 0.4334 | 0.7221 |
| raw cosine activation | class_average | 0.4974 | 0.3870 | 0.7103 |
| predicted ensemble, pos_w=1 | majority_vote | 0.4233 | 0.4115 | 0.6088 |
| predicted ensemble, pos_w=1 | class_average | 0.4233 | 0.4112 | 0.6794 |
| predicted ensemble, pos_w=3 | majority_vote | 0.4286 | 0.4151 | 0.6131 |
| predicted ensemble, pos_w=3 | class_average | 0.4286 | 0.4151 | 0.6981 |
| predicted seed44, pos_w=3 | majority_vote | 0.4444 | 0.4547 | 0.6294 |

Interpretation:

- The MLP learns the train/validation target strongly, with validation
  Macro-F1 around `0.88-0.91`, but test performance does not transfer.
- The best single seed is slightly above raw top10 Macro-F1, but the effect is
  unstable and the 3-seed ensemble is worse than the raw cosine activation.
- The predicted vector suppresses false-class dimensions, but also pulls
  true-class dimensions far below their intended cosine range.
- This confirms that the raw vector is noisy, but direct hard-mask vector
  prediction is not yet a reliable replacement.

Detailed result note:

```text
docs/CONCEPT_VECTOR_PREDICTION_RESULT.md
```
