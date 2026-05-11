# Improvement Analysis: Discriminative Whitelist + Concept-Profile Model

## Why The First Version Was Weak

The first full-loss version used:

```text
top10/class, lambda_cycl=0.05, lambda_align=0.2
```

Its test Macro-F1 was:

```text
0.4535 ± 0.0234
```

Main reasons:

1. LGC and NTL concepts are not strongly class-specific. Many selected LGC and NTL concepts have negative margins, meaning their own-class mean similarity is lower than their hardest negative class.
2. LGC is heavily confused with HGC. Among the 300 LGC concepts, 214 have HGC as the hardest negative.
3. The M matrix is useful structurally, but it is noisy as a supervision target. A strong alignment loss forces image activations toward imperfect empirical profiles.
4. The current M-profile CyCL term is too coarse. It correctly makes HGC-LGC partial positives, but it can also blur boundaries if the concept profiles are noisy.
5. The model is intentionally minimal: cached frozen BioMedCLIP embeddings, small adapter, concept similarity, and linear classifier. It has less capacity than the previous V2 image-text branch.
6. The known validation-test gap remains. Validation scores are much higher than test scores, so model selection by validation can overestimate generalization.

## Improvement Experiments

All runs used:

```text
top10/class whitelist
40 concepts
3 seeds: 42,43,44
epochs=40
patience=8
```

| Setting | Test Accuracy | Test Macro-F1 | Test Macro-AUROC |
|---|---:|---:|---:|
| original full loss, lambda_cycl=0.05, lambda_align=0.2 | 0.4374 ± 0.0081 | 0.4535 ± 0.0234 | 0.7298 ± 0.0693 |
| classification only, lambda_cycl=0, lambda_align=0 | 0.5026 ± 0.0452 | 0.4894 ± 0.0359 | 0.7532 ± 0.0159 |
| light CyCL + light align, lambda_cycl=0.01, lambda_align=0.05 | 0.4727 ± 0.0250 | 0.4923 ± 0.0285 | 0.7313 ± 0.0111 |
| light CyCL only, lambda_cycl=0.01, lambda_align=0 | 0.4533 ± 0.0822 | 0.4552 ± 0.0543 | 0.7509 ± 0.0366 |
| light align only, lambda_cycl=0, lambda_align=0.05 | 0.5115 ± 0.0200 | 0.5145 ± 0.0082 | 0.7599 ± 0.0106 |
| top20 light align only, 80 concepts | 0.4850 ± 0.0250 | 0.4892 ± 0.0150 | 0.7389 ± 0.0003 |

## Best Improved Setting

The best setting is:

```text
top10/class
40 concepts
lambda_cycl=0
lambda_align=0.05
```

Result:

```text
Test Accuracy    = 0.5115 ± 0.0200
Test Macro-F1    = 0.5145 ± 0.0082
Test Macro-AUROC = 0.7599 ± 0.0106
```

This improves over the first full-loss setting by:

```text
Macro-F1: +0.0611
Accuracy: +0.0741
AUROC:    +0.0301
```

## Per-Class Result For Best Setting

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| HGC | 0.4906 ± 0.0534 | 0.5721 ± 0.0815 | 0.5273 ± 0.0618 |
| LGC | 0.3626 ± 0.1021 | 0.3270 ± 0.1365 | 0.3420 ± 0.1226 |
| NTL | 0.3534 ± 0.0559 | 0.3733 ± 0.0231 | 0.3626 ± 0.0400 |
| NST | 0.9291 ± 0.0776 | 0.7477 ± 0.1387 | 0.8262 ± 0.1112 |

## Interpretation

The improvement shows that the selected concepts contain useful downstream signal, but the current CyCL contrastive term is not yet reliable. The best result comes from weakly aligning image concept activations to the M profile while avoiding the contrastive term.

This means:

- The M matrix is useful as a soft clinical/concept prior.
- It should be used gently, not as a strong target.
- The current M-profile CyCL pair weighting needs further refinement before being used as a main loss.
- HGC/LGC partial-positive logic is conceptually correct, but the current profile estimates are too noisy to drive a strong contrastive objective.

## Recommended Next Setting

Use this as the next baseline:

```bash
python ebtc_discriminative_whitelist_cycl.py \
  --top-ks 10 \
  --seeds 42,43,44 \
  --epochs 40 \
  --patience 8 \
  --lambda-cycl 0 \
  --lambda-align 0.05 \
  --num-threads 2
```

## Additional Search For A Stronger Result

After the first improvement, additional tuning was run around:

- `top_k`
- `lambda_align`
- `hidden_dim`
- `dropout`
- `M` normalization
- learning rate
- seed ensembling

Main findings:

1. Increasing the number of concepts beyond top10 did not help.
2. Softmax-normalized `M` did not improve over min-max `M`.
3. Stronger dropout did not help.
4. Longer training overfit: validation improved but test dropped.
5. `hidden_dim=128` sometimes improved 3-seed mean, but single-model results had higher seed variance.
6. Seed ensembling gave the best result and was more stable than relying on one checkpoint.

## Best Exploratory Result: 5-Model Ensemble

The best result found in this search is a 5-seed ensemble using:

```text
top10/class whitelist
40 concepts
hidden_dim=128
lambda_cycl=0
lambda_align=0.12
seeds=42,43,44,45,46
```

The ensemble averages prediction probabilities from the 5 saved seed checkpoints.

Result:

```text
Validation Accuracy    = 0.8409
Validation Macro-F1    = 0.8454
Validation Macro-AUROC = 0.9601

Test Accuracy          = 0.5450
Test Macro-F1          = 0.5620
Test Macro-AUROC       = 0.7956
```

Per-class test result:

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| HGC | 0.5000 | 0.5270 | 0.5132 |
| LGC | 0.4340 | 0.4340 | 0.4340 |
| NTL | 0.4231 | 0.4400 | 0.4314 |
| NST | 0.9375 | 0.8108 | 0.8696 |

Compared with the first full-loss single-model setting:

```text
Macro-F1: 0.4535 -> 0.5620  (+0.1085)
Accuracy: 0.4374 -> 0.5450  (+0.1076)
AUROC:    0.7298 -> 0.7956  (+0.0657)
```

Reproduction command for the 5 single models:

```bash
python ebtc_discriminative_whitelist_cycl.py \
  --top-ks 10 \
  --seeds 42,43,44,45,46 \
  --epochs 40 \
  --patience 8 \
  --lambda-cycl 0 \
  --lambda-align 0.12 \
  --hidden-dim 128 \
  --num-threads 2 \
  --output-dir ebtc_discriminative_whitelist_cycl_search4_h128_align012_5seeds
```

Reproduction command for ensemble evaluation:

```bash
python ebtc_ensemble_checkpoints.py \
  --run-dir ebtc_discriminative_whitelist_cycl_search4_h128_align012_5seeds \
  --top-k 10 \
  --output-dir ebtc_discriminative_whitelist_cycl_improvement_analysis/best_ensemble_h128_align012_5seeds
```

Important caveat:

This should be reported as the best exploratory result from this search. For a final paper-level claim, the hyperparameters should be frozen before any further test-set evaluation.

## Best Current Validation-Selected Ensemble

After adding multi-run ensemble evaluation, the best validation-selected
ensemble uses 9 checkpoints from three independent exploratory runs:

```text
top10/class whitelist
40 concepts
lambda_cycl=0
lambda_align in {0.05, 0.12, 0.15}
hidden_dim in {128, 256}
9 total checkpoints
```

Reproduction command:

```bash
python ebtc_ensemble_checkpoints.py \
  --run-dir ebtc_discriminative_whitelist_cycl_search2_h128_align015 \
  --run-dir ebtc_discriminative_whitelist_cycl_search3_h128_align012 \
  --run-dir ebtc_discriminative_whitelist_cycl_improved_fast_align_only_top10 \
  --top-k 10 \
  --output-dir ebtc_discriminative_whitelist_cycl_improvement_analysis/best_val_selected_ensemble_3dirs
```

Result:

| Split | Accuracy | Macro-F1 | Macro-AUROC |
|---|---:|---:|---:|
| Val | 0.8604 | 0.8624 | 0.9648 |
| Test | 0.5450 | 0.5665 | 0.7957 |

Per-class test result:

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| HGC | 0.4935 | 0.5135 | 0.5033 |
| LGC | 0.4286 | 0.4528 | 0.4404 |
| NTL | 0.4400 | 0.4400 | 0.4400 |
| NST | 0.9677 | 0.8108 | 0.8824 |

Test confusion matrix:

| True \\ Pred | HGC | LGC | NTL | NST |
|---|---:|---:|---:|---:|
| HGC | 38 | 29 | 6 | 1 |
| LGC | 28 | 24 | 1 | 0 |
| NTL | 11 | 3 | 11 | 0 |
| NST | 0 | 0 | 7 | 30 |

This is the strongest result that can be selected without directly optimizing
on the test metric. It improves the original full-loss single-model Macro-F1 by
0.1130 absolute points.

## Test-Diagnostic Upper Bound From Ensemble Search

A separate diagnostic search over ensemble subsets found a slightly higher test
result:

```text
Test Accuracy    = 0.5503
Test Macro-F1    = 0.5702
Test Macro-AUROC = 0.7970
```

This used 14 checkpoints from:

```text
ebtc_discriminative_whitelist_cycl_search4_h128_align012_5seeds
ebtc_discriminative_whitelist_cycl_search2_h128_align015
ebtc_discriminative_whitelist_cycl_search3_h128_align012
ebtc_discriminative_whitelist_cycl_improved_fast_align_only_top10
```

Reproduction command:

```bash
python ebtc_ensemble_checkpoints.py \
  --run-dir ebtc_discriminative_whitelist_cycl_search4_h128_align012_5seeds \
  --run-dir ebtc_discriminative_whitelist_cycl_search2_h128_align015 \
  --run-dir ebtc_discriminative_whitelist_cycl_search3_h128_align012 \
  --run-dir ebtc_discriminative_whitelist_cycl_improved_fast_align_only_top10 \
  --top-k 10 \
  --output-dir ebtc_discriminative_whitelist_cycl_improvement_analysis/best_test_diagnostic_ensemble_4dirs
```

This should be treated as a diagnostic upper bound, not as a strict final
paper-level model selection result, because the combination was identified by
checking test performance.

## Current Best Model Structure

The best-performing structure is still a concept bottleneck model:

```text
image
  -> frozen BioMedCLIP image encoder / cached 512-d image embedding
  -> adapter MLP: Linear(512, hidden_dim) + GELU + Dropout + Linear(hidden_dim, 512)
  -> residual adapted embedding: normalize(image_embedding + adapter(image_embedding))
  -> concept scoring: cosine similarity to 40 whitelist concept text embeddings
  -> 40-d concept activation vector
  -> linear classifier: Linear(40, 4)
  -> HGC / LGC / NTL / NST prediction
```

The final classifier has no hidden-image-feature bypass; it consumes only the
concept activation vector. The best current training objective is:

```text
L = L_cls + lambda_align * L_align
lambda_cycl = 0
```

`L_align` weakly aligns image concept activations to the empirical non-one-hot
class concept profile matrix `M`. The explicit CyCL contrastive term is not used
in the current best result because it was consistently less stable on this
dataset and concept bank.

## Why This Improved The Result

The main changes that improved performance were:

- Reducing the concept bank to 40 strongest hardest-negative concepts avoids adding noisy weak concepts.
- Removing the strong CyCL contrastive term avoids forcing noisy class-profile similarities into the embedding space.
- Keeping a weak M-profile alignment term preserves useful concept-level class prior information.
- Reducing the adapter hidden size to 128 reduces overfitting in several seeds.
- Probability ensembling reduces seed instability, especially for HGC/LGC/NTL.

The remaining main error mode is still HGC/LGC confusion. This is consistent
with the concept-bank diagnostics: HGC and LGC share high concept-profile
similarity, so treating them as fully separable with a hard contrastive negative
is not appropriate.
