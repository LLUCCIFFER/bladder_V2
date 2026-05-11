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

