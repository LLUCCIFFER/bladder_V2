# Concept Vector Prediction Result

## Setup

Motivation: the raw 40-d concept activation vector is noisy, so downstream
classification, class weighting, and augmentation may all be operating on an
unclean bottleneck representation.

New script:

```text
ebtc_concept_vector_prediction.py
```

Default input:

- image vectors: refined BioMedCLIP image embeddings
- concept bank: original top10 whitelist, 40 concepts total
- target vector: true-class concept positions keep their image-concept cosine
  values; all other class concept positions are set to `-1`
- predictor: MLP, 512 -> 256 -> 256 -> 40
- seeds: 42, 43, 44
- evaluation: same top10 `majority_vote` and `class_average` rules used by
  the cosine-only assignment diagnostic

Two loss-weight settings were run:

- `positive_weight=1.0`: plain MSE over all 40 dimensions
- `positive_weight=3.0`: upweight the 10 true-class concept positions to
  counter the 30 false-class positions

## Main Test Results

| Source | Rule | Test Acc | Test Macro-F1 | Test AUROC |
|---|---|---:|---:|---:|
| raw cosine activation | majority_vote | 0.5291 | 0.4334 | 0.7221 |
| raw cosine activation | class_average | 0.4974 | 0.3870 | 0.7103 |
| predicted ensemble, pos_w=1 | majority_vote | 0.4233 | 0.4115 | 0.6088 |
| predicted ensemble, pos_w=1 | class_average | 0.4233 | 0.4112 | 0.6794 |
| predicted seed44, pos_w=1 | majority_vote | 0.4233 | 0.4431 | 0.6233 |
| predicted seed44, pos_w=1 | class_average | 0.4233 | 0.4431 | 0.6843 |
| predicted ensemble, pos_w=3 | majority_vote | 0.4286 | 0.4151 | 0.6131 |
| predicted ensemble, pos_w=3 | class_average | 0.4286 | 0.4151 | 0.6981 |
| predicted seed44, pos_w=3 | majority_vote | 0.4444 | 0.4547 | 0.6294 |
| predicted seed44, pos_w=3 | class_average | 0.4392 | 0.4421 | 0.7021 |

## Validation/Test Gap

The MLP strongly fits the train/validation target but does not generalize to
the frozen test split.

| Setting | Val Macro-F1 Range | Test Macro-F1 Range |
|---|---:|---:|
| pos_w=1 predicted seeds | 0.8880-0.8956 | 0.3806-0.4431 |
| pos_w=3 predicted seeds | 0.8822-0.9127 | 0.3874-0.4547 |

This is a large validation/test gap. The predicted vector is not a stable
drop-in replacement for the raw concept activation vector.

## Per-Class Pattern

For `positive_weight=3.0`, the best single seed is seed 44:

| Rule | HGC F1 | LGC F1 | NTL F1 | NST F1 |
|---|---:|---:|---:|---:|
| majority_vote | 0.2419 | 0.4521 | 0.2105 | 0.9143 |
| class_average | 0.2400 | 0.4521 | 0.1622 | 0.9143 |

The predicted vector partially fixes the raw-vector LGC collapse, but it loses
too much HGC and NTL performance. The ensemble makes this worse rather than
stabilizing it.

## Vector Quality

The plain raw vector has zero target error on true-class positions by
construction, but false-class concept positions are high and noisy:

| Source | Test MSE All | True-Pos MSE | False-Pos MSE | Mean True Pos | Mean False Pos |
|---|---:|---:|---:|---:|---:|
| raw cosine activation | 0.8949 | 0.0000 | 1.1933 | 0.1199 | 0.0919 |
| predicted ensemble, pos_w=1 | 0.2282 | 0.5071 | 0.1352 | -0.5166 | -0.7844 |
| predicted ensemble, pos_w=3 | 0.2391 | 0.3745 | 0.1940 | -0.4051 | -0.7194 |

The MLP does suppress false-class dimensions, but it also pulls true-class
concept dimensions far below their intended cosine range. This explains why the
output vector is cleaner in an MSE sense but worse for top10 concept assignment.

## Interpretation

- The raw concept vector is indeed noisy: false-class concept activations are
  close to true-class activations.
- Direct supervised vector cleaning with this MLP target is not sufficient.
  It learns validation-specific class structure and does not transfer to test.
- Upweighting true-class positions helps slightly but does not solve the
  generalization problem.
- The best predicted-vector row (`pos_w=3`, seed 44, majority vote) reaches
  test Macro-F1 `0.4547`, only slightly above raw majority vote `0.4334`, and
  is not stable across seeds.

## Local Outputs

```text
ebtc_concept_vector_prediction_outputs/
ebtc_concept_vector_prediction_posw3_outputs/
```

Key files:

- `concept_vector_prediction_results.csv`
- `concept_vector_prediction_per_class.csv`
- `concept_vector_prediction_quality.csv`
- `concept_vector_prediction_report.md`
- `training/seed_*/best_checkpoint.pt`

## Next Engineering Direction

The next version should avoid training only against a hard masked target. More
promising variants are:

- predict a class-conditioned residual correction over the raw concept vector
  instead of predicting all 40 values from scratch;
- use class-balanced sampling or explicit class-balanced vector loss;
- add a margin/ranking loss that only requires true-class concepts to rank
  above false-class concepts, instead of forcing false concepts to exactly `-1`;
- validate with top10 Macro-F1, not only vector MSE, while still keeping test
  frozen for final reporting.
