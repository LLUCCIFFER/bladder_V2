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
- fixed concept order: HGC positions `0-9`, LGC positions `10-19`, NTL
  positions `20-29`, NST positions `30-39`
- target vector: true-class concept block keeps its image-concept cosine
  values; all other class blocks are set to `-1`
- predictor: MLP, 512 -> 256 -> 256 -> 40
- seeds: 42, 43, 44
- evaluation: same top10 `majority_vote` and `class_average` rules used by
  the cosine-only assignment diagnostic

Two loss-weight settings were run:

- `positive_weight=1.0`: plain MSE over all 40 dimensions
- `positive_weight=3.0`: upweight the 10 true-class concept positions to
  counter the 30 false-class positions

## Main Test Results

The strict MSE/no-tanh run is stored in:

```text
ebtc_concept_vector_prediction_strict_outputs/
```

Final comparison requested for reporting:

| Vector source | Classifier | Acc | Macro-F1 | HGC F1 | LGC F1 | NTL F1 | NST F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| mild-aug mean refined vectors | regular CBM, ntl_boost2.5, 3-seed ensemble | 0.5979 | 0.6289 | 0.5333 | 0.4727 | 0.5652 | 0.9444 |
| raw cosine activation | top10 majority_vote | 0.5291 | 0.4334 | 0.6105 | 0.0000 | 0.3137 | 0.8095 |
| raw cosine activation | top10 class_average | 0.4974 | 0.3870 | 0.5744 | 0.0000 | 0.1304 | 0.8434 |
| predicted clean vector | top10 majority_vote, 3-seed mean | 0.4233 | 0.4115 | 0.2308 | 0.4384 | 0.0625 | 0.9143 |
| predicted clean vector | top10 class_average, 3-seed mean | 0.4233 | 0.4112 | 0.2326 | 0.4354 | 0.0625 | 0.9143 |
| predicted clean vector | top10 majority_vote, best seed44 diagnostic | 0.4233 | 0.4431 | 0.2034 | 0.4267 | 0.2439 | 0.8986 |

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

For the strict MSE/no-tanh run, the predicted-vector 3-seed mean changes the
failure mode:

- raw top10 majority never predicts LGC on test, so LGC F1 is `0.0000`;
- predicted-vector top10 majority recovers LGC F1 to `0.4384`;
- HGC F1 drops sharply from `0.6105` to `0.2308`;
- NTL F1 drops from `0.3137` to `0.0625`;
- NST remains high (`0.9143`).

For `positive_weight=3.0`, the best single seed is seed 44:

| Rule | HGC F1 | LGC F1 | NTL F1 | NST F1 |
|---|---:|---:|---:|---:|
| majority_vote | 0.2419 | 0.4521 | 0.2105 | 0.9143 |
| class_average | 0.2400 | 0.4521 | 0.1622 | 0.9143 |

The predicted vector partially fixes the raw-vector LGC collapse, but it loses
too much HGC and NTL performance. The ensemble makes this worse rather than
stabilizing it.

## Top10 Concept Class Counts

Mean concept-class counts among the global top10 concepts on the test split:

| Vector | True class | HGC concepts | LGC concepts | NTL concepts | NST concepts |
|---|---|---:|---:|---:|---:|
| raw cosine | HGC | 5.85 | 1.68 | 2.07 | 0.41 |
| raw cosine | LGC | 7.23 | 0.92 | 1.08 | 0.77 |
| raw cosine | NTL | 3.88 | 0.92 | 3.68 | 1.52 |
| raw cosine | NST | 0.08 | 0.08 | 0.76 | 9.08 |
| predicted 3-seed mean | HGC | 2.08 | 7.54 | 0.24 | 0.14 |
| predicted 3-seed mean | LGC | 3.77 | 6.08 | 0.15 | 0.00 |
| predicted 3-seed mean | NTL | 7.84 | 1.76 | 0.40 | 0.00 |
| predicted 3-seed mean | NST | 0.27 | 0.27 | 0.81 | 8.65 |

Readout:

- LGC is no longer dominated by HGC concepts; LGC concepts rise from `0.92`
  to `6.08` in true-LGC top10 lists.
- The gain is not class-specific enough: true-HGC images now contain `7.54`
  LGC concepts on average, which causes HGC to collapse into LGC.
- NTL becomes less stable. True-NTL top10 lists shift from `3.68` NTL concepts
  to only `0.40` NTL concepts and become HGC-dominated.
- NST remains clean, with `8.65` NST concepts on average after prediction.

## Confusion Pattern

Top10 majority vote confusion matrices on the test split:

Raw cosine activation:

| true\pred | HGC | LGC | NTL | NST |
|---|---:|---:|---:|---:|
| HGC | 58 | 0 | 11 | 5 |
| LGC | 45 | 0 | 4 | 4 |
| NTL | 13 | 0 | 8 | 4 |
| NST | 0 | 0 | 3 | 34 |

Predicted clean vector, 3-seed mean:

| true\pred | HGC | LGC | NTL | NST |
|---|---:|---:|---:|---:|
| HGC | 15 | 56 | 2 | 1 |
| LGC | 20 | 32 | 1 | 0 |
| NTL | 20 | 4 | 1 | 0 |
| NST | 1 | 1 | 3 | 32 |

The vector prediction network corrects the specific "never predict LGC"
problem, but it overcorrects toward LGC for HGC and toward HGC for NTL.

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
ebtc_concept_vector_prediction_strict_outputs/
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
