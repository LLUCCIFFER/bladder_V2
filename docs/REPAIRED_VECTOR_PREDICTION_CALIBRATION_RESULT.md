# Repaired Vector Prediction and Calibration Result

## Goal

This diagnostic checks whether the repaired `z_margin` concept bank can be
further improved by adding learned vector prediction or conservative vector
calibration.

Two methods were tested:

1. Hard `-1` MLP prediction.
   - Input: refined 512-d BioMedCLIP image embedding.
   - Output: 40-d repaired-bank concept vector.
   - Target: keep true-class repaired-bank cosine values and set all other
     concept positions to `-1`.

2. Conservative residual calibration.
   - Input: direct repaired-bank cosine vector.
   - Output: `raw_vector + scale * tanh(MLP(raw_vector))`.
   - Tested residual scales: `0.01` and `0.02`.
   - Training uses class-block ranking/CE plus strong preservation and residual
     penalties.

The raw+repaired fusion tests keep the previously selected fixed fusion weight:

```text
c_fused = 0.15 * c_raw_original_top10 + 0.85 * c_repaired_source
```

## Script

```text
ebtc_repaired_vector_prediction_calibration.py
```

Local output:

```text
ebtc_repaired_vector_prediction_calibration_outputs/
```

## Main Frozen-Test Results

| Vector source | Classifier | Acc | Macro-F1 | AUROC | HGC F1 | LGC F1 | NTL F1 | NST F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| direct repaired z_margin cosine | top10 majority_vote | 0.5661 | 0.5588 | 0.7931 | 0.5135 | 0.5234 | 0.2917 | 0.9067 |
| direct raw + repaired fusion | fixed w=0.85 fusion majority_vote | 0.5767 | 0.5648 | 0.8144 | 0.5333 | 0.5333 | 0.2979 | 0.8947 |
| hard `-1` MLP, positive weight 1, ensemble | top10 majority_vote | 0.4762 | 0.4712 | 0.6419 | 0.3256 | 0.4895 | 0.1579 | 0.9118 |
| hard `-1` MLP, positive weight 1, fusion | fixed w=0.85 fusion majority_vote | 0.4762 | 0.4712 | 0.7350 | 0.3256 | 0.4895 | 0.1579 | 0.9118 |
| hard `-1` MLP, positive weight 3, ensemble | top10 majority_vote | 0.4921 | 0.4924 | 0.6618 | 0.3759 | 0.4818 | 0.2000 | 0.9118 |
| hard `-1` MLP, positive weight 3, fusion | fixed w=0.85 fusion majority_vote | 0.4921 | 0.4924 | 0.7490 | 0.3759 | 0.4818 | 0.2000 | 0.9118 |
| residual scale 0.01, ensemble | top10 majority_vote | 0.5608 | 0.5350 | 0.7620 | 0.4035 | 0.5890 | 0.2500 | 0.8974 |
| residual scale 0.01, fusion | fixed w=0.85 fusion majority_vote | 0.5767 | 0.5539 | 0.7878 | 0.4706 | 0.5942 | 0.2791 | 0.8718 |
| residual scale 0.02, ensemble | top10 majority_vote | 0.5026 | 0.4487 | 0.7401 | 0.1538 | 0.5814 | 0.1622 | 0.8974 |
| residual scale 0.02, fusion | fixed w=0.85 fusion majority_vote | 0.5132 | 0.4779 | 0.7781 | 0.2151 | 0.5749 | 0.2500 | 0.8718 |

## Validation-Test Gap

The hard `-1` MLP reproduced the same problem seen in the earlier
`original_top10` prediction experiment. Validation top10 scores became very
high:

| Source | Val Acc | Val Macro-F1 |
|---|---:|---:|
| hard `-1` MLP, positive weight 1, ensemble | 0.9253 | 0.9268 |
| hard `-1` MLP, positive weight 3, ensemble | 0.9123 | 0.9011 |

But frozen-test Macro-F1 dropped to `0.4712` and `0.4924`. This is a strong
validation-test mismatch, not a useful repaired-vector improvement.

## Confusion Matrices

Direct raw + repaired fusion:

| true\pred | HGC | LGC | NTL | NST |
|---|---:|---:|---:|---:|
| HGC | 40 | 22 | 10 | 2 |
| LGC | 23 | 28 | 2 | 0 |
| NTL | 13 | 2 | 7 | 3 |
| NST | 0 | 0 | 3 | 34 |

Hard `-1` MLP, positive weight 1, fusion:

| true\pred | HGC | LGC | NTL | NST |
|---|---:|---:|---:|---:|
| HGC | 21 | 47 | 6 | 0 |
| LGC | 16 | 35 | 2 | 0 |
| NTL | 16 | 6 | 3 | 0 |
| NST | 2 | 2 | 2 | 31 |

Residual scale 0.01 ensemble, fusion:

| true\pred | HGC | LGC | NTL | NST |
|---|---:|---:|---:|---:|
| HGC | 28 | 37 | 7 | 2 |
| LGC | 9 | 41 | 2 | 1 |
| NTL | 8 | 7 | 6 | 4 |
| NST | 0 | 0 | 3 | 34 |

## Best Single-Seed Diagnostic

The best frozen-test row among learned residual variants was:

| Source | Classifier | Acc | Macro-F1 | HGC F1 | LGC F1 | NTL F1 | NST F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| residual scale 0.01, seed 44, fusion | fixed w=0.85 fusion majority_vote | 0.5926 | 0.5670 | 0.4960 | 0.6212 | 0.2791 | 0.8718 |

This is only `+0.0022` Macro-F1 above direct fusion (`0.5648`) and it is not
robust:

- the 3-seed residual scale 0.01 fusion ensemble is lower (`0.5539` Macro-F1);
- seed 42 is lower (`0.5552` Macro-F1);
- seed 43 is lower (`0.5308` Macro-F1);
- validation Macro-F1 favors hard MLP rows that fail on test, so this single
  seed is not a defensible validation-selected model.

Therefore, this single-seed result should not be promoted as a supplemental
improvement.

## Interpretation

Hard `-1` prediction is still too aggressive. It learns an apparently clean
validation vector, but on frozen test it shifts many HGC and NTL images toward
LGC and damages the decision surface.

Conservative residual calibration is safer, but still does not beat the direct
cosine/fusion baseline in a robust ensemble setting. The main residual effect
is to increase LGC recall, but this comes at the cost of HGC and NTL.

The best current vector-only result remains:

```text
direct raw + repaired fusion, w=0.85
Acc:      0.5767
Macro-F1: 0.5648
AUROC:    0.8144
```

## Conclusion

Do not add hard `-1` MLP prediction or the tested residual calibration as a
positive supplemental result. They are useful negative diagnostics:

- hard `-1` prediction confirms that label-supervised vector generation is
  unstable across the EBTC validation-test split;
- residual calibration confirms that even very small learned shifts can improve
  LGC at the expense of HGC/NTL;
- the next high-value improvement should still be NTL-specific bank repair or
  a more constrained calibration method selected by cross-validation-style
  stability, not by a single validation split.
