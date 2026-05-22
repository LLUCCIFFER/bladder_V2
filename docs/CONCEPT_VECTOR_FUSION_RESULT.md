# Concept Vector Fusion Result

## Goal

The repaired `z_margin` concept bank fixed the complete LGC collapse, but it
also weakened HGC and did not improve NTL enough. The raw `original_top10`
bank has the opposite behavior: it preserves more HGC and NTL evidence but
almost never predicts LGC.

This experiment tests whether the two concept-vector sources are complementary
without training a high-capacity downstream model.

## Script

```text
ebtc_concept_vector_fusion.py
```

Local output:

```text
ebtc_concept_vector_fusion_outputs/
```

## Inputs

Raw source:

```text
ebtc_embedding_refinement_stage_conservative_outputs/refined_vectors_original_whitelist/
```

Repaired source:

```text
ebtc_confusion_aware_bank_repair_outputs/banks/z_margin/
```

Both sources keep the same interpretable format:

```text
4 classes x 10 concepts = 40 concepts
HGC positions 0-9
LGC positions 10-19
NTL positions 20-29
NST positions 30-39
```

## Method

For each image, compute two top10 class-count evidence vectors:

```text
c_raw(x)      = class counts from current refined original_top10 bank
c_repaired(x) = class counts from repaired z_margin bank
```

Then evaluate:

```text
c_fused(x) = (1 - w) * c_raw(x) + w * c_repaired(x)
```

The prediction is the class with the largest fused top10 count evidence.

Validation selects `w` by majority-vote Macro-F1, with an accuracy tie-break
inside the configured Macro-F1 tolerance.

## Validation Selection

Selected source:

```text
count_fusion_0.85
```

Top validation rows:

| Source | Rule | w | Acc | Macro-F1 | AUROC | HGC F1 | LGC F1 | NTL F1 | NST F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| count_fusion_0.85 | majority_vote | 0.85 | 0.6429 | 0.6022 | 0.8497 | 0.5833 | 0.5185 | 0.4571 | 0.8500 |
| count_fusion_0.90 | majority_vote | 0.90 | 0.6429 | 0.6022 | 0.8505 | 0.5833 | 0.5185 | 0.4571 | 0.8500 |
| count_fusion_0.95 | majority_vote | 0.95 | 0.6429 | 0.6022 | 0.8519 | 0.5833 | 0.5185 | 0.4571 | 0.8500 |
| count_fusion_1.00 | majority_vote | 1.00 | 0.6396 | 0.5998 | 0.8102 | 0.5773 | 0.5106 | 0.4571 | 0.8543 |
| repaired_majority | majority_vote | 1.00 | 0.6364 | 0.5876 | 0.8102 | 0.5729 | 0.5158 | 0.4118 | 0.8500 |
| raw_majority | majority_vote | 0.00 | 0.5617 | 0.4756 | 0.8080 | 0.5932 | 0.0351 | 0.4324 | 0.8416 |

## Frozen Test Results

| Vector source | Classifier | Acc | Macro-F1 | AUROC | HGC F1 | LGC F1 | NTL F1 | NST F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| mild-aug mean refined vectors | regular CBM, ntl_boost2.5 | 0.5979 | 0.6289 | 0.8272 | 0.5333 | 0.4727 | 0.5652 | 0.9444 |
| raw_original_top10 | top10 majority_vote | 0.5291 | 0.4334 | 0.7221 | 0.6105 | 0.0000 | 0.3137 | 0.8095 |
| raw_original_top10 | top10 class_average | 0.4974 | 0.3870 | 0.7103 | 0.5744 | 0.0000 | 0.1304 | 0.8434 |
| repaired z_margin bank | top10 majority_vote | 0.5661 | 0.5588 | 0.7931 | 0.5135 | 0.5234 | 0.2917 | 0.9067 |
| repaired z_margin bank | top10 class_average | 0.4921 | 0.4035 | 0.7376 | 0.5596 | 0.0882 | 0.1053 | 0.8608 |
| raw + repaired count fusion | top10 majority_vote, w=0.85 | 0.5767 | 0.5648 | 0.8144 | 0.5333 | 0.5333 | 0.2979 | 0.8947 |

The selected fusion improves the top10 majority result over both individual
concept-vector sources:

| Metric | Raw original_top10 | Repaired z_margin | Fusion w=0.85 |
|---|---:|---:|---:|
| Accuracy | 0.5291 | 0.5661 | 0.5767 |
| Macro-F1 | 0.4334 | 0.5588 | 0.5648 |
| AUROC | 0.7221 | 0.7931 | 0.8144 |
| HGC F1 | 0.6105 | 0.5135 | 0.5333 |
| LGC F1 | 0.0000 | 0.5234 | 0.5333 |
| NTL F1 | 0.3137 | 0.2917 | 0.2979 |
| NST F1 | 0.8095 | 0.9067 | 0.8947 |

## Test Confusion Matrices

Raw original_top10, top10 majority:

| true\pred | HGC | LGC | NTL | NST |
|---|---:|---:|---:|---:|
| HGC | 58 | 0 | 11 | 5 |
| LGC | 45 | 0 | 4 | 4 |
| NTL | 13 | 0 | 8 | 4 |
| NST | 0 | 0 | 3 | 34 |

Repaired z_margin, top10 majority:

| true\pred | HGC | LGC | NTL | NST |
|---|---:|---:|---:|---:|
| HGC | 38 | 24 | 11 | 1 |
| LGC | 23 | 28 | 2 | 0 |
| NTL | 13 | 2 | 7 | 3 |
| NST | 0 | 0 | 3 | 34 |

Fusion w=0.85, top10 majority:

| true\pred | HGC | LGC | NTL | NST |
|---|---:|---:|---:|---:|
| HGC | 40 | 22 | 10 | 2 |
| LGC | 23 | 28 | 2 | 0 |
| NTL | 13 | 2 | 7 | 3 |
| NST | 0 | 0 | 3 | 34 |

Readout:

- LGC remains repaired: `28/53` true-LGC test images are predicted as LGC.
- HGC recovers slightly compared with repaired z_margin: `38/74 -> 40/74`.
- NST remains stable.
- NTL is still not fixed: `7/25` true-NTL test images are predicted as NTL.

## Top10 Evidence Counts

Mean top10 class-count evidence on the test split:

| Source | True class | HGC evidence | LGC evidence | NTL evidence | NST evidence |
|---|---|---:|---:|---:|---:|
| raw_original_top10 | HGC | 5.85 | 1.68 | 2.07 | 0.41 |
| raw_original_top10 | LGC | 7.23 | 0.92 | 1.08 | 0.77 |
| raw_original_top10 | NTL | 3.88 | 0.92 | 3.68 | 1.52 |
| raw_original_top10 | NST | 0.08 | 0.08 | 0.76 | 9.08 |
| repaired z_margin | HGC | 4.35 | 3.78 | 1.59 | 0.27 |
| repaired z_margin | LGC | 4.30 | 5.15 | 0.34 | 0.21 |
| repaired z_margin | NTL | 3.72 | 2.16 | 2.72 | 1.40 |
| repaired z_margin | NST | 0.14 | 0.00 | 0.59 | 9.27 |
| fusion w=0.85 | HGC | 4.58 | 3.47 | 1.67 | 0.29 |
| fusion w=0.85 | LGC | 4.74 | 4.52 | 0.45 | 0.29 |
| fusion w=0.85 | NTL | 3.74 | 1.97 | 2.86 | 1.42 |
| fusion w=0.85 | NST | 0.13 | 0.01 | 0.62 | 9.24 |

The fusion result confirms the main geometry issue:

- LGC is now competitive with HGC, but still close.
- NTL evidence remains weaker than HGC evidence for true-NTL images.
- NST is clean in all variants.

## Additional Bias Check

A small LGC/NTL class-bias grid was checked on top of the fused count evidence.
Some NTL-biased settings improved frozen-test NTL F1 in diagnostic mode, but
validation selection did not choose them robustly and several variants were
sensitive to near-tie behavior in floating-point fused counts.

Therefore, the bias grid is not promoted as the final result. It is useful
evidence that NTL may benefit from a dedicated correction, but the correction
should be implemented as a train-only NTL-focused bank repair rather than as a
post-hoc test-tuned bias.

## Conclusion

This is a modest but real improvement over the previous vector repair:

```text
raw top10 Macro-F1:      0.4334
repaired top10 Macro-F1: 0.5588
fusion top10 Macro-F1:   0.5648
```

However, the result is still below the current regular CBM classifier
(`0.6289` Macro-F1). The remaining bottleneck is no longer complete LGC
collapse. It is now:

1. HGC/LGC overlap: repaired LGC concepts also activate for many HGC images.
2. NTL weakness: true-NTL images still contain more HGC evidence than NTL
   evidence after repair.
3. Class-average remains unreliable; the useful interpretable rule is still
   top10 majority-style evidence.

## Recommended Next Step

The next improvement should focus on the vector source, not on more downstream
voting rules:

1. Add an NTL-specific bank-repair stage.
   - Select NTL concepts by pairwise margin against HGC and LGC, not only by
     own-vs-average margin.
   - Keep the same 10 NTL concept slots.
   - Validate by true-NTL top10 NTL evidence and NTL F1.

2. Add a conservative residual vector calibrator after the repaired bank.
   - Input: raw 40-d repaired-bank cosine vector.
   - Output: `raw_vector + scale * tanh(residual_vector)`.
   - Keep `scale` small, for example `0.05` to `0.10`.
   - Train with top10/ranking losses plus a preservation loss, not hard `-1`
     block MSE.

3. Keep reporting two tracks.
   - Regular CBM classifier result.
   - Interpretable vector-only top10 majority/class-average diagnostics.

The current fusion result should be treated as the best interpretable
vector-only result so far, but not as the final solution for the whole system.
