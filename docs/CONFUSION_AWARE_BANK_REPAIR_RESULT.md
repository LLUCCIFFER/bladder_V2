# Confusion-Aware Bank Repair Result

## Goal

The previous vector-prediction experiment showed that the raw 40-d concept
activation vector is noisy, but direct hard-mask MSE prediction overcorrects.
This experiment repairs the concept vector at its source: the concept bank.

Constraint:

```text
Keep original_top10 format:
4 classes x 10 concepts = 40 concepts
HGC positions 0-9
LGC positions 10-19
NTL positions 20-29
NST positions 30-39
```

## Script

```text
ebtc_confusion_aware_bank_repair.py
```

Local output:

```text
ebtc_confusion_aware_bank_repair_outputs/
```

## Pipeline

1. Load refined BioMedCLIP image embeddings from:

```text
ebtc_embedding_refinement_stage_conservative_outputs/adapter_refinement/
```

2. Load candidate concepts from the existing `filtered_top300` bank:

```text
ebtc_cycl_retrieval_stage_revised_outputs/banks/filtered_top300/
```

3. Use refined text embeddings for all 1200 filtered candidates:

```text
ebtc_embedding_refinement_stage_conservative_outputs/adapter_refinement/refined_filtered_top300_text_embeddings.npz
```

4. Compute train-only class-wise statistics for every candidate concept:

```text
mu_HGC, mu_LGC, mu_NTL, mu_NST
std_HGC, std_LGC, std_NTL, std_NST
```

5. Score candidates with multiple train-only strategies and select the top 10
per class for each strategy.

6. Select the final strategy by validation top10 majority-vote Macro-F1.

7. Evaluate the selected repaired bank on frozen test with:

- top10 majority vote,
- top10 class-average,
- confusion matrix,
- per-class F1,
- top10 concept class counts.

## Selected Strategy

Validation selected:

```text
z_margin
```

Formula:

```text
z_margin = (mu_own - mean(mu_other_classes)) /
           (std_own + mean(std_other_classes))
```

This is different from the old hardest-negative margin. It favors concepts
whose own-class separation is large relative to train-set variability, rather
than concepts with only a large raw mean difference.

## Validation Strategy Selection

Top10 majority-vote validation results:

| Strategy | Acc | Macro-F1 | HGC F1 | LGC F1 | NTL F1 | NST F1 |
|---|---:|---:|---:|---:|---:|---:|
| raw_original_top10 | 0.5617 | 0.4756 | 0.5932 | 0.0351 | 0.4324 | 0.8416 |
| hard_margin | 0.5682 | 0.5077 | 0.5873 | 0.1111 | 0.4865 | 0.8458 |
| z_margin | 0.6364 | 0.5876 | 0.5729 | 0.5158 | 0.4118 | 0.8500 |
| pair_z | 0.5065 | 0.4610 | 0.4845 | 0.2400 | 0.2500 | 0.8696 |
| hybrid_z_margin | 0.5974 | 0.5151 | 0.6111 | 0.2462 | 0.3529 | 0.8500 |
| own_rank | 0.6006 | 0.5163 | 0.3492 | 0.5897 | 0.2857 | 0.8406 |
| own_minus_hgc_for_lgc | 0.5519 | 0.4533 | 0.5821 | 0.0174 | 0.3636 | 0.8500 |

## Main Test Results

| Vector source | Classifier | Acc | Macro-F1 | AUROC | HGC F1 | LGC F1 | NTL F1 | NST F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| mild-aug mean refined vectors | regular CBM, ntl_boost2.5 | 0.5979 | 0.6289 | 0.8272 | 0.5333 | 0.4727 | 0.5652 | 0.9444 |
| raw_original_top10 | top10 majority_vote | 0.5291 | 0.4334 | 0.7221 | 0.6105 | 0.0000 | 0.3137 | 0.8095 |
| raw_original_top10 | top10 class_average | 0.4974 | 0.3870 | 0.7103 | 0.5744 | 0.0000 | 0.1304 | 0.8434 |
| repaired z_margin bank | top10 majority_vote | 0.5661 | 0.5588 | 0.7931 | 0.5135 | 0.5234 | 0.2917 | 0.9067 |
| repaired z_margin bank | top10 class_average | 0.4921 | 0.4035 | 0.7376 | 0.5596 | 0.0882 | 0.1053 | 0.8608 |

## Improvement Over Raw Top10 Majority

| Metric | Raw original_top10 | Repaired z_margin | Change |
|---|---:|---:|---:|
| Accuracy | 0.5291 | 0.5661 | +0.0370 |
| Macro-F1 | 0.4334 | 0.5588 | +0.1254 |
| AUROC | 0.7221 | 0.7931 | +0.0710 |
| HGC F1 | 0.6105 | 0.5135 | -0.0970 |
| LGC F1 | 0.0000 | 0.5234 | +0.5234 |
| NTL F1 | 0.3137 | 0.2917 | -0.0221 |
| NST F1 | 0.8095 | 0.9067 | +0.0971 |

## Confusion Matrices

Raw original_top10, top10 majority:

| true\pred | HGC | LGC | NTL | NST |
|---|---:|---:|---:|---:|
| HGC | 58 | 0 | 11 | 5 |
| LGC | 45 | 0 | 4 | 4 |
| NTL | 13 | 0 | 8 | 4 |
| NST | 0 | 0 | 3 | 34 |

Repaired z_margin bank, top10 majority:

| true\pred | HGC | LGC | NTL | NST |
|---|---:|---:|---:|---:|
| HGC | 38 | 24 | 11 | 1 |
| LGC | 23 | 28 | 2 | 0 |
| NTL | 13 | 2 | 7 | 3 |
| NST | 0 | 0 | 3 | 34 |

Readout:

- LGC is no longer a zero-recall class: `28/53` LGC test images are predicted
  as LGC.
- HGC/LGC confusion is now bidirectional. This is an improvement over complete
  LGC collapse, but HGC loses specificity.
- NTL remains the weakest class. Repairing HGC/LGC geometry did not solve NTL.
- NST remains stable.

## Top10 Concept Class Counts

Mean concept-class counts among the global top10 concepts on the test split:

| Bank | True class | HGC concepts | LGC concepts | NTL concepts | NST concepts |
|---|---|---:|---:|---:|---:|
| raw_original_top10 | HGC | 5.85 | 1.68 | 2.07 | 0.41 |
| raw_original_top10 | LGC | 7.23 | 0.92 | 1.08 | 0.77 |
| raw_original_top10 | NTL | 3.88 | 0.92 | 3.68 | 1.52 |
| raw_original_top10 | NST | 0.08 | 0.08 | 0.76 | 9.08 |
| repaired z_margin | HGC | 4.35 | 3.78 | 1.59 | 0.27 |
| repaired z_margin | LGC | 4.30 | 5.15 | 0.34 | 0.21 |
| repaired z_margin | NTL | 3.72 | 2.16 | 2.72 | 1.40 |
| repaired z_margin | NST | 0.14 | 0.00 | 0.59 | 9.27 |

This confirms that the concept vector itself improved for LGC:

```text
true-LGC LGC concepts in top10: 0.92 -> 5.15
```

The remaining issue is now clearer: HGC and LGC are too close, and NTL still
does not dominate its own top10 list strongly enough.

## Selected z_margin Concepts

The repaired bank is saved at:

```text
ebtc_confusion_aware_bank_repair_outputs/banks/z_margin/
```

Key selected concepts:

| Class | Concepts |
|---|---|
| HGC | solid appearing fused fronds; rich crimson and tan mosaic; purplish dark red nodules; congested red capillary fronds; solid foundation with papillary fronds; fused papillary fronds; deep red branching channels; papillary fronds with crowding; papillary fronds on solid foundation; heavily granulated red surface |
| LGC | array of evenly spaced papillary fronds; fronds displaying coral hue; soft coral-pink frond cluster; granular appearance of papillary fronds; papillary fronds with frilled margin; papillary fronds with coral hue; papillary fronds with granular appearance; papillary fronds with fringed appearance; orderly branching papillary fronds; papillary fronds with velvety appearance |
| NTL | coppery red mucosal plaque; red plaque with diffuse rim; superficial well-demarcated red surface defect; soft-edged red plaque; plush-looking red plaque; oval red mucosal plaque; broad red plaque with ridges; broad red mucosal thickening; flat brick-red plaque; broad roughened red plaque |
| NST | clean smooth background surface; gently bending fold lines; smooth continuous wall plane; smooth delicate surface sheen; structured line layout; quiet uniform mucosal vista; gentle fold symmetry throughout; balanced fold shape; smooth uniform wall background; neutral fold undulation |

## Interpretation

This is the first vector-level repair that meaningfully improves the top10
majority confusion matrix.

Main success:

- LGC no longer collapses.
- top10 Macro-F1 improves substantially.
- AUROC improves.
- NST stays strong.

Current limitations:

- HGC F1 drops because HGC/LGC separation is now less conservative.
- NTL remains weak and needs a separate NTL-focused bank repair.
- `class_average` is still not reliable. The useful decision rule remains
  top10 majority vote.

## Recommended Next Step

Use the repaired `z_margin` bank as the new concept-vector candidate for the
next stage, then test:

1. regular CBM training on repaired z_margin bank;
2. NTL-specific repair, using explicit NTL-vs-HGC/LGC margins;
3. conservative residual calibration on top of the repaired bank, not on the
   old original_top10 bank.

The immediate next script should reuse:

```text
ebtc_confusion_aware_bank_repair_outputs/banks/z_margin/whitelist_top10.npz
```

as the 40-concept bank for downstream CBM/class-weight/mild-augmentation
evaluation.
