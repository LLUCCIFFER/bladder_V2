# Concept Vector Repair Solution

## Problem Diagnosis

The latest predicted-vector experiment shows that the current bottleneck is not
only a noisy downstream classifier input. The 40-d concept vector itself is not
class-separable enough under the fixed `original_top10` bank.

Fixed bank:

```text
HGC positions 0-9
LGC positions 10-19
NTL positions 20-29
NST positions 30-39
```

Observed on the test split:

| Vector source | Rule | Acc | Macro-F1 | HGC F1 | LGC F1 | NTL F1 | NST F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| raw cosine activation | top10 majority | 0.5291 | 0.4334 | 0.6105 | 0.0000 | 0.3137 | 0.8095 |
| predicted clean vector | top10 majority, 3-seed mean | 0.4233 | 0.4115 | 0.2308 | 0.4384 | 0.0625 | 0.9143 |

The MLP-cleaned vector fixes one symptom: LGC is no longer always predicted as
HGC. However, it overcorrects:

- HGC collapses toward LGC.
- NTL collapses toward HGC/LGC.
- NST remains easy and stable.

## Key Evidence

### 1. Raw top10 is HGC-dominated for LGC

Mean top10 concept-class counts on true-LGC test images:

| Vector | HGC concepts | LGC concepts | NTL concepts | NST concepts |
|---|---:|---:|---:|---:|
| raw cosine activation | 7.23 | 0.92 | 1.08 | 0.77 |
| predicted clean vector | 3.77 | 6.08 | 0.15 | 0.00 |

The predicted vector successfully increases LGC concepts for LGC images, but
it is not class-specific: true-HGC images also become LGC-heavy.

### 2. Full-block class scoring still does not recover LGC

As a diagnostic, each class block was scored using all 10 concepts rather than
the global top10. This checks whether the LGC block contains enough signal if
we stop letting HGC concepts dominate top10 selection.

Test results:

| Rule | Acc | Macro-F1 | HGC F1 | LGC F1 | NTL F1 | NST F1 |
|---|---:|---:|---:|---:|---:|---:|
| block_top5_mean | 0.5503 | 0.4501 | 0.6289 | 0.0000 | 0.3478 | 0.8235 |
| block_top3_mean | 0.5450 | 0.4464 | 0.6218 | 0.0000 | 0.3404 | 0.8235 |
| block_median | 0.5344 | 0.4442 | 0.6196 | 0.0000 | 0.3571 | 0.8000 |
| block_mean_all10 | 0.5397 | 0.4322 | 0.6349 | 0.0000 | 0.2800 | 0.8140 |

This is the most important diagnostic: even when all LGC concepts are allowed
to contribute as a block, LGC is still never predicted. Therefore, the current
LGC concept block is not separable from HGC in the embedding space.

### 3. Simple vector calibration is insufficient

Train-set per-concept z-score, robust scaling, min-max scaling, and owner-margin
normalization were tested. None improved the main top10 objective over the raw
cosine baseline. These transforms change the bias but do not create reliable
HGC/LGC/NTL separation.

### 4. Residual repair is still unstable with the fixed bank

A bounded residual repair prototype was tested:

```text
corrected_vector = raw_vector + scale * tanh(MLP(raw_vector))
```

Loss terms included block-level CE, ranking, true-block preservation, and
residual regularization. It improved LGC in some settings but still collapsed
NTL and did not beat the raw top10 majority Macro-F1.

Conclusion: fixed-bank post-processing alone is not enough.

## Root Cause

The current `original_top10` bank is class-balanced by count, but not
confusion-balanced by geometry.

The main failure is not simply that false concepts are noisy. It is that:

- HGC concepts score higher than LGC concepts for true-LGC images.
- LGC concepts are not strong enough even under all-block scoring.
- NTL concepts are unstable and easily replaced by HGC concepts after learned
  vector repair.
- NST concepts are already clean and should be preserved.

So the vector repair problem should be reframed as:

```text
Build a confusion-aware concept bank first,
then apply conservative vector calibration.
```

## Recommended Solution

### Stage A: Confusion-Aware Bank Repair

Do not replace the whole pipeline. Keep the same 40-d format, but repair the
10 concepts per class using train-only discriminative criteria.

For every candidate concept, compute on train:

```text
own_mean[class]
negative_mean[each other class]
hardest_negative_class
hardest_negative_margin = own_mean - max(other means)
pairwise margins:
  HGC_vs_LGC
  LGC_vs_HGC
  NTL_vs_HGC
  NTL_vs_LGC
  NST_vs_all
```

Selection should optimize per-class confusion, not only own-class rank:

- HGC concepts must beat LGC and NTL negatives.
- LGC concepts must specifically beat HGC negatives.
- NTL concepts must beat HGC/LGC negatives.
- NST should retain its current stable concepts unless a candidate is clearly
  better.

Acceptance criteria on validation before touching test:

| Metric | Target |
|---|---:|
| true-LGC top10 LGC concept count | >= 3.0 |
| true-HGC top10 HGC concept count | >= 4.0 |
| true-NTL top10 NTL concept count | >= 2.5 |
| true-NST top10 NST concept count | >= 8.0 |
| top10 majority LGC F1 | > 0.25 |
| top10 majority Macro-F1 | > raw baseline |

### Stage B: Conservative Vector Calibration

After the bank is repaired, apply calibration that preserves raw concept
ordering instead of generating a vector from scratch.

Recommended form:

```text
calibrated_score_j =
  raw_cosine_j
  + alpha_class[class(j)]
  + beta_j * normalized_margin_j
```

or a bounded residual:

```text
corrected_vector = raw_vector + 0.1 * tanh(residual_net(raw_vector))
```

Training objective:

```text
L = L_block_CE
  + lambda_rank * L_pairwise_rank
  + lambda_preserve * ||corrected_true_block - raw_true_block||^2
  + lambda_residual * ||corrected - raw||^2
```

Important constraints:

- Do not use hard `-1` MSE as the primary objective.
- Select checkpoints by validation top10 Macro-F1 and LGC F1, not vector MSE.
- Keep residual small so HGC/NTL do not collapse while fixing LGC.

### Stage C: Report Both Outputs

For every future vector-repair run, report:

| Vector source | Classifier | Acc | Macro-F1 | HGC F1 | LGC F1 | NTL F1 | NST F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| regular refined vectors | regular CBM | ... | ... | ... | ... | ... | ... |
| raw concept vector | top10 majority | ... | ... | ... | ... | ... | ... |
| repaired concept vector | top10 majority | ... | ... | ... | ... | ... | ... |
| repaired concept vector | top10 class-average | ... | ... | ... | ... | ... | ... |

Also report:

- confusion matrix,
- per-class F1,
- top10 concept class counts by true class,
- vector-quality diagnostics.

## Next Experiment

The next implementation should be:

```text
ebtc_confusion_aware_bank_repair.py
```

Inputs:

- current filtered top300 bank,
- current refined image embeddings,
- current original_top10 bank as baseline.

Outputs:

- repaired 40-concept bank with the same class-block format,
- top10 majority/class-average metrics,
- class-count matrices,
- comparison against raw original_top10 and regular CBM.

This is the most direct path to fixing the concept vector. Post-hoc MLP
prediction should be treated as secondary until the bank itself passes the
validation-level block separability checks.
