# Reproducibility Guide

## 1. Prepare Environment

```bash
conda activate torch
pip install -r requirements.txt
```

## 2. Configure Paths

```bash
cp configs/paths.example.env .env
```

Set the required values:

```bash
export EBTC_DATA_ROOT=/path/to/EBTC
export BIOMEDCLIP_MODEL_DIR=/path/to/biomedclip/model
export EBTC_CONCEPT_DIR=/path/to/baseconcept
export EBTC_OUTPUT_ROOT=/path/to/output/root
```

Optional variables are listed in `configs/paths.example.env`.

## 3. Verify BioMedCLIP Loading

```bash
python load_biomedclip_cpu.py
```

## 4. Export Image Embeddings

If the official split embedding cache already exists:

```bash
python export_biomedclip_image_embeddings.py
```

This writes CSV/pickle files with image names, labels, and 512-dimensional
BioMedCLIP image embeddings.

## 5. Run Matrix Verification

```bash
python ebtc_cbm_cycl_v2.py matrix
```

This computes the 4x4 image-text class-similarity matrices for the configured
concept bank.

## 6. Run V2 Smoke Tests

```bash
python ebtc_cbm_cycl_v2.py smoke-test --model-type cbm_no_cycl_v2
python ebtc_cbm_cycl_v2.py smoke-test --model-type cycl_v2
```

## 7. Run Formal V2 Comparison

Use this only after the caches and fixed banks exist:

```bash
python ebtc_cbm_cycl_v2.py formal-it-compare \
  --seeds 42,43,44 \
  --warmup-epochs 5 \
  --joint-epochs 10 \
  --n-views 4
```

## 8. Run Discriminative Whitelist + Concept-Profile CyCL

This stage starts from the fixed `filtered_top300` bank and computes:

- concept-wise own-class mean similarity,
- hardest-negative class and margin,
- top-k whitelists for `k=5,10,20,30`,
- non-one-hot concept-class association matrix `M`,
- class-class concept profile cosine similarity,
- a minimal adapter + concept-similarity + linear-classifier model trained with:
  `L = L_cls + lambda_cycl * L_CyCL + lambda_align * L_align`.

```bash
python ebtc_discriminative_whitelist_cycl.py \
  --top-ks 5,10,20,30 \
  --seeds 42,43,44 \
  --epochs 80 \
  --patience 15
```

Current recommended lightweight setting:

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

Best exploratory ensemble setting:

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

python ebtc_ensemble_checkpoints.py \
  --run-dir ebtc_discriminative_whitelist_cycl_search4_h128_align012_5seeds \
  --top-k 10
```

Default output:

```text
ebtc_discriminative_whitelist_cycl_outputs/
```

Key files:

- `concept_discriminative_scores.csv`
- `top_concepts_k*_*.csv`
- `whitelist_top*.csv`
- `M_matrix_top*_normalized.csv`
- `M_matrix_top*_heatmap.png`
- `class_profile_similarity_top*.csv`
- `class_profile_similarity_top*.png`
- `training_seed_results.csv`
- `training_results_summary.csv`
- `model_architecture.png`
- `experiment_report.md`

## Notes

- Test split should remain frozen for final evaluation.
- Use validation metrics for checkpoint selection.
- Do not regenerate concept filtering while running a fixed-bank formal comparison.
