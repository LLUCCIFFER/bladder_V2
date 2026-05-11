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

## Notes

- Test split should remain frozen for final evaluation.
- Use validation metrics for checkpoint selection.
- Do not regenerate concept filtering while running a fixed-bank formal comparison.

