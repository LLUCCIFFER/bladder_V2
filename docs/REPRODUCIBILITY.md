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

Best validation-selected multi-run ensemble found so far:

```bash
python ebtc_ensemble_checkpoints.py \
  --run-dir ebtc_discriminative_whitelist_cycl_search2_h128_align015 \
  --run-dir ebtc_discriminative_whitelist_cycl_search3_h128_align012 \
  --run-dir ebtc_discriminative_whitelist_cycl_improved_fast_align_only_top10 \
  --top-k 10 \
  --output-dir ebtc_discriminative_whitelist_cycl_improvement_analysis/best_val_selected_ensemble_3dirs
```

This averages probabilities from 9 checkpoints and produced:

```text
Val  accuracy/macro-F1/AUROC = 0.8604 / 0.8624 / 0.9648
Test accuracy/macro-F1/AUROC = 0.5450 / 0.5665 / 0.7957
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

## 9. Run Image/Text Embedding Refinement Stage

This stage keeps `filtered_top300` fixed, trains lightweight residual MLP
adapters on cached BioMedCLIP image/text embeddings with multi-positive
InfoNCE, recomputes refined image/text similarities, rebuilds top-k whitelists,
and reruns the same concept-bottleneck classifier.

Conservative setting used for the first comparison:

```bash
python ebtc_embedding_refinement_stage.py \
  --adapter-hidden-dim 128 \
  --refine-lr 1e-4 \
  --lambda-t2i 0.25 \
  --refine-epochs 8 \
  --refine-patience 3 \
  --run-cbm \
  --cbm-seeds 42,43,44 \
  --cbm-epochs 40 \
  --cbm-patience 8 \
  --cbm-hidden-dim 128 \
  --cbm-lambda-cycl 0 \
  --cbm-lambda-align 0.05 \
  --output-dir ebtc_embedding_refinement_stage_conservative_outputs \
  --device cuda
```

Evaluate the 3-checkpoint ensemble for the best refinement-stage variant:

```bash
python ebtc_embedding_refinement_ensemble.py \
  --stage-output-dir ebtc_embedding_refinement_stage_conservative_outputs \
  --stage refined_vectors_original_whitelist_top10
```

Evaluate the 5-checkpoint ensemble after adding seeds 45 and 46:

```bash
python ebtc_embedding_refinement_ensemble.py \
  --stage-output-dir ebtc_embedding_refinement_stage_conservative_outputs \
  --stage-output-dir ebtc_embedding_refinement_stage_conservative_moreseeds_outputs \
  --stage refined_vectors_original_whitelist_top10 \
  --output-dir ebtc_embedding_refinement_stage_conservative_outputs/cbm_training/refined_vectors_original_whitelist_top10/ensemble_5seed
```

Semantic soft-target refinement can be enabled with:

```bash
python ebtc_embedding_refinement_stage.py \
  --refine-target-mode semantic \
  --semantic-related-weight 0.35 \
  --semantic-other-weight 0.02 \
  --adapter-hidden-dim 128 \
  --refine-lr 1e-4 \
  --lambda-t2i 0.25
```

Key files:

- `adapter_refinement/refinement_train_log.csv`
- `adapter_refinement/refined_image_embeddings_{train,val,test}.npy`
- `adapter_refinement/refined_filtered_top300_text_embeddings.npz`
- `matrix_verification/matrix_summary.csv`
- `retrieval_majority_vote/majority_vote_results.csv`
- `refined_filtering/whitelist_top10.csv`
- `refined_vectors_original_whitelist/whitelist_top10.npz`
- `cbm_training/combined_results_summary.csv`
- `cbm_training/*/ensemble/ensemble_metrics.csv`

## Notes

- Test split should remain frozen for final evaluation.
- Use validation metrics for checkpoint selection.
- Do not regenerate concept filtering while running a fixed-bank formal comparison.
