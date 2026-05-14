# EBTC Concept Bottleneck / CyCL

This repository contains research code for EBTC concept-bank construction,
BioMedCLIP image/text embeddings, concept-bottleneck models (CBM), and CyCL V2
experiments.

The repository is intentionally code-first. Large local artifacts are not
tracked in git.

## What Is Included

- Concept cleaning, variance filtering, binary probing, and multiclass probing.
- Retrieval-based concept-bank compression and filtering diagnostics.
- 4x4 image-text class-similarity verification.
- CBM-no-CyCL and CyCL training pipelines.
- V2 pair-definition, image-concept pairing, augmentation, diagnostics, and formal comparison scripts.
- Utilities for exporting BioMedCLIP image embeddings with metadata.

## What Is Not Included

- EBTC image data.
- BioMedCLIP model weights.
- Cached `.npy`, `.npz`, `.pkl` embeddings.
- Training checkpoints.
- Generated plots, logs, and experiment output directories.

These files are excluded by `.gitignore` because they are large, local, or
dataset/model artifacts.

## Repository Layout

```text
.
├── ebtc_project_paths.py                  # central path/env configuration
├── ebtc_concept_experiment.py             # concept filtering + probing
├── ebtc_cycl_retrieval_stage.py           # original retrieval-compression stage
├── ebtc_cycl_retrieval_stage_revised.py   # revised retrieval/CBM/CyCL stage
├── ebtc_revised_filtering_only.py         # ranking-only filtering diagnostics
├── ebtc_revised_bank_formal_compare.py    # formal bank comparison
├── ebtc_cbm_cycl_v2.py                    # CBM/CyCL V2 CLI
├── ebtc_cbm_cycl_v2_lib.py                # V2 model/data/pair/loss utilities
├── ebtc_cycl_hierarchical_stage.py        # hierarchical experiments
├── ebtc_discriminative_whitelist_cycl.py  # hardest-negative whitelist + M-profile CyCL
├── ebtc_embedding_refinement_stage.py     # image/text adapter refinement + refined whitelist/CBM
├── ebtc_embedding_refinement_ensemble.py  # ensemble evaluation for refinement-stage checkpoints
├── export_biomedclip_image_embeddings.py  # embedding export utility
├── configs/                               # example environment/config files
├── docs/                                  # reproducibility and structure docs
└── scripts/                               # lightweight repo utilities
```

See [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) for more detail.

## Environment

The original experiments used a conda environment named `torch`.

Install Python dependencies:

```bash
pip install -r requirements.txt
```

For development checks:

```bash
pip install -r requirements-dev.txt
```

## Path Configuration

All major local paths are configurable with environment variables. Start from:

```bash
cp configs/paths.example.env .env
```

Then edit `.env` or export variables manually:

```bash
export EBTC_DATA_ROOT=/path/to/EBTC
export BIOMEDCLIP_MODEL_DIR=/path/to/biomedclip/model
export EBTC_CONCEPT_DIR=/path/to/baseconcept
export EBTC_OUTPUT_ROOT=/path/to/output/root
```

The scripts keep historical defaults for backward compatibility, but new users
should set these variables explicitly.

## Quick Checks

Check that the repository-level Python files compile:

```bash
make check
```

Export cached BioMedCLIP image embeddings to CSV/pickle:

```bash
python export_biomedclip_image_embeddings.py
```

Run V2 4x4 image-text matrix verification:

```bash
python ebtc_cbm_cycl_v2.py matrix
```

Run V2 smoke test:

```bash
python ebtc_cbm_cycl_v2.py smoke-test --model-type cbm_no_cycl_v2
python ebtc_cbm_cycl_v2.py smoke-test --model-type cycl_v2
```

Run discriminative whitelist + concept-profile CyCL:

```bash
python ebtc_discriminative_whitelist_cycl.py \
  --top-ks 5,10,20,30 \
  --seeds 42,43,44
```

Current recommended lightweight setting from the follow-up ablation:

```bash
python ebtc_discriminative_whitelist_cycl.py \
  --top-ks 10 \
  --seeds 42,43,44 \
  --lambda-cycl 0 \
  --lambda-align 0.05 \
  --num-threads 2
```

Evaluate a seed ensemble from saved checkpoints:

```bash
python ebtc_ensemble_checkpoints.py \
  --run-dir ebtc_discriminative_whitelist_cycl_search4_h128_align012_5seeds \
  --top-k 10
```

The ensemble utility also supports multiple run directories. The current best
validation-selected exploratory ensemble uses 9 checkpoints from three runs:

```bash
python ebtc_ensemble_checkpoints.py \
  --run-dir ebtc_discriminative_whitelist_cycl_search2_h128_align015 \
  --run-dir ebtc_discriminative_whitelist_cycl_search3_h128_align012 \
  --run-dir ebtc_discriminative_whitelist_cycl_improved_fast_align_only_top10 \
  --top-k 10 \
  --output-dir ebtc_discriminative_whitelist_cycl_improvement_analysis/best_val_selected_ensemble_3dirs
```

This configuration uses 40 concepts, an adapter-CBM with concept-only
classification, and probability averaging across checkpoints.

Run the next-stage image/text embedding refinement experiment:

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

Evaluate 3-checkpoint ensembles for a refinement stage:

```bash
python ebtc_embedding_refinement_ensemble.py \
  --stage-output-dir ebtc_embedding_refinement_stage_conservative_outputs \
  --stage refined_vectors_original_whitelist_top10
```

## Reproducibility

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the expected data
layout, environment variables, and recommended command sequence.

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
