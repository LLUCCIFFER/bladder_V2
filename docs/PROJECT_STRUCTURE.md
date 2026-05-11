# Project Structure

The repository keeps historical experiment scripts at the top level for
backward compatibility. New project-level configuration is centralized in
`ebtc_project_paths.py`.

## Core Scripts

- `ebtc_concept_experiment.py`: concept cleaning, variance scoring, binary probing, and multiclass probing.
- `ebtc_cycl_retrieval_stage.py`: original retrieval compression and CBM/CyCL workflow.
- `ebtc_cycl_retrieval_stage_revised.py`: revised retrieval compression, diagnostics, and CBM/CyCL workflow.
- `ebtc_revised_filtering_only.py`: ranking-only repair stage with 4x4 heatmaps.
- `ebtc_cbm_cycl_v2.py`: V2 CLI for matrix verification, preparation, augmentation, pair debugging, smoke tests, training diagnostics, formal comparisons, and final analysis.
- `ebtc_cbm_cycl_v2_lib.py`: V2 reusable components.
- `ebtc_cycl_hierarchical_stage.py`: hierarchical classification experiments.

## Utility Scripts

- `export_biomedclip_image_embeddings.py`: combines cached BioMedCLIP image embeddings with image names and labels into CSV/pickle.
- `load_biomedclip_cpu.py`: minimal CPU loading check for the local BioMedCLIP checkpoint.
- `export_officialsplit_final_banks.py`: exports selected concept banks from official-split probing outputs.

## Artifact Policy

The following are generated locally and intentionally excluded from git:

- `*_outputs/`
- `*_cache/`
- `*_exports/`
- `*.npy`, `*.npz`, `*.pkl`
- model checkpoints
- generated plots and logs

If a result table is needed for a paper or review, summarize it in markdown
rather than committing the entire output directory.

