# EBTC CBM/CyCL Project Code

This repository contains the code used for EBTC concept filtering, BioMedCLIP
embedding export, CBM/CyCL training, revised V2 training, and related analysis.

Large generated artifacts are intentionally not tracked in git:

- EBTC image data
- BioMedCLIP model weights
- cached image/text embeddings
- `.npy`, `.npz`, `.pkl` caches
- experiment output directories
- generated plots, logs, and result bundles

Local default paths used by the scripts:

- EBTC dataset: `/dpc/kunf0084/bladder/data/EBTC`
- BioMedCLIP model: `/dpc/kunf0084/bladder/model`
- project working directory: `/home/kunet.ae/100069491/newcode`

Main scripts:

- `ebtc_concept_experiment.py`: initial concept cleaning, variance filtering, binary/multiclass probing.
- `ebtc_cycl_retrieval_stage.py`: initial retrieval-compression and CBM/CyCL stage.
- `ebtc_cycl_retrieval_stage_revised.py`: revised retrieval compression and CBM/CyCL experiments.
- `ebtc_revised_filtering_only.py`: ranking-only and 4x4 heatmap filtering diagnostics.
- `ebtc_cbm_cycl_v2.py`: V2 CBM/CyCL training and diagnostics.
- `ebtc_cbm_cycl_v2_lib.py`: V2 model, dataset, augmentation, pair definition, and loss utilities.
- `ebtc_revised_bank_formal_compare.py`: formal comparison for selected revised banks.
- `ebtc_cycl_hierarchical_stage.py`: hierarchical classification experiments.
- `export_biomedclip_image_embeddings.py`: export cached BioMedCLIP image embeddings with metadata.

