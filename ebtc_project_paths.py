"""Centralized path configuration for the EBTC CBM/CyCL project.

The original experiment scripts were developed on a fixed workstation with
absolute paths. For an open-source research repository, paths must be
overridable without editing code. This module keeps the historical defaults
for backward compatibility while allowing users to set environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path


def env_path(name: str, default: Path | str) -> Path:
    """Return a filesystem path from an environment variable or a default."""
    return Path(os.environ.get(name, str(default))).expanduser()


PROJECT_ROOT = env_path("EBTC_PROJECT_ROOT", Path(__file__).resolve().parent)
OUTPUT_ROOT = env_path("EBTC_OUTPUT_ROOT", PROJECT_ROOT)
EXPERIMENT_DIR = env_path("EBTC_EXPERIMENT_DIR", PROJECT_ROOT.parent / "experiment")

DATASET_ROOT = env_path("EBTC_DATA_ROOT", "/dpc/kunf0084/bladder/data/EBTC")
MODEL_DIR = env_path("BIOMEDCLIP_MODEL_DIR", "/dpc/kunf0084/bladder/model")
CONCEPT_DIR = env_path("EBTC_CONCEPT_DIR", EXPERIMENT_DIR / "baseconcept")
BIOMEDBERT_TEXT_CONFIG_DIR = env_path(
    "EBTC_BIOMEDBERT_TEXT_CONFIG_DIR",
    PROJECT_ROOT / "biomedbert_text_config",
)

OFFICIAL_EMBEDDINGS_DIR = env_path(
    "EBTC_EMBEDDINGS_DIR",
    OUTPUT_ROOT / "ebtc_official_split_embedding_cache" / "embeddings",
)
BACKUP_EMBEDDINGS_DIR = env_path(
    "EBTC_BACKUP_EMBEDDINGS_DIR",
    OUTPUT_ROOT / "ebtc_concept_experiment_outputs_backup_groupaware_20260414" / "embeddings",
)

CONCEPT_EXPERIMENT_OUTPUT_DIR = env_path(
    "EBTC_CONCEPT_EXPERIMENT_OUTPUT_DIR",
    OUTPUT_ROOT / "ebtc_concept_experiment_outputs",
)
CYCL_STAGE_OUTPUT_DIR = env_path(
    "EBTC_CYCL_STAGE_OUTPUT_DIR",
    OUTPUT_ROOT / "ebtc_cycl_retrieval_stage_multiseed_outputs",
)
CYCL_REVISED_OUTPUT_DIR = env_path(
    "EBTC_CYCL_REVISED_OUTPUT_DIR",
    OUTPUT_ROOT / "ebtc_cycl_retrieval_stage_revised_outputs",
)
FILTERING_ONLY_OUTPUT_DIR = env_path(
    "EBTC_FILTERING_ONLY_OUTPUT_DIR",
    OUTPUT_ROOT / "ebtc_revised_filtering_only_outputs",
)
FORMAL_COMPARE_OUTPUT_DIR = env_path(
    "EBTC_FORMAL_COMPARE_OUTPUT_DIR",
    OUTPUT_ROOT / "ebtc_revised_bank_formal_compare_outputs",
)
HIERARCHICAL_OUTPUT_DIR = env_path(
    "EBTC_HIERARCHICAL_OUTPUT_DIR",
    OUTPUT_ROOT / "ebtc_cycl_hierarchical_stage_outputs",
)
CBM_CYCL_V2_OUTPUT_DIR = env_path(
    "EBTC_CBM_CYCL_V2_OUTPUT_DIR",
    OUTPUT_ROOT / "ebtc_cbm_cycl_v2_outputs" / "outputs_v2",
)
IMAGE_EMBEDDING_EXPORT_DIR = env_path(
    "EBTC_IMAGE_EMBEDDING_EXPORT_DIR",
    OUTPUT_ROOT / "ebtc_image_embedding_exports",
)

FILTERED_TOP300_BANK_DIR = env_path(
    "EBTC_FILTERED_TOP300_BANK_DIR",
    CYCL_REVISED_OUTPUT_DIR / "banks" / "filtered_top300",
)
RETRIEVAL_TOP10_BANK_DIR = env_path(
    "EBTC_RETRIEVAL_TOP10_BANK_DIR",
    CYCL_REVISED_OUTPUT_DIR / "banks" / "retrieval_top10_per_class",
)
RETRIEVAL_TOP10_DISC_BANK_DIR = env_path(
    "EBTC_RETRIEVAL_TOP10_DISC_BANK_DIR",
    CYCL_REVISED_OUTPUT_DIR / "banks" / "retrieval_top10_per_class_disc",
)
WEIGHTED_SYM10_BANK_DIR = env_path(
    "EBTC_WEIGHTED_SYM10_BANK_DIR",
    FILTERING_ONLY_OUTPUT_DIR / "banks" / "bank_weighted_sym10",
)

