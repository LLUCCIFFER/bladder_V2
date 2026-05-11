#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from ebtc_project_paths import (
    BACKUP_EMBEDDINGS_DIR,
    CYCL_REVISED_OUTPUT_DIR,
    CYCL_STAGE_OUTPUT_DIR,
    EXPERIMENT_DIR,
    OFFICIAL_EMBEDDINGS_DIR,
    PROJECT_ROOT,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))
NEWCODE_DIR = PROJECT_ROOT
if str(NEWCODE_DIR) not in sys.path:
    sys.path.insert(0, str(NEWCODE_DIR))

from etbc_wli_train_cbm_cycl import (  # noqa: E402
    CLASS_NAMES,
    CLASS_TO_INDEX,
    LABEL_CODES,
    build_loaders,
    compute_class_weights,
    ensure_dir,
    evaluate_main,
    group_records_by_split,
    load_biomedclip,
    save_case_examples_figure,
    save_confusion_figure,
    save_prediction_records,
    save_soft_concept_artifacts,
    select_case_examples,
    set_seed,
    train_baseline,
    train_main_model,
    write_csv,
    write_json,
)
from ebtc_concept_experiment import build_manifest, create_splits  # noqa: E402


DEFAULT_FILTERED_TOP300_DIR = Path(
    CYCL_STAGE_OUTPUT_DIR / "banks" / "filtered_top300"
)
DEFAULT_EMBEDDINGS_DIR = OFFICIAL_EMBEDDINGS_DIR
DEFAULT_BACKUP_EMBEDDINGS_DIR = BACKUP_EMBEDDINGS_DIR
DEFAULT_OUTPUT_DIR = CYCL_REVISED_OUTPUT_DIR
EPS = 1e-6
TOP_N_CHOICES = (30, 20, 10)


@dataclass
class ConceptCandidate:
    class_name: str
    concept: str
    source_rank_in_filtered_top300: int
    concept_id: str
    embedding_index: int
    embedding: np.ndarray


@dataclass
class RetrievalBank:
    name: str
    per_class_rows: dict[str, list[dict[str, Any]]]
    merged_rows: list[dict[str, Any]]
    merged_concepts: list[str]
    merged_embeddings: np.ndarray
    total_before_dedup: int
    total_after_dedup: int
    dedup_removed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Revised retrieval compression and CBM/CyCL experiments on official-split EBTC filtered_top300."
    )
    parser.add_argument("--filtered-top300-dir", type=Path, default=DEFAULT_FILTERED_TOP300_DIR)
    parser.add_argument("--embeddings-dir", type=Path, default=DEFAULT_EMBEDDINGS_DIR)
    parser.add_argument("--backup-embeddings-dir", type=Path, default=DEFAULT_BACKUP_EMBEDDINGS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", default="42,43")
    parser.add_argument(
        "--ranking-method",
        choices=["mean_cosine", "auroc", "disc_mean_margin", "disc_max_margin"],
        default="disc_mean_margin",
    )
    parser.add_argument("--disc-lambda", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--schedule", choices=["short_debug", "improved_main"], default="improved_main")
    parser.add_argument("--baseline-epochs", type=int, default=None)
    parser.add_argument("--warmup-epochs", type=int, default=None)
    parser.add_argument("--joint-epochs", type=int, default=None)
    parser.add_argument("--baseline-lr", type=float, default=1e-3)
    parser.add_argument("--main-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-concept", type=float, default=1.0)
    parser.add_argument("--lambda-align", type=float, default=0.3)
    parser.add_argument("--lambda-cycl", type=float, default=0.1)
    parser.add_argument("--lambda-img-txt", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=0.07)
    parser.add_argument("--tau-img-txt", type=float, default=0.07)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--proj-dim", type=int, default=128)
    parser.add_argument("--num-views", type=int, default=2)
    parser.add_argument("--use-color-aug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-geom-aug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-image-text-pairing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--run-banks",
        default="filtered_top300,retrieval_top10_per_class_disc,retrieval_top20_per_class_disc",
        help="Comma-separated bank names for revised CBM/CyCL runs.",
    )
    parser.add_argument(
        "--run-models",
        default="cbm_only,cycl",
        help="Comma-separated model variants among cbm_only and cycl.",
    )
    parser.add_argument("--run-view-ablation", action="store_true")
    parser.add_argument("--view-ablation-seeds", default="42")
    parser.add_argument("--skip-retrieval", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-class-similarity", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def parse_seed_list(seeds_arg: str) -> list[int]:
    seeds = [int(item.strip()) for item in seeds_arg.split(",") if item.strip()]
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def l2_normalize(array: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(array, axis=-1, keepdims=True)
    return array / np.clip(denom, EPS, None)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def ensure_official_split_embedding_cache(embeddings_dir: Path, backup_embeddings_dir: Path) -> None:
    required_files = [
        embeddings_dir / "image_embeddings_train.npy",
        embeddings_dir / "image_embeddings_train_manifest.csv",
        embeddings_dir / "image_embeddings_val.npy",
        embeddings_dir / "image_embeddings_val_manifest.csv",
        embeddings_dir / "image_embeddings_test.npy",
        embeddings_dir / "image_embeddings_test_manifest.csv",
        embeddings_dir / "text_embeddings_HGC.npy",
        embeddings_dir / "text_embeddings_HGC_concepts.csv",
        embeddings_dir / "text_embeddings_LGC.npy",
        embeddings_dir / "text_embeddings_LGC_concepts.csv",
        embeddings_dir / "text_embeddings_NTL.npy",
        embeddings_dir / "text_embeddings_NTL_concepts.csv",
        embeddings_dir / "text_embeddings_NST.npy",
        embeddings_dir / "text_embeddings_NST_concepts.csv",
    ]
    if all(path.exists() for path in required_files):
        return

    ensure_dir(embeddings_dir)
    if not backup_embeddings_dir.exists():
        raise FileNotFoundError(
            f"Backup embeddings directory not found and official cache is missing: {backup_embeddings_dir}"
        )

    backup_rows: list[dict[str, str]] = []
    backup_embeddings_list: list[np.ndarray] = []
    for split in ["train", "val", "test"]:
        manifest_path = backup_embeddings_dir / f"image_embeddings_{split}_manifest.csv"
        embedding_path = backup_embeddings_dir / f"image_embeddings_{split}.npy"
        if not manifest_path.exists() or not embedding_path.exists():
            raise FileNotFoundError(f"Missing backup cache for split={split}: {manifest_path} / {embedding_path}")
        split_rows = read_csv_rows(manifest_path)
        split_embeddings = np.load(embedding_path).astype(np.float32)
        if len(split_rows) != split_embeddings.shape[0]:
            raise RuntimeError(
                f"Backup manifest/embedding mismatch for split={split}: {len(split_rows)} vs {split_embeddings.shape[0]}"
            )
        backup_rows.extend(split_rows)
        backup_embeddings_list.append(split_embeddings)

    backup_embeddings = np.concatenate(backup_embeddings_list, axis=0).astype(np.float32)
    image_path_to_index = {str(row["image_path"]): idx for idx, row in enumerate(backup_rows)}
    if len(image_path_to_index) != len(backup_rows):
        raise RuntimeError("Duplicate image paths found in backup embedding manifests.")

    cache_root = embeddings_dir.parent
    manifest_tmp_dir = cache_root / "_official_split_cache_tmp"
    manifest_df = build_manifest(manifest_tmp_dir)
    split_map = create_splits(manifest_df, manifest_tmp_dir, seed=0)

    for split, df in split_map.items():
        manifest_rows = df.to_dict(orient="records")
        missing_paths = [str(row["image_path"]) for row in manifest_rows if str(row["image_path"]) not in image_path_to_index]
        if missing_paths:
            raise RuntimeError(
                f"Official split cache build failed; missing image embeddings for {split}: {missing_paths[:10]}"
            )
        split_indices = [image_path_to_index[str(row["image_path"])] for row in manifest_rows]
        split_embeddings = backup_embeddings[np.array(split_indices, dtype=np.int64)]
        np.save(embeddings_dir / f"image_embeddings_{split}.npy", split_embeddings.astype(np.float32))
        with (embeddings_dir / f"image_embeddings_{split}_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0].keys()))
            writer.writeheader()
            writer.writerows(manifest_rows)

    for class_name in LABEL_CODES:
        for suffix in [".npy", "_concepts.csv"]:
            source_path = backup_embeddings_dir / f"text_embeddings_{class_name}{suffix}"
            target_path = embeddings_dir / f"text_embeddings_{class_name}{suffix}"
            if not source_path.exists():
                raise FileNotFoundError(f"Missing backup text embedding file: {source_path}")
            if not target_path.exists():
                shutil.copy2(source_path, target_path)


def load_cached_split_payloads(embeddings_dir: Path) -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    for split in ["train", "val", "test"]:
        manifest_rows = read_csv_rows(embeddings_dir / f"image_embeddings_{split}_manifest.csv")
        image_embeddings = np.load(embeddings_dir / f"image_embeddings_{split}.npy").astype(np.float32)
        if len(manifest_rows) != image_embeddings.shape[0]:
            raise RuntimeError(
                f"Mismatch for split={split}: manifest_rows={len(manifest_rows)} embeddings={image_embeddings.shape[0]}"
            )
        payloads[split] = {
            "rows": manifest_rows,
            "embeddings": l2_normalize(image_embeddings),
        }
    return payloads


def load_class_text_embeddings(embeddings_dir: Path, class_name: str) -> tuple[dict[str, int], np.ndarray]:
    concept_rows = read_csv_rows(embeddings_dir / f"text_embeddings_{class_name}_concepts.csv")
    concept_to_index: dict[str, int] = {}
    for row in concept_rows:
        concept = str(row["concept"])
        concept_to_index.setdefault(concept, int(row["concept_index"]))
    embeddings = np.load(embeddings_dir / f"text_embeddings_{class_name}.npy").astype(np.float32)
    embeddings = l2_normalize(embeddings)
    return concept_to_index, embeddings


def load_filtered_top300(filtered_top300_dir: Path, embeddings_dir: Path) -> dict[str, list[ConceptCandidate]]:
    candidates_by_class: dict[str, list[ConceptCandidate]] = {}
    for class_name in LABEL_CODES:
        class_rows = read_csv_rows(filtered_top300_dir / f"{class_name}.csv")
        concept_to_index, class_embeddings = load_class_text_embeddings(embeddings_dir, class_name)
        class_candidates: list[ConceptCandidate] = []
        for row in class_rows:
            concept = str(row["concept"])
            if concept not in concept_to_index:
                raise KeyError(f"Concept not found in cached embeddings: class={class_name} concept={concept}")
            source_rank = int(row.get("concept_rank") or row.get("source_rank_in_filtered_top300") or row.get("retrieval_rank_in_class") or 0)
            embedding_index = concept_to_index[concept]
            class_candidates.append(
                ConceptCandidate(
                    class_name=class_name,
                    concept=concept,
                    source_rank_in_filtered_top300=source_rank,
                    concept_id=f"{class_name}_{source_rank:03d}",
                    embedding_index=embedding_index,
                    embedding=class_embeddings[embedding_index].astype(np.float32),
                )
            )
        if len(class_candidates) != 300:
            raise RuntimeError(f"Expected 300 concepts for {class_name}, got {len(class_candidates)}")
        candidates_by_class[class_name] = class_candidates
    return candidates_by_class


def compute_class_retrieval_scores(
    candidates_by_class: dict[str, list[ConceptCandidate]],
    split_payloads: dict[str, dict[str, Any]],
    ranking_method: str,
    disc_lambda: float,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    train_rows = split_payloads["train"]["rows"]
    train_embeddings = split_payloads["train"]["embeddings"]
    train_labels = np.array([CLASS_TO_INDEX[str(row["class_name"])] for row in train_rows], dtype=np.int64)

    rankings_by_class: dict[str, list[dict[str, Any]]] = {}
    all_rows: list[dict[str, Any]] = []
    for class_name in LABEL_CODES:
        class_index = CLASS_TO_INDEX[class_name]
        class_mask = train_labels == class_index
        if not np.any(class_mask):
            raise RuntimeError(f"No train samples for class {class_name}")
        class_embeddings = train_embeddings[class_mask]
        rest_embeddings = train_embeddings[~class_mask]
        binary_labels = (train_labels == class_index).astype(np.int64)

        class_rank_rows: list[dict[str, Any]] = []
        for candidate in candidates_by_class[class_name]:
            similarities = train_embeddings @ candidate.embedding
            class_means = {}
            other_class_means: list[float] = []
            for other_class_name in LABEL_CODES:
                other_class_index = CLASS_TO_INDEX[other_class_name]
                other_mask = train_labels == other_class_index
                class_mean = float((train_embeddings[other_mask] @ candidate.embedding).mean())
                class_means[other_class_name] = class_mean
                if other_class_name != class_name:
                    other_class_means.append(class_mean)
            mean_class_similarity = float(class_means[class_name])
            mean_rest_similarity = float(np.mean(other_class_means)) if other_class_means else float("nan")
            max_other_similarity = float(np.max(other_class_means)) if other_class_means else float("nan")
            auroc_ovr = float(roc_auc_score(binary_labels, similarities)) if np.unique(binary_labels).size > 1 else float("nan")
            if ranking_method == "mean_cosine":
                ranking_score = mean_class_similarity
            elif ranking_method == "auroc":
                ranking_score = auroc_ovr
            elif ranking_method == "disc_mean_margin":
                ranking_score = mean_class_similarity - disc_lambda * mean_rest_similarity
            elif ranking_method == "disc_max_margin":
                ranking_score = mean_class_similarity - max_other_similarity
            else:
                raise ValueError(f"Unsupported ranking method: {ranking_method}")

            row = {
                "class_name": class_name,
                "concept": candidate.concept,
                "concept_id": candidate.concept_id,
                "source_rank_in_filtered_top300": candidate.source_rank_in_filtered_top300,
                "embedding_index": candidate.embedding_index,
                "ranking_method": ranking_method,
                "ranking_score": ranking_score,
                "mu_same": mean_class_similarity,
                "mu_diff": mean_rest_similarity,
                "mu_diff_max": max_other_similarity,
                "margin_vs_rest": mean_class_similarity - mean_rest_similarity if np.isfinite(mean_rest_similarity) else float("nan"),
                "margin_vs_max_other": mean_class_similarity - max_other_similarity if np.isfinite(max_other_similarity) else float("nan"),
                "auroc_ovr": auroc_ovr,
                "embedding": candidate.embedding,
            }
            for other_class_name in LABEL_CODES:
                row[f"mu_{other_class_name}"] = float(class_means[other_class_name])
            class_rank_rows.append(row)

        class_rank_rows.sort(
            key=lambda item: (
                -float(item["ranking_score"]),
                -float(item["mu_same"]),
                float(item["source_rank_in_filtered_top300"]),
                str(item["concept"]),
            )
        )
        for rank_index, row in enumerate(class_rank_rows, start=1):
            row["retrieval_rank_in_class"] = rank_index
            all_rows.append({key: value for key, value in row.items() if key != "embedding"})
        rankings_by_class[class_name] = class_rank_rows
    return rankings_by_class, all_rows


def merge_selected_rows(bank_name: str, per_class_rows: dict[str, list[dict[str, Any]]]) -> RetrievalBank:
    merged_rows: list[dict[str, Any]] = []
    concept_to_index: dict[str, int] = {}
    before_count = sum(len(rows) for rows in per_class_rows.values())

    for class_name in LABEL_CODES:
        for row in per_class_rows[class_name]:
            concept = str(row["concept"])
            if concept not in concept_to_index:
                merged_index = len(merged_rows)
                merged_row = {
                    "bank_name": bank_name,
                    "merged_index": merged_index,
                    "concept": concept,
                    "primary_class": class_name,
                    "primary_concept_id": row["concept_id"],
                    "primary_source_rank_in_filtered_top300": int(row["source_rank_in_filtered_top300"]),
                    "primary_retrieval_rank_in_class": int(row["retrieval_rank_in_class"]),
                    "primary_ranking_score": float(row["ranking_score"]),
                    "source_classes": [class_name],
                    "source_concept_ids": [str(row["concept_id"])],
                    "source_retrieval_ranks": [int(row["retrieval_rank_in_class"])],
                    "source_filtered_top300_ranks": [int(row["source_rank_in_filtered_top300"])],
                    "embedding": row["embedding"],
                }
                merged_rows.append(merged_row)
                concept_to_index[concept] = merged_index
            else:
                merged_row = merged_rows[concept_to_index[concept]]
                merged_row["source_classes"].append(class_name)
                merged_row["source_concept_ids"].append(str(row["concept_id"]))
                merged_row["source_retrieval_ranks"].append(int(row["retrieval_rank_in_class"]))
                merged_row["source_filtered_top300_ranks"].append(int(row["source_rank_in_filtered_top300"]))

    merged_concepts = [str(row["concept"]) for row in merged_rows]
    merged_embeddings = np.stack([row["embedding"] for row in merged_rows], axis=0).astype(np.float32)
    merged_embeddings = l2_normalize(merged_embeddings)
    for row in merged_rows:
        row["source_class_count"] = len(row["source_classes"])

    return RetrievalBank(
        name=bank_name,
        per_class_rows=per_class_rows,
        merged_rows=merged_rows,
        merged_concepts=merged_concepts,
        merged_embeddings=merged_embeddings,
        total_before_dedup=before_count,
        total_after_dedup=len(merged_rows),
        dedup_removed=before_count - len(merged_rows),
    )


def build_retrieval_banks(
    rankings_by_class: dict[str, list[dict[str, Any]]],
    bank_suffix: str = "",
    include_filtered_top300: bool = True,
) -> dict[str, RetrievalBank]:
    banks: dict[str, RetrievalBank] = {}

    if include_filtered_top300:
        full_bank_rows = {class_name: list(rankings_by_class[class_name]) for class_name in LABEL_CODES}
        banks["filtered_top300"] = merge_selected_rows("filtered_top300", full_bank_rows)

    for top_n in TOP_N_CHOICES:
        bank_name = f"retrieval_top{top_n}_per_class{bank_suffix}"
        selected_rows = {
            class_name: [dict(item) for item in rankings_by_class[class_name][:top_n]]
            for class_name in LABEL_CODES
        }
        banks[bank_name] = merge_selected_rows(bank_name, selected_rows)
    return banks


def save_retrieval_bank(bank: RetrievalBank, bank_dir: Path) -> None:
    ensure_dir(bank_dir)

    metadata = {
        "bank_name": bank.name,
        "classes": LABEL_CODES,
        "total_before_dedup": bank.total_before_dedup,
        "total_after_dedup": bank.total_after_dedup,
        "dedup_removed": bank.dedup_removed,
        "dedup_rule": (
            "Keep per-class selections separate first, then merge duplicate concept texts by first occurrence "
            "in class order HGC->LGC->NTL->NST and ascending retrieval rank. Later duplicate classes are stored in source_classes."
        ),
        "n_concepts_by_class": {class_name: len(bank.per_class_rows[class_name]) for class_name in LABEL_CODES},
    }
    write_json(bank_dir / "metadata.json", metadata)

    for class_name in LABEL_CODES:
        class_rows = []
        for row in bank.per_class_rows[class_name]:
            class_rows.append(
                {
                    "bank_name": bank.name,
                    "class_name": class_name,
                    "concept_id": row["concept_id"],
                    "concept": row["concept"],
                    "source_rank_in_filtered_top300": int(row["source_rank_in_filtered_top300"]),
                    "retrieval_rank_in_class": int(row["retrieval_rank_in_class"]),
                    "ranking_method": row["ranking_method"],
                    "ranking_score": float(row["ranking_score"]),
                    "mu_same": float(row["mu_same"]),
                    "mu_diff": float(row["mu_diff"]),
                    "mu_diff_max": float(row["mu_diff_max"]),
                    "margin_vs_rest": float(row["margin_vs_rest"]),
                    "margin_vs_max_other": float(row["margin_vs_max_other"]),
                    "auroc_ovr": float(row["auroc_ovr"]),
                    **{f"mu_{other_class_name}": float(row[f'mu_{other_class_name}']) for other_class_name in LABEL_CODES},
                }
            )
        write_csv(
            bank_dir / f"{class_name}.csv",
            [
                "bank_name",
                "class_name",
                "concept_id",
                "concept",
                "source_rank_in_filtered_top300",
                "retrieval_rank_in_class",
                "ranking_method",
                "ranking_score",
                "mu_same",
                "mu_diff",
                "mu_diff_max",
                "margin_vs_rest",
                "margin_vs_max_other",
                "auroc_ovr",
                *[f"mu_{other_class_name}" for other_class_name in LABEL_CODES],
            ],
            class_rows,
        )
        with (bank_dir / f"{class_name}.txt").open("w", encoding="utf-8") as handle:
            for row in class_rows:
                handle.write(f"{row['concept']}\n")

    merged_rows = []
    for row in bank.merged_rows:
        merged_rows.append(
            {
                "bank_name": bank.name,
                "merged_index": int(row["merged_index"]),
                "concept": row["concept"],
                "primary_class": row["primary_class"],
                "primary_concept_id": row["primary_concept_id"],
                "primary_source_rank_in_filtered_top300": int(row["primary_source_rank_in_filtered_top300"]),
                "primary_retrieval_rank_in_class": int(row["primary_retrieval_rank_in_class"]),
                "primary_ranking_score": float(row["primary_ranking_score"]),
                "source_classes": "|".join(row["source_classes"]),
                "source_concept_ids": "|".join(row["source_concept_ids"]),
                "source_retrieval_ranks": "|".join(str(item) for item in row["source_retrieval_ranks"]),
                "source_filtered_top300_ranks": "|".join(str(item) for item in row["source_filtered_top300_ranks"]),
                "source_class_count": int(row["source_class_count"]),
            }
        )
    write_csv(
        bank_dir / "merged_concepts.csv",
        [
            "bank_name",
            "merged_index",
            "concept",
            "primary_class",
            "primary_concept_id",
            "primary_source_rank_in_filtered_top300",
            "primary_retrieval_rank_in_class",
            "primary_ranking_score",
            "source_classes",
            "source_concept_ids",
            "source_retrieval_ranks",
            "source_filtered_top300_ranks",
            "source_class_count",
        ],
        merged_rows,
    )

    write_json(bank_dir / "final_concepts.json", {"final_concepts": bank.merged_concepts})
    np.savez_compressed(
        bank_dir / "filtered_concept_text_embeddings.npz",
        concepts=np.array(bank.merged_concepts, dtype=object),
        concept_embeddings=bank.merged_embeddings.astype(np.float32),
    )


def resolve_schedule(args: argparse.Namespace) -> None:
    if args.schedule == "short_debug":
        baseline_epochs = 6
        warmup_epochs = 3
        joint_epochs = 6
    elif args.schedule == "improved_main":
        baseline_epochs = 10
        warmup_epochs = 5
        joint_epochs = 15
    else:
        raise ValueError(f"Unsupported schedule: {args.schedule}")
    if args.baseline_epochs is None:
        args.baseline_epochs = baseline_epochs
    if args.warmup_epochs is None:
        args.warmup_epochs = warmup_epochs
    if args.joint_epochs is None:
        args.joint_epochs = joint_epochs


def bank_concept_class_membership(bank: RetrievalBank) -> np.ndarray:
    membership = np.zeros((len(LABEL_CODES), bank.total_after_dedup), dtype=np.float32)
    for concept_index, row in enumerate(bank.merged_rows):
        for class_name in row["source_classes"]:
            membership[CLASS_TO_INDEX[str(class_name)], concept_index] = 1.0
    return membership


def bank_per_class_text_embeddings(bank: RetrievalBank) -> dict[str, np.ndarray]:
    per_class_embeddings: dict[str, np.ndarray] = {}
    for class_name in LABEL_CODES:
        per_class_embeddings[class_name] = l2_normalize(
            np.stack([row["embedding"] for row in bank.per_class_rows[class_name]], axis=0).astype(np.float32)
        )
    return per_class_embeddings


def plot_class_similarity_heatmap(
    matrix: np.ndarray,
    path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    image = ax.imshow(matrix, cmap="viridis")
    ax.set_xticks(np.arange(len(LABEL_CODES)))
    ax.set_yticks(np.arange(len(LABEL_CODES)))
    ax.set_xticklabels(LABEL_CODES)
    ax.set_yticklabels(LABEL_CODES)
    ax.set_xlabel("Concept Class")
    ax.set_ylabel("Image Class")
    ax.set_title(title)
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            ax.text(col_index, row_index, f"{matrix[row_index, col_index]:.3f}", ha="center", va="center", color="white")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def compute_class_similarity_matrices(
    banks: dict[str, RetrievalBank],
    split_payloads: dict[str, dict[str, Any]],
    output_root: Path,
) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    outputs_dir = output_root / "outputs"
    ensure_dir(outputs_dir)
    for bank_name, bank in banks.items():
        per_class_text = bank_per_class_text_embeddings(bank)
        file_bank_name = bank_name.replace("retrieval_", "").replace("_per_class", "")
        for split_name in ["train", "val"]:
            split_rows = split_payloads[split_name]["rows"]
            split_embeddings = split_payloads[split_name]["embeddings"]
            matrix = np.zeros((len(LABEL_CODES), len(LABEL_CODES)), dtype=np.float32)
            for image_class in LABEL_CODES:
                image_mask = np.array([str(row["class_name"]) == image_class for row in split_rows], dtype=bool)
                image_class_embeddings = split_embeddings[image_mask]
                if image_class_embeddings.size == 0:
                    raise RuntimeError(f"No {split_name} images for class {image_class}")
                for concept_class in LABEL_CODES:
                    text_embeddings = per_class_text[concept_class]
                    matrix[CLASS_TO_INDEX[image_class], CLASS_TO_INDEX[concept_class]] = float(
                        (image_class_embeddings @ text_embeddings.T).mean()
                    )

            csv_rows = []
            for row_index, image_class in enumerate(LABEL_CODES):
                csv_row: dict[str, Any] = {"image_class": image_class}
                for col_index, concept_class in enumerate(LABEL_CODES):
                    csv_row[concept_class] = float(matrix[row_index, col_index])
                csv_rows.append(csv_row)

            csv_path = outputs_dir / f"class_similarity_{file_bank_name}_{split_name}.csv"
            write_csv(csv_path, ["image_class", *LABEL_CODES], csv_rows)
            plot_class_similarity_heatmap(
                matrix=matrix,
                path=outputs_dir / f"class_similarity_{file_bank_name}_{split_name}.png",
                title=f"{bank_name} image-text similarity ({split_name})",
            )

            diagonal_values = np.diag(matrix)
            off_diagonal_values = matrix[~np.eye(matrix.shape[0], dtype=bool)]
            summary_rows.append(
                {
                    "bank_name": bank_name,
                    "split": split_name,
                    "matrix_csv": str(csv_path),
                    "mean_diagonal": float(diagonal_values.mean()),
                    "mean_off_diagonal": float(off_diagonal_values.mean()),
                    "diagonal_minus_off_diagonal": float(diagonal_values.mean() - off_diagonal_values.mean()),
                }
            )
    return summary_rows


def precision_at_k(relevance_sorted: np.ndarray, k: int) -> float:
    topk = relevance_sorted[:k]
    if topk.size < k:
        topk = np.pad(topk, (0, k - topk.size))
    return float(topk.sum() / k)


def ndcg_at_k(relevance_sorted: np.ndarray, k: int) -> float:
    topk = relevance_sorted[:k]
    if topk.size < k:
        topk = np.pad(topk, (0, k - topk.size))
    gains = (2.0 ** topk - 1.0) / np.log2(np.arange(2, 2 + k, dtype=np.float32))
    dcg = float(gains.sum())
    ideal = np.sort(relevance_sorted)[::-1][:k]
    if ideal.size < k:
        ideal = np.pad(ideal, (0, k - ideal.size))
    ideal_gains = (2.0 ** ideal - 1.0) / np.log2(np.arange(2, 2 + k, dtype=np.float32))
    idcg = float(ideal_gains.sum())
    if idcg <= 0.0:
        return 0.0
    return dcg / idcg


def evaluate_retrieval_on_val(
    banks: dict[str, RetrievalBank],
    split_payloads: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    val_rows = split_payloads["val"]["rows"]
    val_embeddings = split_payloads["val"]["embeddings"]
    metrics_rows: list[dict[str, Any]] = []

    for bank_name, bank in banks.items():
        similarities = val_embeddings @ bank.merged_embeddings.T
        membership = [set(row["source_classes"]) for row in bank.merged_rows]

        per_class_scores: dict[str, list[dict[str, float]]] = {class_name: [] for class_name in LABEL_CODES}
        for query_row, similarity in zip(val_rows, similarities, strict=True):
            class_name = str(query_row["class_name"])
            order = np.argsort(-similarity)
            relevance = np.array([1.0 if class_name in membership[index] else 0.0 for index in order], dtype=np.float32)
            per_class_scores[class_name].append(
                {
                    "precision_at_10": precision_at_k(relevance, 10),
                    "precision_at_20": precision_at_k(relevance, 20),
                    "ndcg_at_10": ndcg_at_k(relevance, 10),
                    "ndcg_at_20": ndcg_at_k(relevance, 20),
                }
            )

        class_rows = []
        for class_name in LABEL_CODES:
            class_metrics = per_class_scores[class_name]
            row = {
                "bank_name": bank_name,
                "split": "val",
                "class_name": class_name,
                "n_queries": len(class_metrics),
                "n_gallery_concepts_before_dedup": bank.total_before_dedup,
                "n_gallery_concepts_after_dedup": bank.total_after_dedup,
                "precision_at_10": float(np.mean([item["precision_at_10"] for item in class_metrics])) if class_metrics else float("nan"),
                "precision_at_20": float(np.mean([item["precision_at_20"] for item in class_metrics])) if class_metrics else float("nan"),
                "ndcg_at_10": float(np.mean([item["ndcg_at_10"] for item in class_metrics])) if class_metrics else float("nan"),
                "ndcg_at_20": float(np.mean([item["ndcg_at_20"] for item in class_metrics])) if class_metrics else float("nan"),
            }
            metrics_rows.append(row)
            class_rows.append(row)

        metrics_rows.append(
            {
                "bank_name": bank_name,
                "split": "val",
                "class_name": "macro_avg",
                "n_queries": sum(int(row["n_queries"]) for row in class_rows),
                "n_gallery_concepts_before_dedup": bank.total_before_dedup,
                "n_gallery_concepts_after_dedup": bank.total_after_dedup,
                "precision_at_10": float(np.mean([row["precision_at_10"] for row in class_rows])),
                "precision_at_20": float(np.mean([row["precision_at_20"] for row in class_rows])),
                "ndcg_at_10": float(np.mean([row["ndcg_at_10"] for row in class_rows])),
                "ndcg_at_20": float(np.mean([row["ndcg_at_20"] for row in class_rows])),
            }
        )
    return metrics_rows


def build_soft_targets_from_cached(
    bank: RetrievalBank,
    split_payloads: dict[str, dict[str, Any]],
    output_dir: Path,
) -> tuple[dict[str, list[dict[str, Any]]], np.ndarray, np.ndarray, np.ndarray]:
    ensure_dir(output_dir)
    ordered_manifest: list[dict[str, str]] = []
    embeddings_chunks: list[np.ndarray] = []
    for split in ["train", "val", "test"]:
        for row in split_payloads[split]["rows"]:
            ordered_manifest.append(
                {
                    "image_path": str(row["image_path"]),
                    "image_name": str(row.get("annotation_image_name") or Path(str(row["image_path"])).name),
                    "split": split,
                    "class_code": str(row["class_name"]),
                }
            )
        embeddings_chunks.append(split_payloads[split]["embeddings"])

    image_embeddings = np.concatenate(embeddings_chunks, axis=0).astype(np.float32)
    similarity_matrix = image_embeddings @ bank.merged_embeddings.T

    split_array = np.array([item["split"] for item in ordered_manifest], dtype=object)
    label_array = np.array([item["class_code"] for item in ordered_manifest], dtype=object)
    train_mask = split_array == "train"

    mu = similarity_matrix[train_mask].mean(axis=0).astype(np.float32)
    sigma = similarity_matrix[train_mask].std(axis=0).astype(np.float32)
    soft_concepts = 1.0 / (1.0 + np.exp(-(similarity_matrix - mu) / (sigma + EPS)))
    soft_concepts = soft_concepts.astype(np.float32)

    prototypes = []
    for class_name in LABEL_CODES:
        class_mask = train_mask & (label_array == class_name)
        prototypes.append(soft_concepts[class_mask].mean(axis=0))
    prototype_matrix = np.stack(prototypes, axis=0).astype(np.float32)

    save_soft_concept_artifacts(
        output_dir=output_dir,
        manifest=ordered_manifest,
        image_embeddings=image_embeddings,
        similarity_matrix=similarity_matrix.astype(np.float32),
        soft_concepts=soft_concepts,
        mu=mu,
        sigma=sigma,
        prototypes=prototype_matrix,
        concepts=bank.merged_concepts,
    )
    split_rows = group_records_by_split(ordered_manifest, image_embeddings, soft_concepts)
    return split_rows, prototype_matrix, mu, sigma


def make_train_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        baseline_epochs=args.baseline_epochs,
        warmup_epochs=args.warmup_epochs,
        joint_epochs=args.joint_epochs,
        baseline_lr=args.baseline_lr,
        main_lr=args.main_lr,
        weight_decay=args.weight_decay,
        lambda_concept=args.lambda_concept,
        lambda_align=args.lambda_align,
        lambda_cycl=args.lambda_cycl,
        lambda_img_txt=args.lambda_img_txt,
        alpha=args.alpha,
        tau=args.tau,
        tau_img_txt=args.tau_img_txt,
        dropout=args.dropout,
        proj_dim=args.proj_dim,
        num_views=args.num_views,
        use_color_aug=args.use_color_aug,
        use_geom_aug=args.use_geom_aug,
        use_image_text_pairing=args.use_image_text_pairing,
        track_pair_diagnostics=True,
    )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def seed_dir_name(seed: int) -> str:
    return f"seed_{seed:03d}"


def run_image_baseline_seed(
    bank: RetrievalBank,
    split_payloads: dict[str, dict[str, Any]],
    resolved_device: str,
    args: argparse.Namespace,
    experiment_dir: Path,
    seed: int,
) -> dict[str, Any]:
    ensure_dir(experiment_dir)
    split_rows, prototype_matrix, mu, sigma = build_soft_targets_from_cached(bank, split_payloads, experiment_dir)
    write_json(
        experiment_dir / "experiment_input_bank.json",
        {
            "bank_name": bank.name,
            "model_type": "image_baseline",
            "seed": seed,
            "concept_count_before_dedup": bank.total_before_dedup,
            "concept_count_after_dedup": bank.total_after_dedup,
            "mu_shape": list(mu.shape),
            "sigma_shape": list(sigma.shape),
        },
    )
    set_seed(seed)
    baseline_metrics, baseline_records = train_baseline(
        train_rows=split_rows["train"],
        val_rows=split_rows["val"],
        test_rows=split_rows["test"],
        device=resolved_device,
        output_dir=experiment_dir,
        epochs=args.baseline_epochs,
        learning_rate=args.baseline_lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
    )
    baseline_cm = np.array(baseline_metrics["confusion_matrix"], dtype=np.int64)
    save_confusion_figure(
        baseline_cm,
        experiment_dir / "baseline_confusion_matrix.png",
        f"Baseline Confusion Matrix ({bank.name})",
    )
    save_prediction_records(experiment_dir / "baseline_test_predictions.csv", baseline_records, None, None)
    payload = load_json(experiment_dir / "baseline_metrics.json")
    write_json(
        experiment_dir / "experiment_summary.json",
        {
            "bank_name": bank.name,
            "model_type": "image_baseline",
            "seed": seed,
            "concept_count_before_dedup": bank.total_before_dedup,
            "concept_count_after_dedup": bank.total_after_dedup,
            "device": resolved_device,
            "baseline_test": baseline_metrics,
            "train_count": len(split_rows["train"]),
            "val_count": len(split_rows["val"]),
            "test_count": len(split_rows["test"]),
        },
    )
    return {
        "setting": "image_baseline_filtered_top300",
        "model_type": "image_baseline",
        "bank_name": bank.name,
        "seed": seed,
        "concept_count_before_dedup": bank.total_before_dedup,
        "concept_count_after_dedup": bank.total_after_dedup,
        "val_accuracy": float(payload["val"]["accuracy"]),
        "val_macro_f1": float(payload["val"]["macro_f1"]),
        "val_macro_auroc": float(payload["val"]["macro_auroc"]) if payload["val"]["macro_auroc"] is not None else float("nan"),
        "test_accuracy": float(baseline_metrics["accuracy"]),
        "test_macro_f1": float(baseline_metrics["macro_f1"]),
        "test_macro_auroc": float(baseline_metrics["macro_auroc"]) if baseline_metrics["macro_auroc"] is not None else float("nan"),
        "output_dir": str(experiment_dir),
    }


def run_cbm_seed(
    bank: RetrievalBank,
    split_payloads: dict[str, dict[str, Any]],
    image_encoder: Any,
    preprocess_val: Any,
    resolved_device: str,
    args: argparse.Namespace,
    experiment_dir: Path,
    seed: int,
    model_type: str,
    use_cycl: bool,
) -> dict[str, Any]:
    ensure_dir(experiment_dir)
    split_rows, prototype_matrix, mu, sigma = build_soft_targets_from_cached(bank, split_payloads, experiment_dir)
    concept_class_membership = bank_concept_class_membership(bank)
    write_json(
        experiment_dir / "experiment_input_bank.json",
        {
            "bank_name": bank.name,
            "model_type": model_type,
            "seed": seed,
            "concept_count_before_dedup": bank.total_before_dedup,
            "concept_count_after_dedup": bank.total_after_dedup,
            "mu_shape": list(mu.shape),
            "sigma_shape": list(sigma.shape),
            "num_views": args.num_views,
            "use_color_aug": args.use_color_aug,
            "use_geom_aug": args.use_geom_aug,
            "use_image_text_pairing": args.use_image_text_pairing,
            "schedule": args.schedule,
        },
    )
    embedding_dim = int(split_rows["train"][0]["embedding"].shape[0])
    class_weights = compute_class_weights(split_rows["train"], resolved_device)
    train_loader, val_loader, test_loader = build_loaders(
        split_rows=split_rows,
        eval_transform=preprocess_val,
        prototype_matrix=prototype_matrix,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=resolved_device,
        num_views=args.num_views,
        use_color_aug=args.use_color_aug,
        use_geom_aug=args.use_geom_aug,
    )
    set_seed(seed)
    main_metrics, main_records, main_model = train_main_model(
        image_encoder=image_encoder,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        output_dir=experiment_dir,
        device=resolved_device,
        embedding_dim=embedding_dim,
        num_concepts=bank.total_after_dedup,
        class_weights=class_weights,
        prototype_matrix=prototype_matrix,
        args=make_train_args(args),
        bank_text_embeddings=bank.merged_embeddings,
        concept_class_membership=concept_class_membership,
        use_cycl=use_cycl,
    )
    classifier_weight = main_model.classifier.weight.detach().cpu().numpy()
    main_cm = np.array(main_metrics["confusion_matrix"], dtype=np.int64)
    save_confusion_figure(
        main_cm,
        experiment_dir / "main_confusion_matrix.png",
        f"{model_type} Confusion Matrix ({bank.name})",
    )
    save_prediction_records(
        experiment_dir / "main_test_predictions.csv",
        main_records,
        concepts=bank.merged_concepts,
        classifier_weight=classifier_weight,
    )
    save_case_examples_figure(
        experiment_dir / "main_case_examples.png",
        select_case_examples(main_records),
        concepts=bank.merged_concepts,
        classifier_weight=classifier_weight,
    )
    payload = load_json(experiment_dir / "main_metrics.json")
    write_json(
        experiment_dir / "experiment_summary.json",
        {
            "bank_name": bank.name,
            "model_type": model_type,
            "seed": seed,
            "concept_count_before_dedup": bank.total_before_dedup,
            "concept_count_after_dedup": bank.total_after_dedup,
            "device": resolved_device,
            "main_test": main_metrics,
            "pair_diagnostics": payload.get("pair_diagnostics", {}),
            "image_text_diagnostics": payload.get("image_text_diagnostics", {}),
            "train_count": len(split_rows["train"]),
            "val_count": len(split_rows["val"]),
            "test_count": len(split_rows["test"]),
        },
    )
    return {
        "setting": f"{model_type}_{bank.name}",
        "model_type": model_type,
        "bank_name": bank.name,
        "seed": seed,
        "concept_count_before_dedup": bank.total_before_dedup,
        "concept_count_after_dedup": bank.total_after_dedup,
        "val_accuracy": float(payload["val"]["accuracy"]),
        "val_macro_f1": float(payload["val"]["macro_f1"]),
        "val_macro_auroc": float(payload["val"]["macro_auroc"]) if payload["val"]["macro_auroc"] is not None else float("nan"),
        "test_accuracy": float(main_metrics["accuracy"]),
        "test_macro_f1": float(main_metrics["macro_f1"]),
        "test_macro_auroc": float(main_metrics["macro_auroc"]) if main_metrics["macro_auroc"] is not None else float("nan"),
        "pair_positive_pair_count": float(payload.get("pair_diagnostics", {}).get("positive_pair_count", float("nan"))),
        "pair_negative_pair_count": float(payload.get("pair_diagnostics", {}).get("negative_pair_count", float("nan"))),
        "pair_positive_weight_mean": float(payload.get("pair_diagnostics", {}).get("positive_weight_mean", float("nan"))),
        "pair_negative_weight_mean": float(payload.get("pair_diagnostics", {}).get("negative_weight_mean", float("nan"))),
        "image_text_positive_similarity_mean": float(payload.get("image_text_diagnostics", {}).get("image_text_positive_similarity_mean", float("nan"))),
        "image_text_negative_similarity_mean": float(payload.get("image_text_diagnostics", {}).get("image_text_negative_similarity_mean", float("nan"))),
        "image_text_similarity_gap": float(payload.get("image_text_diagnostics", {}).get("image_text_similarity_gap", float("nan"))),
        "num_views": int(args.num_views),
        "schedule": str(args.schedule),
        "ranking_method": str(args.ranking_method),
        "output_dir": str(experiment_dir),
    }


def aggregate_seed_rows(training_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, int, int, int, str, str], list[dict[str, Any]]] = {}
    for row in training_rows:
        key = (
            str(row["setting"]),
            str(row["model_type"]),
            str(row["bank_name"]),
            int(row["concept_count_before_dedup"]),
            int(row["concept_count_after_dedup"]),
            int(row["num_views"]),
            str(row["schedule"]),
            str(row["ranking_method"]),
        )
        grouped.setdefault(key, []).append(row)

    metric_names = [
        "val_accuracy",
        "val_macro_f1",
        "val_macro_auroc",
        "test_accuracy",
        "test_macro_f1",
        "test_macro_auroc",
        "retrieval_precision_at_10_val_macro",
        "retrieval_precision_at_20_val_macro",
        "retrieval_ndcg_at_10_val_macro",
        "retrieval_ndcg_at_20_val_macro",
        "pair_positive_pair_count",
        "pair_negative_pair_count",
        "pair_positive_weight_mean",
        "pair_negative_weight_mean",
        "image_text_positive_similarity_mean",
        "image_text_negative_similarity_mean",
        "image_text_similarity_gap",
    ]
    aggregate_rows: list[dict[str, Any]] = []
    for key, rows in grouped.items():
        setting, model_type, bank_name, count_before, count_after, num_views, schedule, ranking_method = key
        summary = {
            "setting": setting,
            "model_type": model_type,
            "bank_name": bank_name,
            "n_seeds": len(rows),
            "concept_count_before_dedup": count_before,
            "concept_count_after_dedup": count_after,
            "num_views": num_views,
            "schedule": schedule,
            "ranking_method": ranking_method,
            "seed_list": ",".join(str(int(row["seed"])) for row in sorted(rows, key=lambda item: int(item["seed"]))),
        }
        for metric_name in metric_names:
            values = np.array([float(row[metric_name]) for row in rows], dtype=np.float64)
            summary[f"{metric_name}_mean"] = float(values.mean())
            summary[f"{metric_name}_std"] = float(values.std(ddof=0))
        aggregate_rows.append(summary)
    aggregate_rows.sort(key=lambda row: (str(row["model_type"]), str(row["bank_name"])))
    return aggregate_rows


def run_filtered_top300_baseline_and_cycl(
    bank: RetrievalBank,
    split_payloads: dict[str, dict[str, Any]],
    image_encoder: Any,
    preprocess_val: Any,
    resolved_device: str,
    args: argparse.Namespace,
    experiment_dir: Path,
) -> list[dict[str, Any]]:
    ensure_dir(experiment_dir)
    split_rows, prototype_matrix, mu, sigma = build_soft_targets_from_cached(bank, split_payloads, experiment_dir)
    write_json(
        experiment_dir / "experiment_input_bank.json",
        {
            "bank_name": bank.name,
            "concept_count_before_dedup": bank.total_before_dedup,
            "concept_count_after_dedup": bank.total_after_dedup,
            "mu_shape": list(mu.shape),
            "sigma_shape": list(sigma.shape),
        },
    )

    set_seed(args.seed)
    baseline_metrics, baseline_records = train_baseline(
        train_rows=split_rows["train"],
        val_rows=split_rows["val"],
        test_rows=split_rows["test"],
        device=resolved_device,
        output_dir=experiment_dir,
        epochs=args.baseline_epochs,
        learning_rate=args.baseline_lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
    )
    baseline_cm = np.array(baseline_metrics["confusion_matrix"], dtype=np.int64)
    save_confusion_figure(
        baseline_cm,
        experiment_dir / "baseline_confusion_matrix.png",
        f"Baseline Confusion Matrix ({bank.name})",
    )
    save_prediction_records(experiment_dir / "baseline_test_predictions.csv", baseline_records, None, None)

    embedding_dim = int(split_rows["train"][0]["embedding"].shape[0])
    class_weights = compute_class_weights(split_rows["train"], resolved_device)
    train_loader, val_loader, test_loader = build_loaders(
        split_rows=split_rows,
        eval_transform=preprocess_val,
        prototype_matrix=prototype_matrix,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=resolved_device,
    )

    set_seed(args.seed)
    main_metrics, main_records, main_model = train_main_model(
        image_encoder=image_encoder,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        output_dir=experiment_dir,
        device=resolved_device,
        embedding_dim=embedding_dim,
        num_concepts=bank.total_after_dedup,
        class_weights=class_weights,
        prototype_matrix=prototype_matrix,
        args=make_train_args(args),
    )

    main_cm = np.array(main_metrics["confusion_matrix"], dtype=np.int64)
    save_confusion_figure(
        main_cm,
        experiment_dir / "main_confusion_matrix.png",
        f"CyCL Confusion Matrix ({bank.name})",
    )
    classifier_weight = main_model.classifier.weight.detach().cpu().numpy()
    save_prediction_records(
        experiment_dir / "main_test_predictions.csv",
        main_records,
        concepts=bank.merged_concepts,
        classifier_weight=classifier_weight,
    )
    save_case_examples_figure(
        experiment_dir / "main_case_examples.png",
        select_case_examples(main_records),
        concepts=bank.merged_concepts,
        classifier_weight=classifier_weight,
    )

    experiment_summary = {
        "bank_name": bank.name,
        "concept_count_before_dedup": bank.total_before_dedup,
        "concept_count_after_dedup": bank.total_after_dedup,
        "device": resolved_device,
        "baseline_test": baseline_metrics,
        "cycl_test": main_metrics,
        "train_count": len(split_rows["train"]),
        "val_count": len(split_rows["val"]),
        "test_count": len(split_rows["test"]),
    }
    write_json(experiment_dir / "experiment_summary.json", experiment_summary)

    rows = [
        {
            "setting": "baseline_filtered_top300",
            "model_type": "baseline",
            "bank_name": bank.name,
            "concept_count_before_dedup": bank.total_before_dedup,
            "concept_count_after_dedup": bank.total_after_dedup,
            "val_accuracy": float(json.loads((experiment_dir / "baseline_metrics.json").read_text(encoding="utf-8"))["val"]["accuracy"]),
            "val_macro_f1": float(json.loads((experiment_dir / "baseline_metrics.json").read_text(encoding="utf-8"))["val"]["macro_f1"]),
            "val_macro_auroc": float(json.loads((experiment_dir / "baseline_metrics.json").read_text(encoding="utf-8"))["val"]["macro_auroc"])
            if json.loads((experiment_dir / "baseline_metrics.json").read_text(encoding="utf-8"))["val"]["macro_auroc"] is not None
            else float("nan"),
            "test_accuracy": float(baseline_metrics["accuracy"]),
            "test_macro_f1": float(baseline_metrics["macro_f1"]),
            "test_macro_auroc": float(baseline_metrics["macro_auroc"]) if baseline_metrics["macro_auroc"] is not None else float("nan"),
            "output_dir": str(experiment_dir),
        },
        {
            "setting": "cycl_filtered_top300",
            "model_type": "cycl",
            "bank_name": bank.name,
            "concept_count_before_dedup": bank.total_before_dedup,
            "concept_count_after_dedup": bank.total_after_dedup,
            "val_accuracy": float(json.loads((experiment_dir / "main_metrics.json").read_text(encoding="utf-8"))["val"]["accuracy"]),
            "val_macro_f1": float(json.loads((experiment_dir / "main_metrics.json").read_text(encoding="utf-8"))["val"]["macro_f1"]),
            "val_macro_auroc": float(json.loads((experiment_dir / "main_metrics.json").read_text(encoding="utf-8"))["val"]["macro_auroc"])
            if json.loads((experiment_dir / "main_metrics.json").read_text(encoding="utf-8"))["val"]["macro_auroc"] is not None
            else float("nan"),
            "test_accuracy": float(main_metrics["accuracy"]),
            "test_macro_f1": float(main_metrics["macro_f1"]),
            "test_macro_auroc": float(main_metrics["macro_auroc"]) if main_metrics["macro_auroc"] is not None else float("nan"),
            "output_dir": str(experiment_dir),
        },
    ]
    return rows


def run_cycl_only(
    bank: RetrievalBank,
    split_payloads: dict[str, dict[str, Any]],
    image_encoder: Any,
    preprocess_val: Any,
    resolved_device: str,
    args: argparse.Namespace,
    experiment_dir: Path,
) -> dict[str, Any]:
    ensure_dir(experiment_dir)
    split_rows, prototype_matrix, mu, sigma = build_soft_targets_from_cached(bank, split_payloads, experiment_dir)
    write_json(
        experiment_dir / "experiment_input_bank.json",
        {
            "bank_name": bank.name,
            "concept_count_before_dedup": bank.total_before_dedup,
            "concept_count_after_dedup": bank.total_after_dedup,
            "mu_shape": list(mu.shape),
            "sigma_shape": list(sigma.shape),
        },
    )

    embedding_dim = int(split_rows["train"][0]["embedding"].shape[0])
    class_weights = compute_class_weights(split_rows["train"], resolved_device)
    train_loader, val_loader, test_loader = build_loaders(
        split_rows=split_rows,
        eval_transform=preprocess_val,
        prototype_matrix=prototype_matrix,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=resolved_device,
    )

    set_seed(args.seed)
    main_metrics, main_records, main_model = train_main_model(
        image_encoder=image_encoder,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        output_dir=experiment_dir,
        device=resolved_device,
        embedding_dim=embedding_dim,
        num_concepts=bank.total_after_dedup,
        class_weights=class_weights,
        prototype_matrix=prototype_matrix,
        args=make_train_args(args),
    )

    classifier_weight = main_model.classifier.weight.detach().cpu().numpy()
    main_cm = np.array(main_metrics["confusion_matrix"], dtype=np.int64)
    save_confusion_figure(
        main_cm,
        experiment_dir / "main_confusion_matrix.png",
        f"CyCL Confusion Matrix ({bank.name})",
    )
    save_prediction_records(
        experiment_dir / "main_test_predictions.csv",
        main_records,
        concepts=bank.merged_concepts,
        classifier_weight=classifier_weight,
    )
    save_case_examples_figure(
        experiment_dir / "main_case_examples.png",
        select_case_examples(main_records),
        concepts=bank.merged_concepts,
        classifier_weight=classifier_weight,
    )
    write_json(
        experiment_dir / "experiment_summary.json",
        {
            "bank_name": bank.name,
            "concept_count_before_dedup": bank.total_before_dedup,
            "concept_count_after_dedup": bank.total_after_dedup,
            "device": resolved_device,
            "cycl_test": main_metrics,
            "train_count": len(split_rows["train"]),
            "val_count": len(split_rows["val"]),
            "test_count": len(split_rows["test"]),
        },
    )
    main_metrics_payload = json.loads((experiment_dir / "main_metrics.json").read_text(encoding="utf-8"))
    return {
        "setting": f"cycl_{bank.name}",
        "model_type": "cycl",
        "bank_name": bank.name,
        "concept_count_before_dedup": bank.total_before_dedup,
        "concept_count_after_dedup": bank.total_after_dedup,
        "val_accuracy": float(main_metrics_payload["val"]["accuracy"]),
        "val_macro_f1": float(main_metrics_payload["val"]["macro_f1"]),
        "val_macro_auroc": float(main_metrics_payload["val"]["macro_auroc"])
        if main_metrics_payload["val"]["macro_auroc"] is not None
        else float("nan"),
        "test_accuracy": float(main_metrics["accuracy"]),
        "test_macro_f1": float(main_metrics["macro_f1"]),
        "test_macro_auroc": float(main_metrics["macro_auroc"]) if main_metrics["macro_auroc"] is not None else float("nan"),
        "output_dir": str(experiment_dir),
    }


def maybe_run_training(
    banks: dict[str, RetrievalBank],
    split_payloads: dict[str, dict[str, Any]],
    output_root: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    results_rows: list[dict[str, Any]] = []
    experiment_root = output_root / "experiments"
    ensure_dir(experiment_root)
    requested_banks = [item.strip() for item in args.run_banks.split(",") if item.strip()]
    requested_models = [item.strip() for item in args.run_models.split(",") if item.strip()]
    seeds = parse_seed_list(args.seeds)

    image_encoder, _, preprocess_val, resolved_device = load_biomedclip(args.device)
    for parameter in image_encoder.parameters():
        parameter.requires_grad = False
    image_encoder.eval()

    def row_from_existing(
        bank: RetrievalBank,
        payload: dict[str, Any],
        local_args: argparse.Namespace,
        model_type: str,
        seed: int,
        experiment_dir: Path,
    ) -> dict[str, Any]:
        return {
            "setting": f"{model_type}_{bank.name}",
            "model_type": model_type,
            "bank_name": bank.name,
            "seed": seed,
            "concept_count_before_dedup": bank.total_before_dedup,
            "concept_count_after_dedup": bank.total_after_dedup,
            "val_accuracy": float(payload["val"]["accuracy"]),
            "val_macro_f1": float(payload["val"]["macro_f1"]),
            "val_macro_auroc": float(payload["val"]["macro_auroc"]) if payload["val"]["macro_auroc"] is not None else float("nan"),
            "test_accuracy": float(payload["test"]["accuracy"]),
            "test_macro_f1": float(payload["test"]["macro_f1"]),
            "test_macro_auroc": float(payload["test"]["macro_auroc"]) if payload["test"]["macro_auroc"] is not None else float("nan"),
            "pair_positive_pair_count": float(payload.get("pair_diagnostics", {}).get("positive_pair_count", float("nan"))),
            "pair_negative_pair_count": float(payload.get("pair_diagnostics", {}).get("negative_pair_count", float("nan"))),
            "pair_positive_weight_mean": float(payload.get("pair_diagnostics", {}).get("positive_weight_mean", float("nan"))),
            "pair_negative_weight_mean": float(payload.get("pair_diagnostics", {}).get("negative_weight_mean", float("nan"))),
            "image_text_positive_similarity_mean": float(payload.get("image_text_diagnostics", {}).get("image_text_positive_similarity_mean", float("nan"))),
            "image_text_negative_similarity_mean": float(payload.get("image_text_diagnostics", {}).get("image_text_negative_similarity_mean", float("nan"))),
            "image_text_similarity_gap": float(payload.get("image_text_diagnostics", {}).get("image_text_similarity_gap", float("nan"))),
            "num_views": int(local_args.num_views),
            "schedule": str(local_args.schedule),
            "ranking_method": str(local_args.ranking_method),
            "output_dir": str(experiment_dir),
        }

    def run_spec(bank_name: str, model_type: str, use_cycl: bool, seed: int, local_args: argparse.Namespace) -> None:
        bank = banks[bank_name]
        experiment_dir = (
            experiment_root
            / bank_name
            / model_type
            / f"views_{local_args.num_views}_{local_args.schedule}"
            / seed_dir_name(seed)
        )
        if args.skip_existing and (experiment_dir / "main_metrics.json").exists():
            payload = load_json(experiment_dir / "main_metrics.json")
            results_rows.append(
                row_from_existing(
                    bank=bank,
                    payload=payload,
                    local_args=local_args,
                    model_type=model_type,
                    seed=seed,
                    experiment_dir=experiment_dir,
                )
            )
        else:
            results_rows.append(
                run_cbm_seed(
                    bank=bank,
                    split_payloads=split_payloads,
                    image_encoder=image_encoder,
                    preprocess_val=preprocess_val,
                    resolved_device=resolved_device,
                    args=local_args,
                    experiment_dir=experiment_dir,
                    seed=seed,
                    model_type=model_type,
                    use_cycl=use_cycl,
                )
            )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for bank_name in requested_banks:
        for model_type, use_cycl in [("cbm_only", False), ("cycl", True)]:
            if model_type not in requested_models:
                continue
            for seed in seeds:
                run_spec(
                    bank_name=bank_name,
                    model_type=model_type,
                    use_cycl=use_cycl,
                    seed=seed,
                    local_args=args,
                )

    if args.run_view_ablation and "retrieval_top10_per_class_disc" in requested_banks and "cycl" in requested_models:
        ablation_seeds = parse_seed_list(args.view_ablation_seeds)
        for num_views in [2, 4]:
            local_args = clone_args(args, num_views=num_views)
            for seed in ablation_seeds:
                run_spec(
                    bank_name="retrieval_top10_per_class_disc",
                    model_type="cycl",
                    use_cycl=True,
                    seed=seed,
                    local_args=local_args,
                )
    return results_rows


def bank_summary_rows(banks: dict[str, RetrievalBank]) -> list[dict[str, Any]]:
    rows = []
    for bank_name, bank in banks.items():
        rows.append(
            {
                "bank_name": bank_name,
                "total_before_dedup": bank.total_before_dedup,
                "total_after_dedup": bank.total_after_dedup,
                "dedup_removed": bank.dedup_removed,
                **{f"{class_name}_count": len(bank.per_class_rows[class_name]) for class_name in LABEL_CODES},
            }
        )
    return rows


def attach_retrieval_macro_metrics(
    training_rows: list[dict[str, Any]],
    retrieval_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    macro_lookup = {
        row["bank_name"]: row
        for row in retrieval_rows
        if row["class_name"] == "macro_avg"
    }
    merged = []
    for row in training_rows:
        bank_macro = macro_lookup.get(row["bank_name"])
        merged_row = dict(row)
        if bank_macro is not None:
            merged_row["retrieval_precision_at_10_val_macro"] = float(bank_macro["precision_at_10"])
            merged_row["retrieval_precision_at_20_val_macro"] = float(bank_macro["precision_at_20"])
            merged_row["retrieval_ndcg_at_10_val_macro"] = float(bank_macro["ndcg_at_10"])
            merged_row["retrieval_ndcg_at_20_val_macro"] = float(bank_macro["ndcg_at_20"])
        merged.append(merged_row)
    return merged


def clone_args(args: argparse.Namespace, **overrides: Any) -> argparse.Namespace:
    payload = dict(vars(args))
    payload.update(overrides)
    return argparse.Namespace(**payload)


def write_markdown_report(
    output_root: Path,
    args: argparse.Namespace,
    banks: dict[str, RetrievalBank],
    retrieval_rows: list[dict[str, Any]],
    aggregate_rows: list[dict[str, Any]],
    class_similarity_rows: list[dict[str, Any]],
) -> None:
    retrieval_macro = {
        row["bank_name"]: row
        for row in retrieval_rows
        if row["class_name"] == "macro_avg"
    }
    class_similarity_lookup = {
        (row["bank_name"], row["split"]): row
        for row in class_similarity_rows
    }

    lines = [
        "# Revised EBTC Retrieval Compression + CBM/CyCL Report",
        "",
        "## What Changed Relative to the Previous Pipeline",
        "",
        "1. Filtering verification is no longer based only on retrieval P@k / nDCG; it now also uses 4x4 class-wise image-text similarity matrices.",
        "2. Concept ranking now supports a discriminative train-only score that penalizes similarity to non-target classes.",
        "3. Image-image positive/negative pair definitions are now explicit:",
        "   - positives = same class",
        "   - negatives = different class",
        "   - concept similarity only refines weights within the positive set.",
        "4. Image-concept pairing is added as an optional refinement branch.",
        "5. Training now supports multi-view augmentation with configurable color and geometric transforms.",
        "",
        "## Setup",
        "",
        "- Official EBTC split from `annotations.csv` only.",
        "- Starting bank: official-split `filtered_top300`.",
        f"- Simple ranking baseline retained: `mean_cosine`.",
        f"- Revised ranking used for the revised banks: `{args.ranking_method}`.",
        f"- Discriminative lambda: `{args.disc_lambda}`.",
        f"- Main schedule: `{args.schedule}` with baseline={args.baseline_epochs}, warmup={args.warmup_epochs}, joint={args.joint_epochs}.",
        f"- Main multi-seed list: `{args.seeds}`.",
        f"- Main views: `{args.num_views}`.",
        f"- Color augmentation enabled: `{args.use_color_aug}`.",
        f"- Geometric augmentation enabled: `{args.use_geom_aug}`.",
        f"- Image-concept pairing enabled: `{args.use_image_text_pairing}`.",
        "",
    ]
    lines.extend(["## Concept Banks", ""])
    for bank_name in sorted(banks.keys()):
        bank = banks[bank_name]
        lines.append(
            f"- `{bank_name}`: before dedup={bank.total_before_dedup}, after dedup={bank.total_after_dedup}, dedup_removed={bank.dedup_removed}"
        )
    lines.extend(["", "## Retrieval Metrics on Validation", ""])
    for bank_name in sorted(retrieval_macro.keys()):
        row = retrieval_macro.get(bank_name)
        if row is None:
            continue
        lines.append(
            f"- `{bank_name}`: P@10={row['precision_at_10']:.4f}, P@20={row['precision_at_20']:.4f}, "
            f"nDCG@10={row['ndcg_at_10']:.4f}, nDCG@20={row['ndcg_at_20']:.4f}"
        )

    lines.extend(["", "## 4x4 Class-Wise Image-Text Similarity Verification", ""])
    for bank_name in sorted({row["bank_name"] for row in class_similarity_rows}):
        train_row = class_similarity_lookup.get((bank_name, "train"))
        val_row = class_similarity_lookup.get((bank_name, "val"))
        if train_row is not None:
            lines.append(
                f"- `{bank_name}` train: diag={train_row['mean_diagonal']:.4f}, offdiag={train_row['mean_off_diagonal']:.4f}, margin={train_row['diagonal_minus_off_diagonal']:.4f}"
            )
        if val_row is not None:
            lines.append(
                f"- `{bank_name}` val: diag={val_row['mean_diagonal']:.4f}, offdiag={val_row['mean_off_diagonal']:.4f}, margin={val_row['diagonal_minus_off_diagonal']:.4f}"
            )

    lines.extend(["", "## Revised CBM / CyCL Results", ""])
    for row in sorted(
        aggregate_rows,
        key=lambda item: (
            float(item["test_macro_f1_mean"]),
            float(item["test_macro_auroc_mean"]),
        ),
        reverse=True,
    ):
        lines.append(
            f"- `{row['setting']}` | bank=`{row['bank_name']}` | model=`{row['model_type']}` | "
            f"views={row['num_views']} | schedule=`{row['schedule']}` | ranking=`{row['ranking_method']}` | "
            f"n={row['n_seeds']} | val_macro_f1={row['val_macro_f1_mean']:.4f}±{row['val_macro_f1_std']:.4f}, "
            f"test_accuracy={row['test_accuracy_mean']:.4f}±{row['test_accuracy_std']:.4f}, "
            f"test_macro_f1={row['test_macro_f1_mean']:.4f}±{row['test_macro_f1_std']:.4f}, "
            f"test_macro_auroc={row['test_macro_auroc_mean']:.4f}±{row['test_macro_auroc_std']:.4f}, "
            f"pair_w+={row['pair_positive_weight_mean_mean']:.4f}, "
            f"img-txt-gap={row['image_text_similarity_gap_mean']:.4f}"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The main filtering verification target is now the diagonal-vs-off-diagonal margin in the class-wise image-text similarity matrix.",
            "- Positive image-image pairs are explicitly same-class pairs only; concept similarity only reweights their strength.",
            "- Negative image-image pairs are explicitly cross-class pairs, regardless of concept overlap.",
            "- Image-concept pairing is an added branch and does not replace the concept bottleneck or classification path.",
            "- `CBM-no-CyCL` remains the fair control: same backbone, same adapter, same concept bottleneck, same warmup and alignment, but no image-image contrastive term.",
            f"- Raw per-seed rows: `{output_root / 'outputs' / 'revised_cycl_seed_results.csv'}`.",
            "",
        ]
    )
    (output_root / "outputs" / "revised_experiment_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    resolve_schedule(args)
    ensure_dir(args.output_dir)
    ensure_dir(args.output_dir / "outputs")
    ensure_dir(args.output_dir / "banks")

    set_seed(args.seed)
    ensure_official_split_embedding_cache(args.embeddings_dir, args.backup_embeddings_dir)

    split_payloads = load_cached_split_payloads(args.embeddings_dir)
    filtered_top300_candidates = load_filtered_top300(args.filtered_top300_dir, args.embeddings_dir)
    mean_rankings_by_class, mean_ranking_rows = compute_class_retrieval_scores(
        candidates_by_class=filtered_top300_candidates,
        split_payloads=split_payloads,
        ranking_method="mean_cosine",
        disc_lambda=args.disc_lambda,
    )
    revised_rankings_by_class, revised_ranking_rows = compute_class_retrieval_scores(
        candidates_by_class=filtered_top300_candidates,
        split_payloads=split_payloads,
        ranking_method=args.ranking_method,
        disc_lambda=args.disc_lambda,
    )

    write_csv(
        args.output_dir / "outputs" / "filtered_top300_train_ranking_scores.csv",
        [
            "class_name",
            "concept",
            "concept_id",
            "source_rank_in_filtered_top300",
            "embedding_index",
            "ranking_method",
            "ranking_score",
            "mu_same",
            "mu_diff",
            "mu_diff_max",
            "margin_vs_rest",
            "margin_vs_max_other",
            "auroc_ovr",
            *[f"mu_{class_name}" for class_name in LABEL_CODES],
            "retrieval_rank_in_class",
        ],
        mean_ranking_rows,
    )
    write_csv(
        args.output_dir / "outputs" / "revised_ranking_scores.csv",
        [
            "class_name",
            "concept",
            "concept_id",
            "source_rank_in_filtered_top300",
            "embedding_index",
            "ranking_method",
            "ranking_score",
            "mu_same",
            "mu_diff",
            "mu_diff_max",
            "margin_vs_rest",
            "margin_vs_max_other",
            "auroc_ovr",
            *[f"mu_{class_name}" for class_name in LABEL_CODES],
            "retrieval_rank_in_class",
        ],
        revised_ranking_rows,
    )

    baseline_banks = build_retrieval_banks(mean_rankings_by_class, bank_suffix="", include_filtered_top300=True)
    revised_banks = build_retrieval_banks(revised_rankings_by_class, bank_suffix="_disc", include_filtered_top300=False)
    banks = {**baseline_banks, **revised_banks}
    for bank_name, bank in banks.items():
        save_retrieval_bank(bank, args.output_dir / "banks" / bank_name)

    write_csv(
        args.output_dir / "outputs" / "bank_summary.csv",
        [
            "bank_name",
            "total_before_dedup",
            "total_after_dedup",
            "dedup_removed",
            "HGC_count",
            "LGC_count",
            "NTL_count",
            "NST_count",
        ],
        bank_summary_rows(banks),
    )

    for top_n in TOP_N_CHOICES:
        bank_name = f"retrieval_top{top_n}_per_class"
        write_csv(
            args.output_dir / "outputs" / f"{bank_name}.csv",
            [
                "bank_name",
                "class_name",
                "concept_id",
                "concept",
                "source_rank_in_filtered_top300",
                "retrieval_rank_in_class",
                "ranking_method",
                "ranking_score",
                "mu_same",
                "mu_diff",
                "mu_diff_max",
                "margin_vs_rest",
                "margin_vs_max_other",
                "auroc_ovr",
                *[f"mu_{class_name}" for class_name in LABEL_CODES],
            ],
            [
                {
                    "bank_name": bank_name,
                    "class_name": class_name,
                    "concept_id": row["concept_id"],
                    "concept": row["concept"],
                    "source_rank_in_filtered_top300": int(row["source_rank_in_filtered_top300"]),
                    "retrieval_rank_in_class": int(row["retrieval_rank_in_class"]),
                    "ranking_method": row["ranking_method"],
                    "ranking_score": float(row["ranking_score"]),
                    "mu_same": float(row["mu_same"]),
                    "mu_diff": float(row["mu_diff"]),
                    "mu_diff_max": float(row["mu_diff_max"]),
                    "margin_vs_rest": float(row["margin_vs_rest"]),
                    "margin_vs_max_other": float(row["margin_vs_max_other"]),
                    "auroc_ovr": float(row["auroc_ovr"]),
                    **{f"mu_{other_class_name}": float(row[f'mu_{other_class_name}']) for other_class_name in LABEL_CODES},
                }
                for class_name in LABEL_CODES
                for row in baseline_banks[bank_name].per_class_rows[class_name]
            ],
        )
        revised_bank_name = f"retrieval_top{top_n}_per_class_disc"
        write_csv(
            args.output_dir / "outputs" / f"revised_top{top_n}_per_class.csv",
            [
                "bank_name",
                "class_name",
                "concept_id",
                "concept",
                "source_rank_in_filtered_top300",
                "retrieval_rank_in_class",
                "ranking_method",
                "ranking_score",
                "mu_same",
                "mu_diff",
                "mu_diff_max",
                "margin_vs_rest",
                "margin_vs_max_other",
                "auroc_ovr",
                *[f"mu_{class_name}" for class_name in LABEL_CODES],
            ],
            [
                {
                    "bank_name": revised_bank_name,
                    "class_name": class_name,
                    "concept_id": row["concept_id"],
                    "concept": row["concept"],
                    "source_rank_in_filtered_top300": int(row["source_rank_in_filtered_top300"]),
                    "retrieval_rank_in_class": int(row["retrieval_rank_in_class"]),
                    "ranking_method": row["ranking_method"],
                    "ranking_score": float(row["ranking_score"]),
                    "mu_same": float(row["mu_same"]),
                    "mu_diff": float(row["mu_diff"]),
                    "mu_diff_max": float(row["mu_diff_max"]),
                    "margin_vs_rest": float(row["margin_vs_rest"]),
                    "margin_vs_max_other": float(row["margin_vs_max_other"]),
                    "auroc_ovr": float(row["auroc_ovr"]),
                    **{f"mu_{other_class_name}": float(row[f'mu_{other_class_name}']) for other_class_name in LABEL_CODES},
                }
                for class_name in LABEL_CODES
                for row in revised_banks[revised_bank_name].per_class_rows[class_name]
            ],
        )

    class_similarity_rows: list[dict[str, Any]] = []
    if not args.skip_class_similarity:
        class_similarity_rows = compute_class_similarity_matrices(banks, split_payloads, args.output_dir)
        write_csv(
            args.output_dir / "outputs" / "class_similarity_summary.csv",
            [
                "bank_name",
                "split",
                "matrix_csv",
                "mean_diagonal",
                "mean_off_diagonal",
                "diagonal_minus_off_diagonal",
            ],
            class_similarity_rows,
        )

    retrieval_rows: list[dict[str, Any]] = []
    if not args.skip_retrieval:
        retrieval_rows = evaluate_retrieval_on_val(banks, split_payloads)
        write_csv(
            args.output_dir / "outputs" / "retrieval_metrics_val.csv",
            [
                "bank_name",
                "split",
                "class_name",
                "n_queries",
                "n_gallery_concepts_before_dedup",
                "n_gallery_concepts_after_dedup",
                "precision_at_10",
                "precision_at_20",
                "ndcg_at_10",
                "ndcg_at_20",
            ],
            retrieval_rows,
        )

    seed_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    if not args.skip_training:
        seed_rows = maybe_run_training(banks, split_payloads, args.output_dir, args)
        seed_rows = attach_retrieval_macro_metrics(seed_rows, retrieval_rows)
        write_csv(
            args.output_dir / "outputs" / "revised_cycl_seed_results.csv",
            [
                "setting",
                "model_type",
                "bank_name",
                "seed",
                "concept_count_before_dedup",
                "concept_count_after_dedup",
                "num_views",
                "schedule",
                "ranking_method",
                "val_accuracy",
                "val_macro_f1",
                "val_macro_auroc",
                "test_accuracy",
                "test_macro_f1",
                "test_macro_auroc",
                "retrieval_precision_at_10_val_macro",
                "retrieval_precision_at_20_val_macro",
                "retrieval_ndcg_at_10_val_macro",
                "retrieval_ndcg_at_20_val_macro",
                "pair_positive_pair_count",
                "pair_negative_pair_count",
                "pair_positive_weight_mean",
                "pair_negative_weight_mean",
                "image_text_positive_similarity_mean",
                "image_text_negative_similarity_mean",
                "image_text_similarity_gap",
                "output_dir",
            ],
            seed_rows,
        )
        aggregate_rows = aggregate_seed_rows(seed_rows)
        write_csv(
            args.output_dir / "outputs" / "revised_cycl_results_summary.csv",
            [
                "setting",
                "model_type",
                "bank_name",
                "n_seeds",
                "seed_list",
                "concept_count_before_dedup",
                "concept_count_after_dedup",
                "num_views",
                "schedule",
                "ranking_method",
                "val_accuracy_mean",
                "val_accuracy_std",
                "val_macro_f1_mean",
                "val_macro_f1_std",
                "val_macro_auroc_mean",
                "val_macro_auroc_std",
                "test_accuracy_mean",
                "test_accuracy_std",
                "test_macro_f1_mean",
                "test_macro_f1_std",
                "test_macro_auroc_mean",
                "test_macro_auroc_std",
                "retrieval_precision_at_10_val_macro_mean",
                "retrieval_precision_at_10_val_macro_std",
                "retrieval_precision_at_20_val_macro_mean",
                "retrieval_precision_at_20_val_macro_std",
                "retrieval_ndcg_at_10_val_macro_mean",
                "retrieval_ndcg_at_10_val_macro_std",
                "retrieval_ndcg_at_20_val_macro_mean",
                "retrieval_ndcg_at_20_val_macro_std",
                "pair_positive_pair_count_mean",
                "pair_positive_pair_count_std",
                "pair_negative_pair_count_mean",
                "pair_negative_pair_count_std",
                "pair_positive_weight_mean_mean",
                "pair_positive_weight_mean_std",
                "pair_negative_weight_mean_mean",
                "pair_negative_weight_mean_std",
                "image_text_positive_similarity_mean_mean",
                "image_text_positive_similarity_mean_std",
                "image_text_negative_similarity_mean_mean",
                "image_text_negative_similarity_mean_std",
                "image_text_similarity_gap_mean",
                "image_text_similarity_gap_std",
            ],
            aggregate_rows,
        )
        write_csv(
            args.output_dir / "outputs" / "pair_definition_diagnostics.csv",
            [
                "setting",
                "model_type",
                "bank_name",
                "num_views",
                "schedule",
                "ranking_method",
                "n_seeds",
                "pair_positive_pair_count_mean",
                "pair_negative_pair_count_mean",
                "pair_positive_weight_mean_mean",
                "pair_negative_weight_mean_mean",
            ],
            [
                {
                    "setting": row["setting"],
                    "model_type": row["model_type"],
                    "bank_name": row["bank_name"],
                    "num_views": row["num_views"],
                    "schedule": row["schedule"],
                    "ranking_method": row["ranking_method"],
                    "n_seeds": row["n_seeds"],
                    "pair_positive_pair_count_mean": row["pair_positive_pair_count_mean"],
                    "pair_negative_pair_count_mean": row["pair_negative_pair_count_mean"],
                    "pair_positive_weight_mean_mean": row["pair_positive_weight_mean_mean"],
                    "pair_negative_weight_mean_mean": row["pair_negative_weight_mean_mean"],
                }
                for row in aggregate_rows
            ],
        )
        write_csv(
            args.output_dir / "outputs" / "image_text_pairing_diagnostics.csv",
            [
                "setting",
                "model_type",
                "bank_name",
                "num_views",
                "schedule",
                "ranking_method",
                "n_seeds",
                "image_text_positive_similarity_mean_mean",
                "image_text_negative_similarity_mean_mean",
                "image_text_similarity_gap_mean",
            ],
            [
                {
                    "setting": row["setting"],
                    "model_type": row["model_type"],
                    "bank_name": row["bank_name"],
                    "num_views": row["num_views"],
                    "schedule": row["schedule"],
                    "ranking_method": row["ranking_method"],
                    "n_seeds": row["n_seeds"],
                    "image_text_positive_similarity_mean_mean": row["image_text_positive_similarity_mean_mean"],
                    "image_text_negative_similarity_mean_mean": row["image_text_negative_similarity_mean_mean"],
                    "image_text_similarity_gap_mean": row["image_text_similarity_gap_mean"],
                }
                for row in aggregate_rows
            ],
        )

    write_markdown_report(
        output_root=args.output_dir,
        args=args,
        banks=banks,
        retrieval_rows=retrieval_rows,
        aggregate_rows=aggregate_rows,
        class_similarity_rows=class_similarity_rows,
    )

    write_json(
        args.output_dir / "outputs" / "stage_config.json",
        {
            "filtered_top300_dir": str(args.filtered_top300_dir),
            "embeddings_dir": str(args.embeddings_dir),
            "output_dir": str(args.output_dir),
            "device": args.device,
            "seed": args.seed,
            "seeds": parse_seed_list(args.seeds),
            "view_ablation_seeds": parse_seed_list(args.view_ablation_seeds),
            "ranking_method": args.ranking_method,
            "disc_lambda": args.disc_lambda,
            "schedule": args.schedule,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "baseline_epochs": args.baseline_epochs,
            "warmup_epochs": args.warmup_epochs,
            "joint_epochs": args.joint_epochs,
            "run_banks": args.run_banks,
            "run_models": args.run_models,
            "run_view_ablation": args.run_view_ablation,
            "num_views": args.num_views,
            "use_color_aug": args.use_color_aug,
            "use_geom_aug": args.use_geom_aug,
            "use_image_text_pairing": args.use_image_text_pairing,
            "lambda_img_txt": args.lambda_img_txt,
            "tau_img_txt": args.tau_img_txt,
        },
    )
    print(
        f"[done] output_dir={args.output_dir} "
        f"banks={','.join(sorted(banks.keys()))} "
        f"class_similarity_rows={len(class_similarity_rows)} "
        f"retrieval_rows={len(retrieval_rows)} "
        f"seed_rows={len(seed_rows)} "
        f"aggregate_rows={len(aggregate_rows)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
