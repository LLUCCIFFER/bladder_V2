#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from ebtc_project_paths import CONCEPT_EXPERIMENT_OUTPUT_DIR, CYCL_STAGE_OUTPUT_DIR, EXPERIMENT_DIR

if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

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


DEFAULT_FILTERED_TOP300_DIR = Path(
    CONCEPT_EXPERIMENT_OUTPUT_DIR / "final_concept_banks_officialsplit" / "filtered_top300"
)
DEFAULT_EMBEDDINGS_DIR = CONCEPT_EXPERIMENT_OUTPUT_DIR / "embeddings"
DEFAULT_OUTPUT_DIR = CYCL_STAGE_OUTPUT_DIR
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
        description="Second-stage retrieval compression and CyCL experiments on official-split EBTC filtered_top300."
    )
    parser.add_argument("--filtered-top300-dir", type=Path, default=DEFAULT_FILTERED_TOP300_DIR)
    parser.add_argument("--embeddings-dir", type=Path, default=DEFAULT_EMBEDDINGS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--ranking-method", choices=["mean_cosine", "auroc"], default="mean_cosine")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--baseline-epochs", type=int, default=6)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--joint-epochs", type=int, default=6)
    parser.add_argument("--baseline-lr", type=float, default=1e-3)
    parser.add_argument("--main-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-concept", type=float, default=1.0)
    parser.add_argument("--lambda-align", type=float, default=0.3)
    parser.add_argument("--lambda-cycl", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=0.07)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--proj-dim", type=int, default=128)
    parser.add_argument(
        "--run-banks",
        default="filtered_top300,retrieval_top30_per_class,retrieval_top20_per_class,retrieval_top10_per_class",
        help="Comma-separated bank names for CyCL runs.",
    )
    parser.add_argument(
        "--run-models",
        default="cbm_only,cycl",
        help="Comma-separated model variants among image_baseline, cbm_only, cycl.",
    )
    parser.add_argument(
        "--include-image-baseline-top300",
        action="store_true",
        help="Also run the frozen-image linear baseline for filtered_top300.",
    )
    parser.add_argument("--skip-retrieval", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
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
            source_rank = int(row["concept_rank"])
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
            class_similarities = class_embeddings @ candidate.embedding
            rest_similarities = rest_embeddings @ candidate.embedding if rest_embeddings.size else np.array([], dtype=np.float32)
            mean_class_similarity = float(class_similarities.mean())
            mean_rest_similarity = float(rest_similarities.mean()) if rest_similarities.size else float("nan")
            auroc_ovr = float(roc_auc_score(binary_labels, similarities)) if np.unique(binary_labels).size > 1 else float("nan")
            if ranking_method == "mean_cosine":
                ranking_score = mean_class_similarity
            elif ranking_method == "auroc":
                ranking_score = auroc_ovr
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
                "mean_class_similarity": mean_class_similarity,
                "mean_other_similarity": mean_rest_similarity,
                "margin_vs_rest": mean_class_similarity - mean_rest_similarity if np.isfinite(mean_rest_similarity) else float("nan"),
                "auroc_ovr": auroc_ovr,
                "embedding": candidate.embedding,
            }
            class_rank_rows.append(row)

        class_rank_rows.sort(
            key=lambda item: (
                -float(item["ranking_score"]),
                -float(item["mean_class_similarity"]),
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


def build_retrieval_banks(rankings_by_class: dict[str, list[dict[str, Any]]]) -> dict[str, RetrievalBank]:
    banks: dict[str, RetrievalBank] = {}

    full_bank_rows = {class_name: list(rankings_by_class[class_name]) for class_name in LABEL_CODES}
    banks["filtered_top300"] = merge_selected_rows("filtered_top300", full_bank_rows)

    for top_n in TOP_N_CHOICES:
        bank_name = f"retrieval_top{top_n}_per_class"
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
                    "mean_class_similarity": float(row["mean_class_similarity"]),
                    "mean_other_similarity": float(row["mean_other_similarity"]),
                    "margin_vs_rest": float(row["margin_vs_rest"]),
                    "auroc_ovr": float(row["auroc_ovr"]),
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
                "mean_class_similarity",
                "mean_other_similarity",
                "margin_vs_rest",
                "auroc_ovr",
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
        alpha=args.alpha,
        tau=args.tau,
        dropout=args.dropout,
        proj_dim=args.proj_dim,
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
        "output_dir": str(experiment_dir),
    }


def aggregate_seed_rows(training_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, int, int], list[dict[str, Any]]] = {}
    for row in training_rows:
        key = (
            str(row["setting"]),
            str(row["model_type"]),
            str(row["bank_name"]),
            int(row["concept_count_before_dedup"]),
            int(row["concept_count_after_dedup"]),
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
    ]
    aggregate_rows: list[dict[str, Any]] = []
    for key, rows in grouped.items():
        setting, model_type, bank_name, count_before, count_after = key
        summary = {
            "setting": setting,
            "model_type": model_type,
            "bank_name": bank_name,
            "n_seeds": len(rows),
            "concept_count_before_dedup": count_before,
            "concept_count_after_dedup": count_after,
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
    for bank_name in requested_banks:
        bank = banks[bank_name]
        if bank_name == "filtered_top300" and args.include_image_baseline_top300:
            for seed in seeds:
                baseline_dir = experiment_root / bank_name / "image_baseline" / seed_dir_name(seed)
                if args.skip_existing and (baseline_dir / "baseline_metrics.json").exists():
                    payload = load_json(baseline_dir / "baseline_metrics.json")
                    results_rows.append(
                        {
                            "setting": "image_baseline_filtered_top300",
                            "model_type": "image_baseline",
                            "bank_name": bank_name,
                            "seed": seed,
                            "concept_count_before_dedup": bank.total_before_dedup,
                            "concept_count_after_dedup": bank.total_after_dedup,
                            "val_accuracy": float(payload["val"]["accuracy"]),
                            "val_macro_f1": float(payload["val"]["macro_f1"]),
                            "val_macro_auroc": float(payload["val"]["macro_auroc"]) if payload["val"]["macro_auroc"] is not None else float("nan"),
                            "test_accuracy": float(payload["test"]["accuracy"]),
                            "test_macro_f1": float(payload["test"]["macro_f1"]),
                            "test_macro_auroc": float(payload["test"]["macro_auroc"]) if payload["test"]["macro_auroc"] is not None else float("nan"),
                            "output_dir": str(baseline_dir),
                        }
                    )
                else:
                    results_rows.append(
                        run_image_baseline_seed(
                            bank=bank,
                            split_payloads=split_payloads,
                            resolved_device=resolved_device,
                            args=args,
                            experiment_dir=baseline_dir,
                            seed=seed,
                        )
                    )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        for model_type, use_cycl in [("cbm_only", False), ("cycl", True)]:
            if model_type not in requested_models:
                continue
            for seed in seeds:
                experiment_dir = experiment_root / bank_name / model_type / seed_dir_name(seed)
                if args.skip_existing and (experiment_dir / "main_metrics.json").exists():
                    payload = load_json(experiment_dir / "main_metrics.json")
                    results_rows.append(
                        {
                            "setting": f"{model_type}_{bank_name}",
                            "model_type": model_type,
                            "bank_name": bank_name,
                            "seed": seed,
                            "concept_count_before_dedup": bank.total_before_dedup,
                            "concept_count_after_dedup": bank.total_after_dedup,
                            "val_accuracy": float(payload["val"]["accuracy"]),
                            "val_macro_f1": float(payload["val"]["macro_f1"]),
                            "val_macro_auroc": float(payload["val"]["macro_auroc"]) if payload["val"]["macro_auroc"] is not None else float("nan"),
                            "test_accuracy": float(payload["test"]["accuracy"]),
                            "test_macro_f1": float(payload["test"]["macro_f1"]),
                            "test_macro_auroc": float(payload["test"]["macro_auroc"]) if payload["test"]["macro_auroc"] is not None else float("nan"),
                            "output_dir": str(experiment_dir),
                        }
                    )
                else:
                    results_rows.append(
                        run_cbm_seed(
                            bank=bank,
                            split_payloads=split_payloads,
                            image_encoder=image_encoder,
                            preprocess_val=preprocess_val,
                            resolved_device=resolved_device,
                            args=args,
                            experiment_dir=experiment_dir,
                            seed=seed,
                            model_type=model_type,
                            use_cycl=use_cycl,
                        )
                    )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
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


def write_markdown_report(
    output_root: Path,
    args: argparse.Namespace,
    banks: dict[str, RetrievalBank],
    retrieval_rows: list[dict[str, Any]],
    seed_rows: list[dict[str, Any]],
    aggregate_rows: list[dict[str, Any]],
) -> None:
    retrieval_macro = {
        row["bank_name"]: row
        for row in retrieval_rows
        if row["class_name"] == "macro_avg"
    }
    aggregate_lookup = {row["setting"]: row for row in aggregate_rows}

    lines = [
        "# EBTC Retrieval Compression + CyCL Report",
        "",
        "## Setup",
        "",
        "- Official EBTC split from `annotations.csv` only.",
        "- Starting candidate bank: official-split `filtered_top300`.",
        f"- Retrieval ranking method on train: `{args.ranking_method}`.",
        "- Default ranking score used in this stage: mean image-concept cosine similarity over train images from the target class.",
        "- Retrieval relevance on validation: a concept is relevant if its source class includes the image class.",
        f"- Training schedule for this run: baseline_epochs={args.baseline_epochs}, warmup_epochs={args.warmup_epochs}, joint_epochs={args.joint_epochs}.",
        f"- Seeds: `{args.seeds}`.",
        "",
        "## Compressed Banks",
        "",
    ]
    for bank_name in ["filtered_top300", "retrieval_top30_per_class", "retrieval_top20_per_class", "retrieval_top10_per_class"]:
        bank = banks[bank_name]
        lines.append(
            f"- `{bank_name}`: before dedup={bank.total_before_dedup}, after dedup={bank.total_after_dedup}, dedup_removed={bank.dedup_removed}"
        )
    lines.extend(["", "## Retrieval Metrics on Validation", ""])
    for bank_name in ["filtered_top300", "retrieval_top30_per_class", "retrieval_top20_per_class", "retrieval_top10_per_class"]:
        row = retrieval_macro.get(bank_name)
        if row is None:
            continue
        lines.append(
            f"- `{bank_name}`: P@10={row['precision_at_10']:.4f}, P@20={row['precision_at_20']:.4f}, "
            f"nDCG@10={row['ndcg_at_10']:.4f}, nDCG@20={row['ndcg_at_20']:.4f}"
        )

    lines.extend(["", "## Multi-Seed CBM / CyCL Results", ""])
    for setting in [
        "image_baseline_filtered_top300",
        "cbm_only_filtered_top300",
        "cycl_filtered_top300",
        "cbm_only_retrieval_top30_per_class",
        "cycl_retrieval_top30_per_class",
        "cbm_only_retrieval_top20_per_class",
        "cycl_retrieval_top20_per_class",
        "cbm_only_retrieval_top10_per_class",
        "cycl_retrieval_top10_per_class",
    ]:
        row = aggregate_lookup.get(setting)
        if row is None:
            continue
        lines.append(
            f"- `{setting}`: n={row['n_seeds']}, val_macro_f1={row['val_macro_f1_mean']:.4f}±{row['val_macro_f1_std']:.4f}, "
            f"test_accuracy={row['test_accuracy_mean']:.4f}±{row['test_accuracy_std']:.4f}, "
            f"test_macro_f1={row['test_macro_f1_mean']:.4f}±{row['test_macro_f1_std']:.4f}, "
            f"test_macro_auroc={row['test_macro_auroc_mean']:.4f}±{row['test_macro_auroc_std']:.4f}"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- The compressed banks were built with a deliberately simple train-only retrieval score.",
            "- Deduplication is documented but class-specific selections are preserved before merge.",
            "- CyCL class concept profiles are the train-split mean soft concept activations saved in `prototype_matrix.csv` per experiment.",
            "- CBM-no-CyCL uses the same frozen BioMedCLIP, adapter, concept head, classifier, warmup, and concept alignment schedule as CyCL, but sets the contrastive term to zero.",
            "- Retrieval metrics are intermediate diagnostics. Final comparison should focus on validation-selected checkpoints and frozen test performance.",
            f"- Raw per-seed rows: `{output_root / 'outputs' / 'cycl_seed_results.csv'}`.",
            "",
        ]
    )
    (output_root / "outputs" / "cycl_experiment_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    ensure_dir(args.output_dir / "outputs")
    ensure_dir(args.output_dir / "banks")

    set_seed(args.seed)

    split_payloads = load_cached_split_payloads(args.embeddings_dir)
    filtered_top300_candidates = load_filtered_top300(args.filtered_top300_dir, args.embeddings_dir)
    rankings_by_class, ranking_rows = compute_class_retrieval_scores(
        candidates_by_class=filtered_top300_candidates,
        split_payloads=split_payloads,
        ranking_method=args.ranking_method,
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
            "mean_class_similarity",
            "mean_other_similarity",
            "margin_vs_rest",
            "auroc_ovr",
            "retrieval_rank_in_class",
        ],
        ranking_rows,
    )

    banks = build_retrieval_banks(rankings_by_class)
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
                "mean_class_similarity",
                "mean_other_similarity",
                "margin_vs_rest",
                "auroc_ovr",
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
                    "mean_class_similarity": float(row["mean_class_similarity"]),
                    "mean_other_similarity": float(row["mean_other_similarity"]),
                    "margin_vs_rest": float(row["margin_vs_rest"]),
                    "auroc_ovr": float(row["auroc_ovr"]),
                }
                for class_name in LABEL_CODES
                for row in banks[bank_name].per_class_rows[class_name]
            ],
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
            args.output_dir / "outputs" / "cycl_seed_results.csv",
            [
                "setting",
                "model_type",
                "bank_name",
                "seed",
                "concept_count_before_dedup",
                "concept_count_after_dedup",
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
                "output_dir",
            ],
            seed_rows,
        )
        aggregate_rows = aggregate_seed_rows(seed_rows)
        write_csv(
            args.output_dir / "outputs" / "cycl_results_summary.csv",
            [
                "setting",
                "model_type",
                "bank_name",
                "n_seeds",
                "seed_list",
                "concept_count_before_dedup",
                "concept_count_after_dedup",
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
            ],
            aggregate_rows,
        )

    write_markdown_report(
        output_root=args.output_dir,
        args=args,
        banks=banks,
        retrieval_rows=retrieval_rows,
        seed_rows=seed_rows,
        aggregate_rows=aggregate_rows,
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
            "ranking_method": args.ranking_method,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "baseline_epochs": args.baseline_epochs,
            "warmup_epochs": args.warmup_epochs,
            "joint_epochs": args.joint_epochs,
            "run_banks": args.run_banks,
            "run_models": args.run_models,
            "include_image_baseline_top300": args.include_image_baseline_top300,
        },
    )
    print(
        f"[done] output_dir={args.output_dir} "
        f"banks={','.join(sorted(banks.keys()))} "
        f"retrieval_rows={len(retrieval_rows)} "
        f"seed_rows={len(seed_rows)} "
        f"aggregate_rows={len(aggregate_rows)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
