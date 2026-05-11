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

import matplotlib
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from ebtc_project_paths import (
    BACKUP_EMBEDDINGS_DIR,
    EXPERIMENT_DIR,
    HIERARCHICAL_OUTPUT_DIR,
    PROJECT_ROOT,
    RETRIEVAL_TOP10_DISC_BANK_DIR,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))
NEWCODE_DIR = PROJECT_ROOT
if str(NEWCODE_DIR) not in sys.path:
    sys.path.insert(0, str(NEWCODE_DIR))

import etbc_wli_train_cbm_cycl as train_mod  # noqa: E402
from etbc_wli_train_cbm_cycl import (  # noqa: E402
    build_loaders,
    compute_class_weights,
    encode_frozen_image_batch,
    ensure_dir,
    group_records_by_split,
    load_biomedclip,
    save_confusion_figure,
    save_prediction_records,
    save_soft_concept_artifacts,
    select_case_examples,
    set_label_space,
    set_seed,
    train_main_model,
    write_csv,
    write_json,
)
from ebtc_cycl_retrieval_stage_revised import (  # noqa: E402
    DEFAULT_EMBEDDINGS_DIR,
    ensure_official_split_embedding_cache,
    l2_normalize,
    load_cached_split_payloads,
    read_csv_rows,
)


DEFAULT_BANK_DIR = RETRIEVAL_TOP10_DISC_BANK_DIR
DEFAULT_BACKUP_EMBEDDINGS_DIR = BACKUP_EMBEDDINGS_DIR
DEFAULT_OUTPUT_DIR = HIERARCHICAL_OUTPUT_DIR
ORIGINAL_CLASS_MAP = {code: name for code, name in zip(train_mod.LABEL_CODES, train_mod.CLASS_NAMES)}
ORIGINAL_LABEL_CODES = list(ORIGINAL_CLASS_MAP.keys())
EPS = 1e-6


@dataclass
class ExportedBank:
    name: str
    per_class_rows: dict[str, list[dict[str, Any]]]


@dataclass
class HierarchicalTask:
    name: str
    class_map: dict[str, str]
    original_to_task: dict[str, str]

    @property
    def label_codes(self) -> list[str]:
        return list(self.class_map.keys())


@dataclass
class TaskBank:
    name: str
    per_class_rows: dict[str, list[dict[str, Any]]]
    merged_rows: list[dict[str, Any]]
    merged_concepts: list[str]
    merged_embeddings: np.ndarray
    total_before_dedup: int
    total_after_dedup: int
    dedup_removed: int


class InferenceImageDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], eval_transform: Any) -> None:
        self.rows = rows
        self.eval_transform = eval_transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str, str, str]:
        row = self.rows[index]
        with Image.open(row["image_path"]) as image:
            tensor = self.eval_transform(image.convert("RGB"))
        return (
            tensor,
            str(row["image_path"]),
            str(row.get("annotation_image_name") or Path(str(row["image_path"])).name),
            str(row["class_name"]),
        )


def save_float_heatmap(
    matrix: np.ndarray,
    path: Path,
    x_labels: list[str],
    y_labels: list[str],
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    image = ax.imshow(matrix, cmap="viridis")
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hierarchical EBTC CBM/CyCL experiments.")
    parser.add_argument("--bank-dir", type=Path, default=DEFAULT_BANK_DIR)
    parser.add_argument("--embeddings-dir", type=Path, default=DEFAULT_EMBEDDINGS_DIR)
    parser.add_argument("--backup-embeddings-dir", type=Path, default=DEFAULT_BACKUP_EMBEDDINGS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--schedule", choices=["short_debug", "improved_main"], default="short_debug")
    parser.add_argument("--warmup-epochs", type=int, default=None)
    parser.add_argument("--joint-epochs", type=int, default=None)
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
    parser.add_argument("--run-models", default="cbm_only,cycl")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def resolve_schedule(args: argparse.Namespace) -> None:
    if args.schedule == "short_debug":
        warmup_epochs = 3
        joint_epochs = 6
    elif args.schedule == "improved_main":
        warmup_epochs = 5
        joint_epochs = 15
    else:
        raise ValueError(f"Unsupported schedule: {args.schedule}")
    if args.warmup_epochs is None:
        args.warmup_epochs = warmup_epochs
    if args.joint_epochs is None:
        args.joint_epochs = joint_epochs


def load_exported_bank(bank_dir: Path) -> ExportedBank:
    embeddings_payload = np.load(bank_dir / "filtered_concept_text_embeddings.npz", allow_pickle=True)
    concept_names = [str(item) for item in embeddings_payload["concepts"].tolist()]
    concept_embeddings = l2_normalize(embeddings_payload["concept_embeddings"].astype(np.float32))
    concept_to_embedding = {
        concept: concept_embeddings[index].astype(np.float32)
        for index, concept in enumerate(concept_names)
    }

    per_class_rows: dict[str, list[dict[str, Any]]] = {}
    for class_name in ORIGINAL_LABEL_CODES:
        rows = []
        for row in read_csv_rows(bank_dir / f"{class_name}.csv"):
            concept = str(row["concept"])
            rows.append(
                {
                    **row,
                    "class_name": class_name,
                    "concept": concept,
                    "concept_id": str(row["concept_id"]),
                    "source_rank_in_filtered_top300": int(row["source_rank_in_filtered_top300"]),
                    "retrieval_rank_in_class": int(row["retrieval_rank_in_class"]),
                    "ranking_score": float(row["ranking_score"]),
                    "embedding": concept_to_embedding[concept],
                }
            )
        per_class_rows[class_name] = rows
    return ExportedBank(name=bank_dir.name, per_class_rows=per_class_rows)


def build_task_bank(base_bank: ExportedBank, task: HierarchicalTask) -> TaskBank:
    per_class_rows: dict[str, list[dict[str, Any]]] = {code: [] for code in task.label_codes}
    for original_class, rows in base_bank.per_class_rows.items():
        if original_class not in task.original_to_task:
            continue
        task_class = task.original_to_task[original_class]
        for row in rows:
            copied = dict(row)
            copied["task_class"] = task_class
            copied["source_original_class"] = original_class
            per_class_rows[task_class].append(copied)

    for task_class in task.label_codes:
        per_class_rows[task_class].sort(
            key=lambda item: (
                -float(item["ranking_score"]),
                int(item["source_rank_in_filtered_top300"]),
                str(item["concept"]),
            )
        )
        for rank_index, row in enumerate(per_class_rows[task_class], start=1):
            row["task_rank_in_class"] = rank_index

    merged_rows: list[dict[str, Any]] = []
    concept_to_index: dict[str, int] = {}
    before_count = sum(len(rows) for rows in per_class_rows.values())
    for task_class in task.label_codes:
        for row in per_class_rows[task_class]:
            concept = str(row["concept"])
            if concept not in concept_to_index:
                concept_to_index[concept] = len(merged_rows)
                merged_rows.append(
                    {
                        "merged_index": len(merged_rows),
                        "concept": concept,
                        "embedding": row["embedding"],
                        "primary_task_class": task_class,
                        "source_task_classes": [task_class],
                        "source_original_classes": [str(row["source_original_class"])],
                        "source_concept_ids": [str(row["concept_id"])],
                    }
                )
            else:
                merged_row = merged_rows[concept_to_index[concept]]
                if task_class not in merged_row["source_task_classes"]:
                    merged_row["source_task_classes"].append(task_class)
                merged_row["source_original_classes"].append(str(row["source_original_class"]))
                merged_row["source_concept_ids"].append(str(row["concept_id"]))

    merged_concepts = [str(row["concept"]) for row in merged_rows]
    merged_embeddings = l2_normalize(np.stack([row["embedding"] for row in merged_rows], axis=0).astype(np.float32))
    for row in merged_rows:
        row["source_task_class_count"] = len(row["source_task_classes"])

    return TaskBank(
        name=f"{base_bank.name}_{task.name}",
        per_class_rows=per_class_rows,
        merged_rows=merged_rows,
        merged_concepts=merged_concepts,
        merged_embeddings=merged_embeddings,
        total_before_dedup=before_count,
        total_after_dedup=len(merged_rows),
        dedup_removed=before_count - len(merged_rows),
    )


def task_bank_concept_membership(task: HierarchicalTask, bank: TaskBank) -> np.ndarray:
    membership = np.zeros((len(task.label_codes), bank.total_after_dedup), dtype=np.float32)
    task_to_index = {code: idx for idx, code in enumerate(task.label_codes)}
    for concept_index, row in enumerate(bank.merged_rows):
        for task_class in row["source_task_classes"]:
            membership[task_to_index[str(task_class)], concept_index] = 1.0
    return membership


def build_task_soft_targets(
    task: HierarchicalTask,
    bank: TaskBank,
    split_payloads: dict[str, dict[str, Any]],
    output_dir: Path,
) -> tuple[dict[str, list[dict[str, Any]]], np.ndarray]:
    ensure_dir(output_dir)
    set_label_space(task.class_map)

    ordered_manifest: list[dict[str, str]] = []
    embeddings_list: list[np.ndarray] = []
    for split_name in ["train", "val", "test"]:
        split_rows = split_payloads[split_name]["rows"]
        split_embeddings = split_payloads[split_name]["embeddings"]
        for row, embedding in zip(split_rows, split_embeddings, strict=True):
            original_class = str(row["class_name"])
            if original_class not in task.original_to_task:
                continue
            ordered_manifest.append(
                {
                    "image_path": str(row["image_path"]),
                    "image_name": str(row.get("annotation_image_name") or Path(str(row["image_path"])).name),
                    "split": split_name,
                    "class_code": task.original_to_task[original_class],
                    "original_class_code": original_class,
                }
            )
            embeddings_list.append(embedding.astype(np.float32))

    image_embeddings = np.stack(embeddings_list, axis=0).astype(np.float32)
    similarity_matrix = image_embeddings @ bank.merged_embeddings.T
    split_array = np.array([row["split"] for row in ordered_manifest], dtype=object)
    label_array = np.array([row["class_code"] for row in ordered_manifest], dtype=object)
    train_mask = split_array == "train"

    mu = similarity_matrix[train_mask].mean(axis=0).astype(np.float32)
    sigma = similarity_matrix[train_mask].std(axis=0).astype(np.float32)
    soft_concepts = 1.0 / (1.0 + np.exp(-(similarity_matrix - mu) / (sigma + EPS)))
    soft_concepts = soft_concepts.astype(np.float32)

    prototypes = []
    for class_code in task.label_codes:
        class_mask = train_mask & (label_array == class_code)
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
    return split_rows, prototype_matrix


def make_train_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        warmup_epochs=args.warmup_epochs,
        joint_epochs=args.joint_epochs,
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


def compute_task_similarity_matrices(
    task: HierarchicalTask,
    bank: TaskBank,
    split_payloads: dict[str, dict[str, Any]],
    outputs_dir: Path,
) -> list[dict[str, Any]]:
    ensure_dir(outputs_dir)
    task_to_index = {code: idx for idx, code in enumerate(task.label_codes)}
    per_class_embeddings = {
        class_code: l2_normalize(
            np.stack([row["embedding"] for row in bank.per_class_rows[class_code]], axis=0).astype(np.float32)
        )
        for class_code in task.label_codes
    }
    summary_rows: list[dict[str, Any]] = []
    for split_name in ["train", "val"]:
        split_rows = split_payloads[split_name]["rows"]
        split_embeddings = split_payloads[split_name]["embeddings"]
        matrix = np.zeros((len(task.label_codes), len(task.label_codes)), dtype=np.float32)
        for task_class in task.label_codes:
            image_mask = np.array(
                [task.original_to_task.get(str(row["class_name"])) == task_class for row in split_rows],
                dtype=bool,
            )
            class_image_embeddings = split_embeddings[image_mask]
            for concept_class in task.label_codes:
                matrix[task_to_index[task_class], task_to_index[concept_class]] = float(
                    (class_image_embeddings @ per_class_embeddings[concept_class].T).mean()
                )

        csv_rows = []
        for row_index, image_class in enumerate(task.label_codes):
            row = {"image_class": image_class}
            for col_index, concept_class in enumerate(task.label_codes):
                row[concept_class] = float(matrix[row_index, col_index])
            csv_rows.append(row)

        csv_path = outputs_dir / f"{task.name}_class_similarity_{split_name}.csv"
        write_csv(csv_path, ["image_class", *task.label_codes], csv_rows)
        save_float_heatmap(
            matrix=matrix,
            path=outputs_dir / f"{task.name}_class_similarity_{split_name}.png",
            x_labels=task.label_codes,
            y_labels=task.label_codes,
            title=f"{task.name} image-text similarity ({split_name})",
        )
        diagonal = np.diag(matrix)
        off_diag = matrix[~np.eye(matrix.shape[0], dtype=bool)]
        summary_rows.append(
            {
                "task_name": task.name,
                "split": split_name,
                "matrix_csv": str(csv_path),
                "mean_diagonal": float(diagonal.mean()),
                "mean_off_diagonal": float(off_diag.mean()) if off_diag.size else float("nan"),
                "diagonal_minus_off_diagonal": float(diagonal.mean() - off_diag.mean()) if off_diag.size else float("nan"),
            }
        )
    return summary_rows


def predict_rows(
    image_encoder: Any,
    model: Any,
    rows: list[dict[str, Any]],
    eval_transform: Any,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    dataset = InferenceImageDataset(rows, eval_transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.startswith("cuda"),
    )
    probabilities_chunks: list[np.ndarray] = []
    metadata_rows: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for images, image_paths, image_names, class_names in loader:
            features = encode_frozen_image_batch(image_encoder, images, device)
            outputs = model.forward_from_features(features)
            probabilities = torch.softmax(outputs["logits"], dim=-1).cpu().numpy()
            probabilities_chunks.append(probabilities)
            for image_path, image_name, class_name in zip(image_paths, image_names, class_names, strict=True):
                metadata_rows.append(
                    {
                        "image_path": str(image_path),
                        "image_name": str(image_name),
                        "true_original_class": str(class_name),
                    }
                )
    return np.concatenate(probabilities_chunks, axis=0), metadata_rows


def compute_multiclass_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    class_codes: list[str],
) -> dict[str, Any]:
    predictions = probabilities.argmax(axis=1)
    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "macro_recall": float(recall_score(labels, predictions, average="macro", zero_division=0)),
        "per_class_recall": recall_score(
            labels,
            predictions,
            labels=np.arange(len(class_codes)),
            average=None,
            zero_division=0,
        ).astype(float).tolist(),
        "confusion_matrix": confusion_matrix(labels, predictions).astype(int).tolist(),
    }
    try:
        one_hot = np.eye(len(class_codes), dtype=np.float32)[labels]
        metrics["macro_auroc"] = float(
            roc_auc_score(one_hot, probabilities, average="macro", multi_class="ovr")
        )
    except Exception:
        metrics["macro_auroc"] = None
    return metrics


def run_task(
    task: HierarchicalTask,
    bank: TaskBank,
    split_payloads: dict[str, dict[str, Any]],
    image_encoder: Any,
    preprocess_val: Any,
    device: str,
    args: argparse.Namespace,
    model_type: str,
    use_cycl: bool,
    experiment_dir: Path,
) -> tuple[dict[str, Any], Any]:
    ensure_dir(experiment_dir)
    if args.skip_existing and (experiment_dir / "main_metrics.json").exists():
        payload = json.loads((experiment_dir / "main_metrics.json").read_text(encoding="utf-8"))
        checkpoint_path = experiment_dir / "main_best.pt"
        set_label_space(task.class_map)
        model = train_mod.BioMedClipCBMCyCL(
            embedding_dim=int(np.load(experiment_dir / "soft_concepts_all.npz", allow_pickle=True)["image_embeddings"].shape[1]),
            num_concepts=bank.total_after_dedup,
            num_classes=len(task.label_codes),
            dropout=args.dropout,
            proj_dim=args.proj_dim,
        ).to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        return payload["test"], model

    set_label_space(task.class_map)
    split_rows, prototype_matrix = build_task_soft_targets(task, bank, split_payloads, experiment_dir)
    class_weights = compute_class_weights(split_rows["train"], device)
    train_loader, val_loader, test_loader = build_loaders(
        split_rows=split_rows,
        eval_transform=preprocess_val,
        prototype_matrix=prototype_matrix,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        num_views=args.num_views,
        use_color_aug=args.use_color_aug,
        use_geom_aug=args.use_geom_aug,
    )
    embedding_dim = int(split_rows["train"][0]["embedding"].shape[0])
    concept_membership = task_bank_concept_membership(task, bank)
    set_seed(args.seed)
    main_metrics, main_records, model = train_main_model(
        image_encoder=image_encoder,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        output_dir=experiment_dir,
        device=device,
        embedding_dim=embedding_dim,
        num_concepts=bank.total_after_dedup,
        class_weights=class_weights,
        prototype_matrix=prototype_matrix,
        args=make_train_args(args),
        bank_text_embeddings=bank.merged_embeddings,
        concept_class_membership=concept_membership,
        use_cycl=use_cycl,
    )
    save_confusion_figure(
        np.array(main_metrics["confusion_matrix"], dtype=np.int64),
        experiment_dir / "main_confusion_matrix.png",
        f"{model_type} Confusion Matrix ({task.name})",
    )
    classifier_weight = model.classifier.weight.detach().cpu().numpy()
    save_prediction_records(
        experiment_dir / "main_test_predictions.csv",
        main_records,
        concepts=bank.merged_concepts,
        classifier_weight=classifier_weight,
    )
    return main_metrics, model


def build_final_probabilities(
    root_probs: np.ndarray,
    hl_probs: np.ndarray,
    nn_probs: np.ndarray,
    root_order: list[str],
    hl_order: list[str],
    nn_order: list[str],
) -> np.ndarray:
    root_index = {code: idx for idx, code in enumerate(root_order)}
    hl_index = {code: idx for idx, code in enumerate(hl_order)}
    nn_index = {code: idx for idx, code in enumerate(nn_order)}
    final = np.zeros((root_probs.shape[0], len(ORIGINAL_LABEL_CODES)), dtype=np.float32)
    final[:, ORIGINAL_LABEL_CODES.index("HGC")] = root_probs[:, root_index["TUMOR"]] * hl_probs[:, hl_index["HGC"]]
    final[:, ORIGINAL_LABEL_CODES.index("LGC")] = root_probs[:, root_index["TUMOR"]] * hl_probs[:, hl_index["LGC"]]
    final[:, ORIGINAL_LABEL_CODES.index("NTL")] = root_probs[:, root_index["BENIGN"]] * nn_probs[:, nn_index["NTL"]]
    final[:, ORIGINAL_LABEL_CODES.index("NST")] = root_probs[:, root_index["BENIGN"]] * nn_probs[:, nn_index["NST"]]
    final = final / np.clip(final.sum(axis=1, keepdims=True), EPS, None)
    return final


def save_hierarchical_route_heatmap(
    matrix: np.ndarray,
    path: Path,
    y_labels: list[str],
    x_labels: list[str],
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            ax.text(col_index, row_index, int(matrix[row_index, col_index]), ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_hierarchical_report(
    output_root: Path,
    task_similarity_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Hierarchical EBTC Revised Pipeline",
        "",
        "## Scheme",
        "",
        "1. Root binary: `HGC+LGC` vs `NTL+NST`",
        "2. High-risk branch: `HGC` vs `LGC`",
        "3. Low-risk branch: `NTL` vs `NST`",
        "",
        "## Task-Level Image-Text Similarity",
        "",
    ]
    for row in task_similarity_rows:
        lines.append(
            f"- `{row['task_name']}` `{row['split']}`: diag={row['mean_diagonal']:.4f}, "
            f"offdiag={row['mean_off_diagonal']:.4f}, margin={row['diagonal_minus_off_diagonal']:.4f}"
        )
    lines.extend(["", "## Final Hierarchical Results", ""])
    for row in summary_rows:
        auroc_text = (
            f"{float(row['final_test_macro_auroc']):.4f}"
            if np.isfinite(float(row["final_test_macro_auroc"]))
            else "-"
        )
        lines.append(
            f"- `{row['model_type']}`: root_acc={row['root_test_accuracy']:.4f}, "
            f"hl_acc={row['hl_test_accuracy']:.4f}, nn_acc={row['nn_test_accuracy']:.4f}, "
            f"final_test_acc={row['final_test_accuracy']:.4f}, final_test_macro_f1={row['final_test_macro_f1']:.4f}, "
            f"final_test_macro_auroc={auroc_text}"
        )
    (output_root / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    resolve_schedule(args)
    ensure_dir(args.output_dir)
    ensure_dir(args.output_dir / "outputs")
    ensure_official_split_embedding_cache(args.embeddings_dir, args.backup_embeddings_dir)
    split_payloads = load_cached_split_payloads(args.embeddings_dir)
    base_bank = load_exported_bank(args.bank_dir)

    tasks = {
        "root": HierarchicalTask(
            name="root",
            class_map={"TUMOR": "HGC + LGC", "BENIGN": "NTL + NST"},
            original_to_task={"HGC": "TUMOR", "LGC": "TUMOR", "NTL": "BENIGN", "NST": "BENIGN"},
        ),
        "branch_hl": HierarchicalTask(
            name="branch_hl",
            class_map={"HGC": ORIGINAL_CLASS_MAP["HGC"], "LGC": ORIGINAL_CLASS_MAP["LGC"]},
            original_to_task={"HGC": "HGC", "LGC": "LGC"},
        ),
        "branch_nn": HierarchicalTask(
            name="branch_nn",
            class_map={"NTL": ORIGINAL_CLASS_MAP["NTL"], "NST": ORIGINAL_CLASS_MAP["NST"]},
            original_to_task={"NTL": "NTL", "NST": "NST"},
        ),
    }

    task_banks = {name: build_task_bank(base_bank, task) for name, task in tasks.items()}
    task_similarity_rows: list[dict[str, Any]] = []
    for task_name, task in tasks.items():
        task_similarity_rows.extend(
            compute_task_similarity_matrices(
                task=task,
                bank=task_banks[task_name],
                split_payloads=split_payloads,
                outputs_dir=args.output_dir / "outputs",
            )
        )
    write_csv(
        args.output_dir / "outputs" / "hierarchical_task_similarity_summary.csv",
        ["task_name", "split", "matrix_csv", "mean_diagonal", "mean_off_diagonal", "diagonal_minus_off_diagonal"],
        task_similarity_rows,
    )

    image_encoder, _, preprocess_val, device = load_biomedclip(args.device)
    for parameter in image_encoder.parameters():
        parameter.requires_grad = False
    image_encoder.eval()

    requested_models = [item.strip() for item in args.run_models.split(",") if item.strip()]
    summary_rows: list[dict[str, Any]] = []

    full_val_rows = split_payloads["val"]["rows"]
    full_test_rows = split_payloads["test"]["rows"]
    original_label_to_index = {code: idx for idx, code in enumerate(ORIGINAL_LABEL_CODES)}
    true_val_labels = np.array([original_label_to_index[str(row["class_name"])] for row in full_val_rows], dtype=np.int64)
    true_test_labels = np.array([original_label_to_index[str(row["class_name"])] for row in full_test_rows], dtype=np.int64)

    for model_type, use_cycl in [("cbm_only", False), ("cycl", True)]:
        if model_type not in requested_models:
            continue

        model_root, model_hl, model_nn = None, None, None
        task_metrics: dict[str, dict[str, Any]] = {}
        for task_name in ["root", "branch_hl", "branch_nn"]:
            task = tasks[task_name]
            experiment_dir = (
                args.output_dir
                / "experiments"
                / model_type
                / task_name
                / f"views_{args.num_views}_{args.schedule}"
                / f"seed_{args.seed:03d}"
            )
            metrics, model = run_task(
                task=task,
                bank=task_banks[task_name],
                split_payloads=split_payloads,
                image_encoder=image_encoder,
                preprocess_val=preprocess_val,
                device=device,
                args=args,
                model_type=model_type,
                use_cycl=use_cycl,
                experiment_dir=experiment_dir,
            )
            task_metrics[task_name] = metrics
            if task_name == "root":
                model_root = model
            elif task_name == "branch_hl":
                model_hl = model
            else:
                model_nn = model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        assert model_root is not None and model_hl is not None and model_nn is not None

        root_val_probs, _ = predict_rows(image_encoder, model_root, full_val_rows, preprocess_val, device, args.batch_size)
        root_test_probs, test_meta = predict_rows(image_encoder, model_root, full_test_rows, preprocess_val, device, args.batch_size)
        hl_val_probs, _ = predict_rows(image_encoder, model_hl, full_val_rows, preprocess_val, device, args.batch_size)
        hl_test_probs, _ = predict_rows(image_encoder, model_hl, full_test_rows, preprocess_val, device, args.batch_size)
        nn_val_probs, _ = predict_rows(image_encoder, model_nn, full_val_rows, preprocess_val, device, args.batch_size)
        nn_test_probs, _ = predict_rows(image_encoder, model_nn, full_test_rows, preprocess_val, device, args.batch_size)

        final_val_probs = build_final_probabilities(
            root_probs=root_val_probs,
            hl_probs=hl_val_probs,
            nn_probs=nn_val_probs,
            root_order=tasks["root"].label_codes,
            hl_order=tasks["branch_hl"].label_codes,
            nn_order=tasks["branch_nn"].label_codes,
        )
        final_test_probs = build_final_probabilities(
            root_probs=root_test_probs,
            hl_probs=hl_test_probs,
            nn_probs=nn_test_probs,
            root_order=tasks["root"].label_codes,
            hl_order=tasks["branch_hl"].label_codes,
            nn_order=tasks["branch_nn"].label_codes,
        )

        final_val_metrics = compute_multiclass_metrics(true_val_labels, final_val_probs, ORIGINAL_LABEL_CODES)
        final_test_metrics = compute_multiclass_metrics(true_test_labels, final_test_probs, ORIGINAL_LABEL_CODES)

        final_output_dir = args.output_dir / "experiments" / model_type / "hierarchical_final" / f"views_{args.num_views}_{args.schedule}" / f"seed_{args.seed:03d}"
        ensure_dir(final_output_dir)

        final_cm = np.array(final_test_metrics["confusion_matrix"], dtype=np.int64)
        save_hierarchical_route_heatmap(
            final_cm,
            final_output_dir / "hierarchical_final_confusion_matrix.png",
            [ORIGINAL_CLASS_MAP[code] for code in ORIGINAL_LABEL_CODES],
            [ORIGINAL_CLASS_MAP[code] for code in ORIGINAL_LABEL_CODES],
            f"Hierarchical Final Confusion ({model_type})",
        )

        root_true_groups = np.array([tasks["root"].original_to_task[str(row["class_name"])] for row in full_test_rows], dtype=object)
        root_pred_groups = np.array(
            [tasks["root"].label_codes[index] for index in root_test_probs.argmax(axis=1)],
            dtype=object,
        )
        group_codes = tasks["root"].label_codes
        group_to_index = {code: idx for idx, code in enumerate(group_codes)}
        route_cm = confusion_matrix(
            np.array([group_to_index[item] for item in root_true_groups], dtype=np.int64),
            np.array([group_to_index[item] for item in root_pred_groups], dtype=np.int64),
        ).astype(np.int64)
        save_hierarchical_route_heatmap(
            route_cm,
            final_output_dir / "hierarchical_root_route_confusion_matrix.png",
            [tasks["root"].class_map[code] for code in group_codes],
            [tasks["root"].class_map[code] for code in group_codes],
            f"Hierarchical Root Route ({model_type})",
        )
        write_csv(
            final_output_dir / "hierarchical_root_route_confusion_matrix.csv",
            ["true_group", *group_codes],
            [
                {"true_group": group_codes[row_index], **{group_codes[col_index]: int(route_cm[row_index, col_index]) for col_index in range(len(group_codes))}}
                for row_index in range(len(group_codes))
            ],
        )

        prediction_rows = []
        for row_index, meta in enumerate(test_meta):
            row = {
                "image_name": meta["image_name"],
                "image_path": meta["image_path"],
                "true_original_class": meta["true_original_class"],
                "root_pred_group": root_pred_groups[row_index],
                "root_prob_tumor": float(root_test_probs[row_index, tasks["root"].label_codes.index("TUMOR")]),
                "root_prob_benign": float(root_test_probs[row_index, tasks["root"].label_codes.index("BENIGN")]),
                "branch_hl_prob_HGC": float(hl_test_probs[row_index, tasks["branch_hl"].label_codes.index("HGC")]),
                "branch_hl_prob_LGC": float(hl_test_probs[row_index, tasks["branch_hl"].label_codes.index("LGC")]),
                "branch_nn_prob_NTL": float(nn_test_probs[row_index, tasks["branch_nn"].label_codes.index("NTL")]),
                "branch_nn_prob_NST": float(nn_test_probs[row_index, tasks["branch_nn"].label_codes.index("NST")]),
            }
            final_pred_index = int(final_test_probs[row_index].argmax())
            row["final_pred_class"] = ORIGINAL_LABEL_CODES[final_pred_index]
            row["final_pred_confidence"] = float(final_test_probs[row_index, final_pred_index])
            for class_code in ORIGINAL_LABEL_CODES:
                row[f"final_prob_{class_code}"] = float(final_test_probs[row_index, ORIGINAL_LABEL_CODES.index(class_code)])
            prediction_rows.append(row)
        write_csv(final_output_dir / "hierarchical_test_predictions.csv", list(prediction_rows[0].keys()), prediction_rows)

        write_json(
            final_output_dir / "hierarchical_metrics.json",
            {
                "root_test": task_metrics["root"],
                "branch_hl_test": task_metrics["branch_hl"],
                "branch_nn_test": task_metrics["branch_nn"],
                "final_val": final_val_metrics,
                "final_test": final_test_metrics,
            },
        )

        summary_rows.append(
            {
                "model_type": model_type,
                "bank_name": args.bank_dir.name,
                "schedule": args.schedule,
                "num_views": args.num_views,
                "seed": args.seed,
                "root_test_accuracy": float(task_metrics["root"]["accuracy"]),
                "root_test_macro_f1": float(task_metrics["root"]["macro_f1"]),
                "hl_test_accuracy": float(task_metrics["branch_hl"]["accuracy"]),
                "hl_test_macro_f1": float(task_metrics["branch_hl"]["macro_f1"]),
                "nn_test_accuracy": float(task_metrics["branch_nn"]["accuracy"]),
                "nn_test_macro_f1": float(task_metrics["branch_nn"]["macro_f1"]),
                "final_val_accuracy": float(final_val_metrics["accuracy"]),
                "final_val_macro_f1": float(final_val_metrics["macro_f1"]),
                "final_val_macro_auroc": float(final_val_metrics["macro_auroc"]) if final_val_metrics["macro_auroc"] is not None else float("nan"),
                "final_test_accuracy": float(final_test_metrics["accuracy"]),
                "final_test_macro_f1": float(final_test_metrics["macro_f1"]),
                "final_test_macro_auroc": float(final_test_metrics["macro_auroc"]) if final_test_metrics["macro_auroc"] is not None else float("nan"),
                "final_output_dir": str(final_output_dir),
            }
        )

    write_csv(
        args.output_dir / "outputs" / "hierarchical_results_summary.csv",
        [
            "model_type",
            "bank_name",
            "schedule",
            "num_views",
            "seed",
            "root_test_accuracy",
            "root_test_macro_f1",
            "hl_test_accuracy",
            "hl_test_macro_f1",
            "nn_test_accuracy",
            "nn_test_macro_f1",
            "final_val_accuracy",
            "final_val_macro_f1",
            "final_val_macro_auroc",
            "final_test_accuracy",
            "final_test_macro_f1",
            "final_test_macro_auroc",
            "final_output_dir",
        ],
        summary_rows,
    )
    write_hierarchical_report(args.output_dir, task_similarity_rows, summary_rows)
    set_label_space(ORIGINAL_CLASS_MAP)
    print(
        f"[done] output_dir={args.output_dir} model_rows={len(summary_rows)} task_similarity_rows={len(task_similarity_rows)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
