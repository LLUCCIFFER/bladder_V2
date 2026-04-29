#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.utils import save_image


LABEL_CODES = ["HGC", "LGC", "NTL", "NST"]
CLASS_TO_INDEX = {class_name: index for index, class_name in enumerate(LABEL_CODES)}
EPS = 1e-8


@dataclass
class FixedConceptBankV2:
    name: str
    bank_dir: Path
    per_class_rows: dict[str, list[dict[str, Any]]]
    per_class_embeddings: dict[str, np.ndarray]
    merged_concepts: list[str]
    merged_embeddings: np.ndarray
    metadata_rows: list[dict[str, Any]]
    concept_class_membership: np.ndarray
    class_label_rule: str


@dataclass
class SplitEmbeddingsV2:
    rows: list[dict[str, str]]
    embeddings: np.ndarray


@dataclass
class PreparedBankCacheV2:
    bank: FixedConceptBankV2
    split_rows: dict[str, list[dict[str, Any]]]
    prototype_matrix: np.ndarray
    mu: np.ndarray
    sigma: np.ndarray
    soft_targets_all: np.ndarray
    ordered_manifest: list[dict[str, Any]]


class MultiViewDatasetV2(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        prototype_matrix: np.ndarray,
        view_transforms: list[Any],
    ) -> None:
        self.rows = rows
        self.prototype_matrix = prototype_matrix
        self.view_transforms = view_transforms

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
        row = self.rows[index]
        image = Image.open(str(row["image_path"])).convert("RGB")
        views = torch.stack([transform(image) for transform in self.view_transforms], dim=0)
        label = torch.tensor(int(row["label_index"]), dtype=torch.long)
        soft_target = torch.from_numpy(row["soft_concepts"].astype(np.float32))
        prototype = torch.from_numpy(self.prototype_matrix[int(row["label_index"])].astype(np.float32))
        return views, label, soft_target, prototype, str(row["image_path"])


def select_balanced_rows(rows: list[dict[str, Any]], max_rows: int) -> list[dict[str, Any]]:
    by_class: dict[int, list[dict[str, Any]]] = {index: [] for index in range(len(LABEL_CODES))}
    for row in rows:
        by_class[int(row["label_index"])].append(row)
    selected: list[dict[str, Any]] = []
    cursor = 0
    while len(selected) < max_rows:
        added = False
        for class_index in range(len(LABEL_CODES)):
            class_rows = by_class[class_index]
            if cursor < len(class_rows):
                selected.append(class_rows[cursor])
                added = True
                if len(selected) >= max_rows:
                    break
        if not added:
            break
        cursor += 1
    return selected


class TextConceptAdapterV2(nn.Module):
    def __init__(self, embedding_dim: int, proj_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        hidden_dim = max(embedding_dim // 2, proj_dim)
        self.adapter = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, text_embeddings: torch.Tensor) -> torch.Tensor:
        hidden = text_embeddings + self.adapter(text_embeddings)
        return F.normalize(self.projection(hidden), dim=-1)


class CBMCyCLV2(nn.Module):
    """V2 CBM/CyCL head. Classification remains strictly concept-bottleneck based."""

    def __init__(
        self,
        embedding_dim: int,
        num_concepts: int,
        num_classes: int = 4,
        dropout: float = 0.2,
        proj_dim: int = 128,
    ) -> None:
        super().__init__()
        hidden_dim = embedding_dim // 2
        self.image_adapter = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.concept_head = nn.Linear(embedding_dim, num_concepts)
        self.classifier = nn.Linear(num_concepts, num_classes)
        self.image_projection = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, proj_dim),
        )
        self.text_adapter = TextConceptAdapterV2(embedding_dim=embedding_dim, proj_dim=proj_dim, dropout=dropout)

    def forward_image_features(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = features + self.image_adapter(features)
        concept_scores = torch.sigmoid(self.concept_head(hidden))
        logits = self.classifier(concept_scores)
        image_projection = F.normalize(self.image_projection(hidden), dim=-1)
        return {
            "hidden": hidden,
            "concept_scores": concept_scores,
            "logits": logits,
            "image_projection": image_projection,
        }

    def forward_text_embeddings(self, text_embeddings: torch.Tensor) -> torch.Tensor:
        return self.text_adapter(text_embeddings)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def l2_normalize(array: np.ndarray) -> np.ndarray:
    return array / np.maximum(np.linalg.norm(array, axis=1, keepdims=True), EPS)


def safe_bank_name(bank_dir: Path) -> str:
    return bank_dir.name


def load_split_embeddings(embeddings_dir: Path) -> dict[str, SplitEmbeddingsV2]:
    payloads: dict[str, SplitEmbeddingsV2] = {}
    for split in ["train", "val", "test"]:
        manifest_path = embeddings_dir / f"image_embeddings_{split}_manifest.csv"
        embedding_path = embeddings_dir / f"image_embeddings_{split}.npy"
        if not manifest_path.exists() or not embedding_path.exists():
            raise FileNotFoundError(f"Missing cached image embeddings for split={split}: {manifest_path} / {embedding_path}")
        rows = read_csv_rows(manifest_path)
        embeddings = np.load(embedding_path).astype(np.float32)
        if len(rows) != embeddings.shape[0]:
            raise RuntimeError(f"Manifest/embedding mismatch for {split}: {len(rows)} rows vs {embeddings.shape[0]} embeddings")
        payloads[split] = SplitEmbeddingsV2(rows=rows, embeddings=l2_normalize(embeddings))
    return payloads


def load_fixed_bank_v2(bank_dir: Path, bank_name: str | None = None) -> FixedConceptBankV2:
    bank_dir = bank_dir.resolve()
    if not bank_dir.exists():
        raise FileNotFoundError(f"Bank directory not found: {bank_dir}")
    embedding_path = bank_dir / "filtered_concept_text_embeddings.npz"
    if not embedding_path.exists():
        raise FileNotFoundError(f"Missing bank text embeddings: {embedding_path}")

    payload = np.load(embedding_path, allow_pickle=True)
    merged_concepts = [str(item) for item in payload["concepts"].tolist()]
    merged_embeddings = l2_normalize(payload["concept_embeddings"].astype(np.float32))
    concept_to_indices: dict[str, list[int]] = {}
    for index, concept in enumerate(merged_concepts):
        concept_to_indices.setdefault(concept, []).append(index)

    per_class_rows: dict[str, list[dict[str, Any]]] = {}
    per_class_embeddings: dict[str, np.ndarray] = {}
    metadata_rows: list[dict[str, Any]] = []
    concept_class_membership = np.zeros((len(merged_concepts), len(LABEL_CODES)), dtype=np.float32)

    for class_name in LABEL_CODES:
        class_csv = bank_dir / f"{class_name}.csv"
        if not class_csv.exists():
            raise FileNotFoundError(f"Missing per-class concept CSV: {class_csv}")
        rows = read_csv_rows(class_csv)
        normalized_rows: list[dict[str, Any]] = []
        class_embeddings: list[np.ndarray] = []
        for rank, row in enumerate(rows, start=1):
            concept = str(row["concept"])
            if concept not in concept_to_indices:
                raise KeyError(f"Concept text not found in merged embedding payload: class={class_name} concept={concept}")
            merged_index = concept_to_indices[concept][0]
            class_embeddings.append(merged_embeddings[merged_index])
            concept_class_membership[merged_index, CLASS_TO_INDEX[class_name]] = 1.0

            normalized = {
                **row,
                "bank_name": bank_name or safe_bank_name(bank_dir),
                "bank_dir": str(bank_dir),
                "concept_class": class_name,
                "concept": concept,
                "concept_id": str(row.get("concept_id") or f"{class_name}_{rank:03d}"),
                "class_rank": int(row.get("retrieval_rank_in_class") or rank),
                "merged_index": int(merged_index),
                "class_label_rule": "loaded_from_per_class_csv_filename",
            }
            normalized_rows.append(normalized)
            metadata_rows.append(normalized)
        per_class_rows[class_name] = normalized_rows
        if class_embeddings:
            per_class_embeddings[class_name] = np.stack(class_embeddings, axis=0).astype(np.float32)
        else:
            raise RuntimeError(f"No concepts found for class {class_name} in {class_csv}")

    return FixedConceptBankV2(
        name=bank_name or safe_bank_name(bank_dir),
        bank_dir=bank_dir,
        per_class_rows=per_class_rows,
        per_class_embeddings=per_class_embeddings,
        merged_concepts=merged_concepts,
        merged_embeddings=merged_embeddings,
        metadata_rows=metadata_rows,
        concept_class_membership=concept_class_membership,
        class_label_rule="Concept class labels are loaded directly from per-class bank files HGC.csv/LGC.csv/NTL.csv/NST.csv.",
    )


def compute_image_text_class_matrix(bank: FixedConceptBankV2, split_payload: SplitEmbeddingsV2) -> np.ndarray:
    matrix = np.zeros((len(LABEL_CODES), len(LABEL_CODES)), dtype=np.float32)
    labels = np.array([str(row["class_name"]) for row in split_payload.rows], dtype=object)
    for image_class in LABEL_CODES:
        image_mask = labels == image_class
        if not image_mask.any():
            raise RuntimeError(f"No images found for class={image_class}")
        image_embeddings = split_payload.embeddings[image_mask]
        for concept_class in LABEL_CODES:
            text_embeddings = bank.per_class_embeddings[concept_class]
            matrix[CLASS_TO_INDEX[image_class], CLASS_TO_INDEX[concept_class]] = float(
                (image_embeddings @ text_embeddings.T).mean()
            )
    return matrix


def matrix_to_rows(matrix: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_index, image_class in enumerate(LABEL_CODES):
        item: dict[str, Any] = {"image_class": image_class}
        for col_index, concept_class in enumerate(LABEL_CODES):
            item[concept_class] = float(matrix[row_index, col_index])
        rows.append(item)
    return rows


def matrix_summary(bank_name: str, split: str, matrix: np.ndarray) -> dict[str, Any]:
    diagonal = np.diag(matrix)
    off_diagonal = matrix[~np.eye(matrix.shape[0], dtype=bool)]
    row_margins = {}
    for idx, class_name in enumerate(LABEL_CODES):
        other_values = np.delete(matrix[idx, :], idx)
        row_margins[f"row_margin_{class_name}"] = float(matrix[idx, idx] - other_values.max())
    col_margins = {}
    for idx, class_name in enumerate(LABEL_CODES):
        other_values = np.delete(matrix[:, idx], idx)
        col_margins[f"col_margin_{class_name}"] = float(matrix[idx, idx] - other_values.mean())
    return {
        "bank_name": bank_name,
        "split": split,
        "mean_diagonal": float(diagonal.mean()),
        "mean_off_diagonal": float(off_diagonal.mean()),
        "diagonal_minus_off_diagonal": float(diagonal.mean() - off_diagonal.mean()),
        "all_row_margins_positive": bool(all(value > 0.0 for value in row_margins.values())),
        **row_margins,
        **col_margins,
    }


def plot_matrix_heatmap(matrix: np.ndarray, path: Path, title: str) -> None:
    ensure_dir(path.parent)
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    image = ax.imshow(matrix, cmap="viridis")
    ax.set_xticks(range(len(LABEL_CODES)))
    ax.set_yticks(range(len(LABEL_CODES)))
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
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_concept_metadata(bank: FixedConceptBankV2, output_path: Path) -> None:
    fields = [
        "bank_name",
        "bank_dir",
        "concept_class",
        "concept_id",
        "concept",
        "class_rank",
        "merged_index",
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
        "mu_HGC",
        "mu_LGC",
        "mu_NTL",
        "mu_NST",
        "class_label_rule",
    ]
    write_csv(output_path, fields, bank.metadata_rows)


def build_soft_targets_v2(
    bank: FixedConceptBankV2,
    split_payloads: dict[str, SplitEmbeddingsV2],
) -> PreparedBankCacheV2:
    ordered_manifest: list[dict[str, Any]] = []
    embedding_chunks: list[np.ndarray] = []
    for split in ["train", "val", "test"]:
        for row in split_payloads[split].rows:
            ordered_manifest.append(
                {
                    "image_path": str(row["image_path"]),
                    "image_name": str(row.get("annotation_image_name") or Path(str(row["image_path"])).name),
                    "split": split,
                    "class_name": str(row["class_name"]),
                    "label_index": CLASS_TO_INDEX[str(row["class_name"])],
                }
            )
        embedding_chunks.append(split_payloads[split].embeddings)

    image_embeddings = np.concatenate(embedding_chunks, axis=0).astype(np.float32)
    similarity_matrix = image_embeddings @ bank.merged_embeddings.T
    split_array = np.array([row["split"] for row in ordered_manifest], dtype=object)
    class_array = np.array([row["class_name"] for row in ordered_manifest], dtype=object)
    train_mask = split_array == "train"

    mu = similarity_matrix[train_mask].mean(axis=0).astype(np.float32)
    sigma = similarity_matrix[train_mask].std(axis=0).astype(np.float32)
    soft_targets_all = (1.0 / (1.0 + np.exp(-(similarity_matrix - mu) / (sigma + EPS)))).astype(np.float32)

    prototypes: list[np.ndarray] = []
    for class_name in LABEL_CODES:
        class_mask = train_mask & (class_array == class_name)
        if not class_mask.any():
            raise RuntimeError(f"No train samples for class={class_name}")
        prototypes.append(soft_targets_all[class_mask].mean(axis=0))
    prototype_matrix = np.stack(prototypes, axis=0).astype(np.float32)

    split_rows: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for index, row in enumerate(ordered_manifest):
        split_rows[str(row["split"])].append(
            {
                **row,
                "soft_concepts": soft_targets_all[index],
                "global_index": index,
            }
        )

    return PreparedBankCacheV2(
        bank=bank,
        split_rows=split_rows,
        prototype_matrix=prototype_matrix,
        mu=mu,
        sigma=sigma,
        soft_targets_all=soft_targets_all,
        ordered_manifest=ordered_manifest,
    )


def save_prepared_bank_cache(cache: PreparedBankCacheV2, output_dir: Path, write_root_alias: bool = False) -> None:
    bank_dir = output_dir / cache.bank.name
    ensure_dir(bank_dir)
    targets = [bank_dir]
    if write_root_alias:
        targets.append(output_dir)

    for target in targets:
        np.savez_compressed(
            target / "soft_targets_all.npz",
            soft_targets=cache.soft_targets_all.astype(np.float32),
            mu=cache.mu.astype(np.float32),
            sigma=cache.sigma.astype(np.float32),
            prototype_matrix=cache.prototype_matrix.astype(np.float32),
            concepts=np.array(cache.bank.merged_concepts, dtype=object),
            concept_embeddings=cache.bank.merged_embeddings.astype(np.float32),
            concept_class_membership=cache.bank.concept_class_membership.astype(np.float32),
            image_path=np.array([row["image_path"] for row in cache.ordered_manifest], dtype=object),
            split=np.array([row["split"] for row in cache.ordered_manifest], dtype=object),
            class_name=np.array([row["class_name"] for row in cache.ordered_manifest], dtype=object),
            label_index=np.array([row["label_index"] for row in cache.ordered_manifest], dtype=np.int64),
        )
        write_prototype_matrix(cache, target / "prototype_matrix.csv")
        write_candidate_pools(cache, target / "image_concept_candidate_pools.csv")
        write_concept_metadata(cache.bank, target / "concept_metadata_with_class.csv")


def write_prototype_matrix(cache: PreparedBankCacheV2, output_path: Path) -> None:
    rows = []
    for concept_index, concept in enumerate(cache.bank.merged_concepts):
        item: dict[str, Any] = {
            "concept_index": concept_index,
            "concept": concept,
        }
        for class_index, class_name in enumerate(LABEL_CODES):
            item[class_name] = float(cache.prototype_matrix[class_index, concept_index])
        rows.append(item)
    write_csv(output_path, ["concept_index", "concept", *LABEL_CODES], rows)


def write_candidate_pools(cache: PreparedBankCacheV2, output_path: Path) -> None:
    concept_classes = cache.bank.concept_class_membership.argmax(axis=1)
    rows = []
    for row in cache.ordered_manifest:
        label_index = int(row["label_index"])
        positive = np.where(concept_classes == label_index)[0]
        negative = np.where(concept_classes != label_index)[0]
        rows.append(
            {
                "global_index": len(rows),
                "split": row["split"],
                "image_path": row["image_path"],
                "class_name": row["class_name"],
                "label_index": label_index,
                "positive_concept_count": int(positive.size),
                "negative_concept_count": int(negative.size),
                "positive_concept_indices": " ".join(str(int(item)) for item in positive.tolist()),
                "negative_concept_indices": " ".join(str(int(item)) for item in negative.tolist()),
            }
        )
    write_csv(
        output_path,
        [
            "global_index",
            "split",
            "image_path",
            "class_name",
            "label_index",
            "positive_concept_count",
            "negative_concept_count",
            "positive_concept_indices",
            "negative_concept_indices",
        ],
        rows,
    )


def write_soft_target_summary(path: Path, caches: list[PreparedBankCacheV2]) -> None:
    lines = [
        "# Phase 2 Soft Target Cache Summary",
        "",
        "Soft targets use the existing definition: image-text cosine similarity, train-only concept-wise mu/sigma, then sigmoid standardization.",
        "",
        "| Bank | Concepts | Train | Val | Test | Prototype Shape | Mean Soft Target |",
        "|---|---:|---:|---:|---:|---|---:|",
    ]
    for cache in caches:
        lines.append(
            f"| `{cache.bank.name}` | {len(cache.bank.merged_concepts)} | "
            f"{len(cache.split_rows['train'])} | {len(cache.split_rows['val'])} | {len(cache.split_rows['test'])} | "
            f"{list(cache.prototype_matrix.shape)} | {float(cache.soft_targets_all.mean()):.6f} |"
        )
    lines.extend(
        [
            "",
            "Concept class labels are loaded from per-class bank CSV filenames. Candidate pools are generated per image by image class == concept class.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_v2_view_transforms(
    normalize_transform: Any | None,
    n_views: int = 4,
    image_size: int = 224,
) -> tuple[list[Any], list[str]]:
    if n_views not in {2, 4, 5}:
        raise ValueError("V2 supports n_views in {2, 4, 5}.")

    def finish(steps: list[Any]) -> transforms.Compose:
        final_steps = [*steps, transforms.ToTensor()]
        if normalize_transform is not None:
            final_steps.append(normalize_transform)
        return transforms.Compose(final_steps)

    color_view = lambda strength: finish(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ColorJitter(
                brightness=0.08 + 0.02 * strength,
                contrast=0.08 + 0.02 * strength,
                saturation=0.03 + 0.02 * strength,
                hue=0.01 + 0.01 * strength,
            ),
        ]
    )
    geom_view = lambda strength: finish(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.90 - 0.02 * strength, 1.0), ratio=(0.96, 1.04)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(6 + 2 * strength),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.01 + 0.01 * strength, 0.01 + 0.01 * strength),
                scale=(0.98 - 0.01 * strength, 1.02 + 0.01 * strength),
                shear=(-2 - strength, 2 + strength),
            ),
        ]
    )
    mixed_view = finish(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.92, 1.0), ratio=(0.96, 1.04)),
            transforms.RandomRotation(5),
            transforms.ColorJitter(brightness=0.06, contrast=0.06, saturation=0.03, hue=0.01),
        ]
    )

    if n_views == 2:
        return [color_view(1), geom_view(1)], ["color_1", "geometry_1"]
    if n_views == 4:
        return [color_view(1), color_view(2), geom_view(1), geom_view(2)], [
            "color_1",
            "color_2",
            "geometry_1",
            "geometry_2",
        ]
    return [color_view(1), color_view(2), geom_view(1), geom_view(2), mixed_view], [
        "color_1",
        "color_2",
        "geometry_1",
        "geometry_2",
        "mixed_1",
    ]


def save_augmentation_preview(
    split_payloads: dict[str, SplitEmbeddingsV2],
    output_dir: Path,
    n_views: int = 4,
    max_images: int = 4,
) -> tuple[Path, list[str]]:
    ensure_dir(output_dir)
    preview_transforms, view_names = build_v2_view_transforms(normalize_transform=None, n_views=n_views)
    image_tensors: list[torch.Tensor] = []
    rows = split_payloads["train"].rows[:max_images]
    for row in rows:
        image = Image.open(str(row["image_path"])).convert("RGB")
        for transform in preview_transforms:
            image_tensors.append(transform(image))
    grid_path = output_dir / f"augmentation_preview_{n_views}views.png"
    save_image(torch.stack(image_tensors, dim=0), grid_path, nrow=n_views)
    return grid_path, view_names


def select_balanced_rows(rows: list[dict[str, Any]], max_rows: int) -> list[dict[str, Any]]:
    """Select a deterministic class-balanced subset so pair debug includes positive and negative pairs."""
    by_class: dict[str, list[dict[str, Any]]] = {class_name: [] for class_name in LABEL_CODES}
    for row in rows:
        class_name = str(row["class_name"])
        if class_name in by_class:
            by_class[class_name].append(row)
    per_class = max(max_rows // len(LABEL_CODES), 1)
    selected: list[dict[str, Any]] = []
    for class_name in LABEL_CODES:
        selected.extend(by_class[class_name][:per_class])
    remaining = max_rows - len(selected)
    if remaining > 0:
        selected_paths = {str(row["image_path"]) for row in selected}
        for row in rows:
            if str(row["image_path"]) not in selected_paths:
                selected.append(row)
                selected_paths.add(str(row["image_path"]))
                remaining -= 1
                if remaining <= 0:
                    break
    return selected[:max_rows]


def write_augmentation_config_summary(path: Path, n_views: int, view_names: list[str], preview_path: Path) -> None:
    lines = [
        "# Phase 3 Augmentation Config Summary",
        "",
        f"Configured V2 training views: `{n_views}`.",
        "",
        "View names:",
        "",
        *[f"- `{name}`" for name in view_names],
        "",
        "Default V2 behavior for 4 views is 2 color-focused views and 2 geometry-focused views.",
        "",
        "Preview image:",
        "",
        f"```text\n{preview_path}\n```",
        "",
        "Validation and test remain single-view using BioMedCLIP evaluation preprocessing.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_image_image_pairs_v2(
    label_indices: torch.Tensor,
    soft_concepts: torch.Tensor,
    n_views: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    same = label_indices[:, None].eq(label_indices[None, :])
    diff = ~same
    normalized = F.normalize(soft_concepts, dim=-1)
    sample_weights = torch.clamp(normalized @ normalized.T, min=0.0, max=1.0) * same.float()
    positive_sample_mask = same
    negative_sample_mask = diff

    positive_mask = positive_sample_mask.repeat_interleave(n_views, 0).repeat_interleave(n_views, 1)
    negative_mask = negative_sample_mask.repeat_interleave(n_views, 0).repeat_interleave(n_views, 1)
    positive_weights = sample_weights.repeat_interleave(n_views, 0).repeat_interleave(n_views, 1)
    identity = torch.eye(positive_mask.shape[0], device=positive_mask.device, dtype=torch.bool)
    positive_mask = positive_mask & ~identity
    negative_mask = negative_mask & ~identity
    positive_weights = positive_weights.masked_fill(~positive_mask, 0.0)
    positive_values = positive_weights[positive_mask]
    diagnostics = {
        "image_image_positive_pair_count": int(positive_mask.sum().item()),
        "image_image_negative_pair_count": int(negative_mask.sum().item()),
        "image_image_positive_weight_mean": float(positive_values.mean().item()) if positive_values.numel() else 0.0,
        "image_image_positive_weight_min": float(positive_values.min().item()) if positive_values.numel() else 0.0,
        "image_image_positive_weight_max": float(positive_values.max().item()) if positive_values.numel() else 0.0,
    }
    return positive_mask, negative_mask, positive_weights, diagnostics


def build_image_concept_pairs_v2(
    label_indices: torch.Tensor,
    soft_concepts: torch.Tensor,
    concept_class_membership: torch.Tensor,
    normalize_positive_weights: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    # concept_class_membership is [num_concepts, num_classes].
    # For each image label, select the concept column for that class and transpose to [batch, num_concepts].
    positive_mask = concept_class_membership[:, label_indices].T.bool()
    negative_mask = ~positive_mask
    positive_weights = soft_concepts * positive_mask.float()
    if normalize_positive_weights:
        positive_weights = positive_weights / positive_weights.sum(dim=1, keepdim=True).clamp_min(EPS)
    positive_values = positive_weights[positive_mask]
    diagnostics = {
        "image_concept_positive_pair_count": int(positive_mask.sum().item()),
        "image_concept_negative_pair_count": int(negative_mask.sum().item()),
        "image_concept_positive_weight_mean": float(positive_values.mean().item()) if positive_values.numel() else 0.0,
        "image_concept_positive_weight_min": float(positive_values.min().item()) if positive_values.numel() else 0.0,
        "image_concept_positive_weight_max": float(positive_values.max().item()) if positive_values.numel() else 0.0,
    }
    return positive_mask, negative_mask, positive_weights, diagnostics


def weighted_image_image_contrastive_loss_v2(
    image_projections: torch.Tensor,
    positive_weights: torch.Tensor,
    tau: float = 0.07,
) -> torch.Tensor:
    projections_flat = image_projections.reshape(-1, image_projections.shape[-1])
    logits = projections_flat @ projections_flat.T / tau
    identity = torch.eye(logits.shape[0], device=logits.device, dtype=torch.bool)
    logits = logits.masked_fill(identity, float("-inf"))
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    log_prob = torch.where(identity, torch.zeros_like(log_prob), log_prob)
    weight_sums = positive_weights.sum(dim=1).clamp_min(EPS)
    return (-(positive_weights * log_prob).sum(dim=1) / weight_sums).mean()


def weighted_image_text_contrastive_loss_v2(
    image_projections: torch.Tensor,
    text_projections: torch.Tensor,
    positive_weights: torch.Tensor,
    tau: float = 0.07,
) -> torch.Tensor:
    logits = image_projections @ text_projections.T / tau
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    weight_sums = positive_weights.sum(dim=1).clamp_min(EPS)
    return (-(positive_weights * log_prob).sum(dim=1) / weight_sums).mean()


def write_pair_debug_outputs(
    path_md: Path,
    path_csv: Path,
    cache: PreparedBankCacheV2,
    n_views: int = 4,
    debug_samples: int = 64,
) -> list[dict[str, Any]]:
    ensure_dir(path_md.parent)
    rows = select_balanced_rows(cache.split_rows["train"], debug_samples)
    labels = torch.tensor([int(row["label_index"]) for row in rows], dtype=torch.long)
    soft = torch.tensor(np.stack([row["soft_concepts"] for row in rows]), dtype=torch.float32)
    membership = torch.tensor(cache.bank.concept_class_membership, dtype=torch.float32)
    _, _, ii_weights, ii_diag = build_image_image_pairs_v2(labels, soft, n_views=n_views)
    _, _, ic_weights, ic_diag = build_image_concept_pairs_v2(labels, soft, membership)
    table_rows = []
    for key, value in {**ii_diag, **ic_diag}.items():
        table_rows.append({"bank_name": cache.bank.name, "n_views": n_views, "debug_samples": len(rows), "metric": key, "value": value})
    table_rows.append({"bank_name": cache.bank.name, "n_views": n_views, "debug_samples": len(rows), "metric": "image_image_weight_shape", "value": list(ii_weights.shape)})
    table_rows.append({"bank_name": cache.bank.name, "n_views": n_views, "debug_samples": len(rows), "metric": "image_concept_weight_shape", "value": list(ic_weights.shape)})
    write_csv(path_csv, ["bank_name", "n_views", "debug_samples", "metric", "value"], table_rows)
    lines = [
        "# Phase 4 Pair Debug Summary",
        "",
        f"Bank: `{cache.bank.name}`",
        f"Debug train samples: `{len(rows)}`",
        f"Views: `{n_views}`",
        "",
        "Image-image positives are same-class pairs. Positive weights are cosine similarities between soft concept vectors.",
        "",
        "Image-concept positives are image class == concept class. Positive weights are per-image soft concept activations normalized over positive concepts.",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for row in table_rows:
        lines.append(f"| `{row['metric']}` | {row['value']} |")
    path_md.write_text("\n".join(lines), encoding="utf-8")
    return table_rows



def write_matrix_summary_markdown(path: Path, summary_rows: list[dict[str, Any]], bank_paths: dict[str, Path]) -> None:
    ensure_dir(path.parent)
    lines: list[str] = [
        "# Phase 1 Image-Text Class Similarity Matrix Summary",
        "",
        "## Concept Class Label Rule",
        "",
        "Concept class labels were loaded from the exported bank's per-class CSV files:",
        "",
        "```text",
        "HGC.csv -> HGC concepts",
        "LGC.csv -> LGC concepts",
        "NTL.csv -> NTL concepts",
        "NST.csv -> NST concepts",
        "```",
        "",
        "No semantic inference was used for class labels in Phase 1.",
        "",
        "## Banks",
        "",
    ]
    for bank_name, bank_path in bank_paths.items():
        lines.append(f"- `{bank_name}`: `{bank_path}`")
    lines.extend(
        [
            "",
            "## Summary",
            "",
            "| Bank | Split | Diagonal Mean | Off-Diagonal Mean | Gap | All Row Margins Positive |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for row in summary_rows:
        lines.append(
            f"| `{row['bank_name']}` | `{row['split']}` | "
            f"{float(row['mean_diagonal']):.6f} | {float(row['mean_off_diagonal']):.6f} | "
            f"{float(row['diagonal_minus_off_diagonal']):.6f} | {row['all_row_margins_positive']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Rule",
            "",
            "The desired class structure is: diagonal entries should be higher than off-diagonal entries.",
            "",
            "The global `Gap` column checks average diagonal minus average off-diagonal similarity. "
            "`All Row Margins Positive` is stricter: every image class must prefer its own concept class over the strongest other concept class.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
