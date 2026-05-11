#!/usr/bin/env python3
"""Discriminative whitelist + concept-profile CyCL experiment.

This script implements the requested next-stage experiment:

1. Start from the fixed top300 concepts/class bank.
2. Compute class-wise mean image-text similarities on the train split.
3. Rank each concept by own-class mean minus hardest-negative mean.
4. Build top-k whitelists for k in {5, 10, 20, 30}.
5. Construct a non-one-hot concept-class association matrix M.
6. Train a minimal frozen-BioMedCLIP-embedding CBM/CyCL model:
   image embedding -> adapter MLP -> concept similarity activations -> linear classifier.

The model uses cached BioMedCLIP embeddings; it does not re-encode raw images.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import ttest_ind
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ebtc_project_paths import FILTERED_TOP300_BANK_DIR, OFFICIAL_EMBEDDINGS_DIR, OUTPUT_ROOT


CLASSES = ["HGC", "LGC", "NTL", "NST"]
CLASS_TO_INDEX = {name: idx for idx, name in enumerate(CLASSES)}
EPS = 1e-8

DEFAULT_OUTPUT_DIR = OUTPUT_ROOT / "ebtc_discriminative_whitelist_cycl_outputs"

NON_VISUAL_PATTERNS = [
    r"\b(histolog\w*|microscop\w*|cytolog\w*|cellular|nuclear|nuclei|mitotic|chromatin)\b",
    r"\b(biopsy|patholog\w*|diagnos\w*|grade|grading|staging|stage)\b",
    r"\b(carcinoma|cancer|malignan\w*|benign|dysplasia|neoplas\w*)\b",
    r"\b(lamina propria|muscularis|invasion|invasive|metast\w*)\b",
    r"\b(hgc|lgc|ntl|nst|high grade|low grade|non[- ]tumou?r|non[- ]suspicious)\b",
]
NON_VISUAL_REGEX = [re.compile(pattern, flags=re.IGNORECASE) for pattern in NON_VISUAL_PATTERNS]
TOKEN_STOPWORDS = {
    "with",
    "and",
    "the",
    "of",
    "in",
    "on",
    "a",
    "an",
    "to",
    "like",
    "showing",
    "image",
}


@dataclass
class SplitPayload:
    embeddings: np.ndarray
    labels: np.ndarray
    rows: list[dict[str, str]]


@dataclass
class BankPayload:
    rows: pd.DataFrame
    concepts: np.ndarray
    embeddings: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discriminative whitelist + concept-profile CyCL experiment.")
    parser.add_argument("--bank-dir", type=Path, default=FILTERED_TOP300_BANK_DIR)
    parser.add_argument("--embeddings-dir", type=Path, default=OFFICIAL_EMBEDDINGS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-ks", default="5,10,20,30")
    parser.add_argument("--m-normalization", choices=["minmax", "softmax"], default="minmax")
    parser.add_argument("--softmax-temperature", type=float, default=0.02)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.995)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lambda-cycl", type=float, default=0.05)
    parser.add_argument("--lambda-align", type=float, default=0.2)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument(
        "--num-threads",
        type=int,
        default=2,
        help="Limit PyTorch CPU threads. Use 0 to leave the runtime default unchanged.",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    ensure_dir(path.parent)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def l2_normalize(array: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.clip(denom, EPS, None)


def parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_split_payloads(embeddings_dir: Path) -> dict[str, SplitPayload]:
    payloads: dict[str, SplitPayload] = {}
    for split in ["train", "val", "test"]:
        embedding_path = embeddings_dir / f"image_embeddings_{split}.npy"
        manifest_path = embeddings_dir / f"image_embeddings_{split}_manifest.csv"
        embeddings = l2_normalize(np.load(embedding_path).astype(np.float32))
        rows = read_manifest(manifest_path)
        if len(rows) != embeddings.shape[0]:
            raise RuntimeError(f"Split {split} row mismatch: {len(rows)} manifest rows vs {embeddings.shape[0]} embeddings.")
        labels = np.array([CLASS_TO_INDEX[row["class_name"]] for row in rows], dtype=np.int64)
        payloads[split] = SplitPayload(embeddings=embeddings, labels=labels, rows=rows)
    return payloads


def load_top300_bank(bank_dir: Path) -> BankPayload:
    rows_path = bank_dir / "merged_concepts.csv"
    embeddings_path = bank_dir / "filtered_concept_text_embeddings.npz"
    rows = pd.read_csv(rows_path)
    payload = np.load(embeddings_path, allow_pickle=True)
    concepts = payload["concepts"]
    embeddings = l2_normalize(payload["concept_embeddings"].astype(np.float32))
    if len(rows) != embeddings.shape[0]:
        raise RuntimeError(f"Bank row mismatch: {len(rows)} rows vs {embeddings.shape[0]} embeddings.")
    return BankPayload(rows=rows, concepts=concepts, embeddings=embeddings)


def compute_class_means(train: SplitPayload, concept_embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    similarities = train.embeddings @ concept_embeddings.T
    class_means = np.zeros((concept_embeddings.shape[0], len(CLASSES)), dtype=np.float32)
    for class_name, class_idx in CLASS_TO_INDEX.items():
        mask = train.labels == class_idx
        class_means[:, class_idx] = similarities[mask].mean(axis=0)
    return class_means, similarities


def compute_discriminative_scores(bank: BankPayload, train: SplitPayload) -> tuple[pd.DataFrame, np.ndarray]:
    class_means, similarities = compute_class_means(train, bank.embeddings)
    rows: list[dict[str, Any]] = []
    for idx, source in bank.rows.iterrows():
        target_class = str(source["primary_class"])
        target_idx = CLASS_TO_INDEX[target_class]
        own = float(class_means[idx, target_idx])
        negative_indices = [i for i in range(len(CLASSES)) if i != target_idx]
        hardest_idx = max(negative_indices, key=lambda class_idx: float(class_means[idx, class_idx]))
        hardest_class = CLASSES[hardest_idx]
        hardest_mean = float(class_means[idx, hardest_idx])
        own_values = similarities[train.labels == target_idx, idx]
        hard_values = similarities[train.labels == hardest_idx, idx]
        ttest = ttest_ind(own_values, hard_values, equal_var=False, nan_policy="omit")
        row = {
            "concept_index": int(idx),
            "concept": str(source["concept"]),
            "target_class": target_class,
            "concept_id": str(source["primary_concept_id"]),
            "source_rank_in_filtered_top300": int(source["primary_source_rank_in_filtered_top300"]),
            "own_class_mean": own,
            "hardest_negative_class": hardest_class,
            "hardest_negative_mean": hardest_mean,
            "margin": own - hardest_mean,
            "variance": float(np.var(similarities[:, idx], ddof=1)),
            "p_value": float(ttest.pvalue) if not math.isnan(float(ttest.pvalue)) else "",
            "mu_HGC": float(class_means[idx, CLASS_TO_INDEX["HGC"]]),
            "mu_LGC": float(class_means[idx, CLASS_TO_INDEX["LGC"]]),
            "mu_NTL": float(class_means[idx, CLASS_TO_INDEX["NTL"]]),
            "mu_NST": float(class_means[idx, CLASS_TO_INDEX["NST"]]),
        }
        rows.append(row)
    score_df = pd.DataFrame(rows)
    return score_df, class_means


def normalize_concept_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def content_tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if token not in TOKEN_STOPWORDS and len(token) > 2}


def is_non_visual_or_microscopic(text: str) -> tuple[bool, str]:
    for regex in NON_VISUAL_REGEX:
        match = regex.search(text)
        if match:
            return True, match.group(0)
    return False, ""


def build_top_tables(score_df: pd.DataFrame, top_ks: list[int], output_dir: Path) -> None:
    for top_k in top_ks:
        all_rows: list[dict[str, Any]] = []
        for class_name in CLASSES:
            class_df = score_df[score_df["target_class"] == class_name].sort_values("margin", ascending=False).head(top_k)
            rows = class_df.to_dict(orient="records")
            for rank, row in enumerate(rows, start=1):
                row["rank_margin"] = rank
            write_csv(output_dir / f"top_concepts_k{top_k}_{class_name}.csv", rows)
            all_rows.extend(rows)
        write_csv(output_dir / f"top_concepts_k{top_k}_all.csv", all_rows)


def try_find_near_duplicate(
    selected: list[dict[str, Any]],
    embedding: np.ndarray,
    normalized_text: str,
    threshold: float,
) -> int | None:
    tokens = content_tokens(normalized_text)
    for idx, record in enumerate(selected):
        if record["concept_clean"] == normalized_text:
            return idx
        other_tokens = content_tokens(record["concept_clean"])
        if not tokens or not other_tokens:
            continue
        jaccard = len(tokens & other_tokens) / max(1, len(tokens | other_tokens))
        cosine = float(np.dot(record["embedding"], embedding))
        if cosine >= threshold and jaccard >= 0.5:
            return idx
    return None


def merge_source_record(record: dict[str, Any], row: dict[str, Any]) -> None:
    target = row["target_class"]
    if target not in record["selected_target_classes"]:
        record["selected_target_classes"].append(target)
    record["source_concept_ids"].append(row["concept_id"])
    record["source_indices"].append(int(row["concept_index"]))
    record["source_margins"].append(float(row["margin"]))
    record["source_hardest_negative_classes"].append(row["hardest_negative_class"])


def build_whitelist(
    score_df: pd.DataFrame,
    bank: BankPayload,
    class_means: np.ndarray,
    top_k: int,
    output_dir: Path,
    near_duplicate_threshold: float,
    normalization: str,
    softmax_temperature: float,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    selected_rows: list[dict[str, Any]] = []
    for class_name in CLASSES:
        rows = (
            score_df[score_df["target_class"] == class_name]
            .sort_values("margin", ascending=False)
            .head(top_k)
            .to_dict(orient="records")
        )
        selected_rows.extend(rows)

    cleaned: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for row in selected_rows:
        concept = str(row["concept"])
        is_bad, reason = is_non_visual_or_microscopic(concept)
        if is_bad:
            dropped.append({**row, "drop_reason": f"non_visual_or_microscopic:{reason}"})
            continue
        concept_idx = int(row["concept_index"])
        embedding = bank.embeddings[concept_idx]
        clean = normalize_concept_text(concept)
        duplicate_idx = try_find_near_duplicate(cleaned, embedding, clean, near_duplicate_threshold)
        if duplicate_idx is None:
            record = {
                "whitelist_index": len(cleaned),
                "concept": concept,
                "concept_clean": clean,
                "representative_concept_index": concept_idx,
                "representative_concept_id": row["concept_id"],
                "primary_target_class": row["target_class"],
                "selected_target_classes": [],
                "source_concept_ids": [],
                "source_indices": [],
                "source_margins": [],
                "source_hardest_negative_classes": [],
                "embedding": embedding,
            }
            merge_source_record(record, row)
            cleaned.append(record)
        else:
            merge_source_record(cleaned[duplicate_idx], row)

    whitelist_rows: list[dict[str, Any]] = []
    representative_indices: list[int] = []
    for new_idx, record in enumerate(cleaned):
        rep_idx = int(record["representative_concept_index"])
        representative_indices.append(rep_idx)
        whitelist_rows.append(
            {
                "whitelist_index": new_idx,
                "concept": record["concept"],
                "concept_clean": record["concept_clean"],
                "representative_concept_index": rep_idx,
                "representative_concept_id": record["representative_concept_id"],
                "primary_target_class": record["primary_target_class"],
                "selected_target_classes": "|".join(record["selected_target_classes"]),
                "source_concept_ids": "|".join(record["source_concept_ids"]),
                "source_indices": "|".join(str(item) for item in record["source_indices"]),
                "source_margins": "|".join(f"{item:.8f}" for item in record["source_margins"]),
                "source_hardest_negative_classes": "|".join(record["source_hardest_negative_classes"]),
                "source_selection_count": len(record["source_indices"]),
            }
        )

    whitelist_df = pd.DataFrame(whitelist_rows)
    representative_indices_np = np.array(representative_indices, dtype=np.int64)
    raw_m = class_means[representative_indices_np].astype(np.float32)
    normalized_m = normalize_m(raw_m, normalization=normalization, temperature=softmax_temperature)
    concept_embeddings = bank.embeddings[representative_indices_np].astype(np.float32)

    write_csv(output_dir / f"whitelist_top{top_k}.csv", whitelist_df.to_dict(orient="records"))
    write_csv(output_dir / f"whitelist_top{top_k}_dropped.csv", dropped)
    save_matrices(output_dir, top_k, whitelist_df, raw_m, normalized_m)
    np.savez_compressed(
        output_dir / f"whitelist_top{top_k}.npz",
        concepts=whitelist_df["concept"].to_numpy(dtype=object),
        concept_embeddings=concept_embeddings,
        raw_M=raw_m,
        M=normalized_m,
        classes=np.array(CLASSES, dtype=object),
    )
    return whitelist_df, concept_embeddings, raw_m, normalized_m


def normalize_m(raw_m: np.ndarray, normalization: str, temperature: float) -> np.ndarray:
    if normalization == "softmax":
        scaled = raw_m / max(temperature, EPS)
        scaled = scaled - scaled.max(axis=1, keepdims=True)
        exp = np.exp(scaled)
        return (exp / np.clip(exp.sum(axis=1, keepdims=True), EPS, None)).astype(np.float32)
    row_min = raw_m.min(axis=1, keepdims=True)
    row_max = raw_m.max(axis=1, keepdims=True)
    denom = row_max - row_min
    return ((raw_m - row_min) / np.clip(denom, EPS, None)).astype(np.float32)


def save_matrices(output_dir: Path, top_k: int, whitelist_df: pd.DataFrame, raw_m: np.ndarray, normalized_m: np.ndarray) -> None:
    raw_rows: list[dict[str, Any]] = []
    norm_rows: list[dict[str, Any]] = []
    for idx, row in whitelist_df.iterrows():
        base = {
            "whitelist_index": int(idx),
            "concept": row["concept"],
            "primary_target_class": row["primary_target_class"],
            "selected_target_classes": row["selected_target_classes"],
        }
        raw_rows.append({**base, **{f"mu_{class_name}": float(raw_m[idx, class_idx]) for class_name, class_idx in CLASS_TO_INDEX.items()}})
        norm_rows.append({**base, **{class_name: float(normalized_m[idx, class_idx]) for class_name, class_idx in CLASS_TO_INDEX.items()}})
    write_csv(output_dir / f"M_matrix_top{top_k}_raw.csv", raw_rows)
    write_csv(output_dir / f"M_matrix_top{top_k}_normalized.csv", norm_rows)

    save_m_heatmap(
        output_dir / f"M_matrix_top{top_k}_heatmap.png",
        normalized_m,
        [str(item) for item in whitelist_df["concept"].tolist()],
        title=f"Concept-class association M (top{top_k}, normalized)",
    )
    class_sim = class_profile_cosine(normalized_m)
    save_square_matrix_csv(output_dir / f"class_profile_similarity_top{top_k}.csv", class_sim, CLASSES)
    save_square_heatmap(
        output_dir / f"class_profile_similarity_top{top_k}.png",
        class_sim,
        CLASSES,
        title=f"Class-class concept profile cosine similarity (top{top_k})",
        vmin=0.0,
        vmax=1.0,
    )


def class_profile_cosine(m_matrix: np.ndarray) -> np.ndarray:
    profiles = m_matrix.T
    denom = np.linalg.norm(profiles, axis=1, keepdims=True)
    normalized = profiles / np.clip(denom, EPS, None)
    return normalized @ normalized.T


def save_square_matrix_csv(path: Path, matrix: np.ndarray, labels: list[str]) -> None:
    rows: list[dict[str, Any]] = []
    for i, row_label in enumerate(labels):
        row = {"class": row_label}
        for j, col_label in enumerate(labels):
            row[col_label] = float(matrix[i, j])
        rows.append(row)
    write_csv(path, rows)


def save_square_heatmap(path: Path, matrix: np.ndarray, labels: list[str], title: str, vmin: float | None = None, vmax: float | None = None) -> None:
    ensure_dir(path.parent)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(labels)), labels=labels)
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", color="white")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_m_heatmap(path: Path, matrix: np.ndarray, concepts: list[str], title: str) -> None:
    ensure_dir(path.parent)
    height = max(6, min(24, 0.22 * len(concepts)))
    fig, ax = plt.subplots(figsize=(8, height))
    im = ax.imshow(matrix, aspect="auto", cmap="magma", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(CLASSES)), labels=CLASSES)
    if len(concepts) <= 45:
        ax.set_yticks(range(len(concepts)), labels=concepts, fontsize=6)
    else:
        ax.set_yticks([])
        ax.set_ylabel(f"{len(concepts)} whitelist concepts")
    ax.set_xlabel("Class")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


class AdapterConceptCBM(nn.Module):
    def __init__(self, concept_embeddings: np.ndarray, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        input_dim = concept_embeddings.shape[1]
        num_concepts = concept_embeddings.shape[0]
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )
        self.classifier = nn.Linear(num_concepts, len(CLASSES))
        self.register_buffer("concept_embeddings", torch.tensor(concept_embeddings, dtype=torch.float32))

    def forward(self, image_embeddings: torch.Tensor) -> dict[str, torch.Tensor]:
        adapted = F.normalize(image_embeddings + self.adapter(image_embeddings), dim=-1)
        concept_activations = adapted @ self.concept_embeddings.T
        logits = self.classifier(concept_activations)
        return {
            "adapted_embeddings": adapted,
            "concept_activations": concept_activations,
            "logits": logits,
        }


def row_minmax_tensor(values: torch.Tensor) -> torch.Tensor:
    row_min = values.min(dim=1, keepdim=True).values
    row_max = values.max(dim=1, keepdim=True).values
    return (values - row_min) / torch.clamp(row_max - row_min, min=EPS)


def profile_weighted_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    class_profiles: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    batch_size = embeddings.shape[0]
    if batch_size <= 1:
        return embeddings.new_tensor(0.0)
    logits = embeddings @ embeddings.T / tau
    eye = torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)
    logits = logits.masked_fill(eye, -1e9)

    profiles = F.normalize(class_profiles[labels], dim=-1)
    weights = torch.clamp(profiles @ profiles.T, min=0.0, max=1.0)
    weights = weights.masked_fill(eye, 0.0)
    denom = weights.sum(dim=1)
    valid = denom > 0
    if not bool(valid.any()):
        return embeddings.new_tensor(0.0)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    loss = -(weights * log_prob).sum(dim=1) / torch.clamp(denom, min=EPS)
    return loss[valid].mean()


def make_loader(payload: SplitPayload, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(payload.embeddings, dtype=torch.float32),
        torch.tensor(payload.labels, dtype=torch.long),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_class_weights(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    counts = np.bincount(labels, minlength=len(CLASSES)).astype(np.float32)
    weights = counts.sum() / np.clip(len(CLASSES) * counts, EPS, None)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def evaluate_model(model: AdapterConceptCBM, payload: SplitPayload, device: torch.device, batch_size: int) -> dict[str, Any]:
    loader = make_loader(payload, batch_size=batch_size, shuffle=False)
    model.eval()
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    with torch.no_grad():
        for features, labels in loader:
            outputs = model(features.to(device))
            all_logits.append(outputs["logits"].detach().cpu().numpy())
            all_labels.append(labels.numpy())
    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    probs = softmax_np(logits)
    preds = probs.argmax(axis=1)
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, preds, labels=list(range(len(CLASSES)))).tolist(),
    }
    try:
        metrics["macro_auroc"] = float(roc_auc_score(labels, probs, multi_class="ovr", average="macro", labels=list(range(len(CLASSES)))))
    except ValueError:
        metrics["macro_auroc"] = ""
    precision, recall, f1, support = precision_recall_fscore_support(labels, preds, labels=list(range(len(CLASSES))), zero_division=0)
    for idx, class_name in enumerate(CLASSES):
        metrics[f"{class_name}_precision"] = float(precision[idx])
        metrics[f"{class_name}_recall"] = float(recall[idx])
        metrics[f"{class_name}_f1"] = float(f1[idx])
        metrics[f"{class_name}_support"] = int(support[idx])
    return metrics


def softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.clip(exp.sum(axis=1, keepdims=True), EPS, None)


def train_one_setting(
    split_payloads: dict[str, SplitPayload],
    concept_embeddings: np.ndarray,
    m_matrix: np.ndarray,
    top_k: int,
    seed: int,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    set_seed(seed)
    device = resolve_device(args.device)
    model = AdapterConceptCBM(concept_embeddings, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    class_weights = compute_class_weights(split_payloads["train"].labels, device)
    class_profiles = torch.tensor(m_matrix.T, dtype=torch.float32, device=device)
    train_loader = make_loader(split_payloads["train"], batch_size=args.batch_size, shuffle=True)
    best_state: dict[str, torch.Tensor] | None = None
    best_val_f1 = -1.0
    best_epoch = 0
    patience_counter = 0
    curve_rows: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = {"total": 0.0, "cls": 0.0, "cycl": 0.0, "align": 0.0}
        seen = 0
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            outputs = model(features)
            loss_cls = F.cross_entropy(outputs["logits"], labels, weight=class_weights)
            activation_norm = row_minmax_tensor(outputs["concept_activations"])
            loss_align = F.mse_loss(activation_norm, class_profiles[labels])
            loss_cycl = profile_weighted_contrastive_loss(outputs["adapted_embeddings"], labels, class_profiles, tau=args.tau)
            loss = loss_cls + args.lambda_cycl * loss_cycl + args.lambda_align * loss_align
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_n = int(labels.numel())
            seen += batch_n
            running["total"] += float(loss.detach().cpu()) * batch_n
            running["cls"] += float(loss_cls.detach().cpu()) * batch_n
            running["cycl"] += float(loss_cycl.detach().cpu()) * batch_n
            running["align"] += float(loss_align.detach().cpu()) * batch_n
        train_loss = {key: value / max(1, seen) for key, value in running.items()}
        val_metrics = evaluate_model(model, split_payloads["val"], device=device, batch_size=args.batch_size)
        row = {
            "top_k": top_k,
            "seed": seed,
            "epoch": epoch,
            "train_total_loss": train_loss["total"],
            "train_cls_loss": train_loss["cls"],
            "train_cycl_loss": train_loss["cycl"],
            "train_align_loss": train_loss["align"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_macro_auroc": val_metrics["macro_auroc"],
        }
        curve_rows.append(row)
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = float(val_metrics["macro_f1"])
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= args.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    seed_dir = output_dir / "training" / f"top{top_k}" / f"seed_{seed}"
    ensure_dir(seed_dir)
    write_csv(seed_dir / "training_curve.csv", curve_rows)
    save_loss_curve(seed_dir / "loss_curve.png", curve_rows, title=f"top{top_k} seed {seed}")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "top_k": top_k,
            "seed": seed,
            "concepts": concept_embeddings.shape[0],
            "classes": CLASSES,
            "args": vars(args),
        },
        seed_dir / "best_checkpoint.pt",
    )
    val_metrics = evaluate_model(model, split_payloads["val"], device=device, batch_size=args.batch_size)
    test_metrics = evaluate_model(model, split_payloads["test"], device=device, batch_size=args.batch_size)
    save_square_matrix_csv(seed_dir / "confusion_matrix_test.csv", np.array(test_metrics["confusion_matrix"]), CLASSES)
    result: dict[str, Any] = {
        "top_k": top_k,
        "seed": seed,
        "n_concepts": int(concept_embeddings.shape[0]),
        "best_epoch": best_epoch,
        "device": str(device),
        "lambda_cycl": args.lambda_cycl,
        "lambda_align": args.lambda_align,
        "val_accuracy": val_metrics["accuracy"],
        "val_macro_f1": val_metrics["macro_f1"],
        "val_macro_auroc": val_metrics["macro_auroc"],
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_macro_auroc": test_metrics["macro_auroc"],
    }
    for class_name in CLASSES:
        result[f"test_{class_name}_precision"] = test_metrics[f"{class_name}_precision"]
        result[f"test_{class_name}_recall"] = test_metrics[f"{class_name}_recall"]
        result[f"test_{class_name}_f1"] = test_metrics[f"{class_name}_f1"]
    write_json(seed_dir / "metrics.json", result)
    return result


def save_loss_curve(path: Path, rows: list[dict[str, Any]], title: str) -> None:
    ensure_dir(path.parent)
    if not rows:
        return
    epochs = [int(row["epoch"]) for row in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    for key in ["train_total_loss", "train_cls_loss", "train_cycl_loss", "train_align_loss"]:
        ax.plot(epochs, [float(row[key]) for row in rows], label=key)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def summarize_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    df = pd.DataFrame(rows)
    summary_rows: list[dict[str, Any]] = []
    metrics = ["val_accuracy", "val_macro_f1", "val_macro_auroc", "test_accuracy", "test_macro_f1", "test_macro_auroc"]
    for top_k, group in df.groupby("top_k", sort=True):
        row: dict[str, Any] = {"top_k": int(top_k), "n_runs": int(len(group)), "n_concepts_mean": float(group["n_concepts"].mean())}
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary_rows.append(row)
    return summary_rows


def save_hardest_negative_analysis(score_df: pd.DataFrame, output_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    for target_class in CLASSES:
        class_df = score_df[score_df["target_class"] == target_class]
        for hard_class in CLASSES:
            if hard_class == target_class:
                continue
            subset = class_df[class_df["hardest_negative_class"] == hard_class]
            rows.append(
                {
                    "target_class": target_class,
                    "hardest_negative_class": hard_class,
                    "count": int(len(subset)),
                    "fraction": float(len(subset) / max(1, len(class_df))),
                    "mean_margin": float(subset["margin"].mean()) if len(subset) else "",
                }
            )
    write_csv(output_dir / "hardest_negative_summary.csv", rows)

    hgc_lgc = score_df[
        ((score_df["target_class"] == "HGC") & (score_df["hardest_negative_class"] == "LGC"))
        | ((score_df["target_class"] == "LGC") & (score_df["hardest_negative_class"] == "HGC"))
    ].sort_values(["target_class", "margin"], ascending=[True, False])
    hgc_lgc.head(100).to_csv(output_dir / "hardest_negative_HGC_LGC_examples.csv", index=False)


def save_architecture_diagram(path: Path) -> None:
    ensure_dir(path.parent)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis("off")
    boxes = [
        ("Input image", 0.05),
        ("Frozen BioMedCLIP\nimage encoder", 0.22),
        ("Adapter MLP\n(cystoscopy domain)", 0.42),
        ("Similarity to whitelist\nconcept text embeddings", 0.62),
        ("Concept activation\nvector K", 0.79),
        ("Linear classifier\nHGC/LGC/NTL/NST", 0.93),
    ]
    for text, x in boxes:
        ax.text(
            x,
            0.62,
            text,
            ha="center",
            va="center",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "#e8f1ff", "edgecolor": "#1f4e79"},
        )
    for (_, x1), (_, x2) in zip(boxes[:-1], boxes[1:]):
        ax.annotate("", xy=(x2 - 0.065, 0.62), xytext=(x1 + 0.065, 0.62), arrowprops={"arrowstyle": "->", "lw": 1.5})
    ax.text(0.73, 0.2, "L_cls: class prediction", ha="center", fontsize=10, bbox={"boxstyle": "round", "facecolor": "#f3f3f3"})
    ax.text(0.47, 0.2, "L_CyCL: pair weight = cosine(class profiles from M)", ha="center", fontsize=10, bbox={"boxstyle": "round", "facecolor": "#f3f3f3"})
    ax.text(0.28, 0.2, "L_align: activations close to class profile M[:, y]", ha="center", fontsize=10, bbox={"boxstyle": "round", "facecolor": "#f3f3f3"})
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_report(
    output_dir: Path,
    args: argparse.Namespace,
    score_df: pd.DataFrame,
    whitelist_summaries: list[dict[str, Any]],
    result_summary: list[dict[str, Any]],
) -> None:
    best = max(result_summary, key=lambda row: row["test_macro_f1_mean"]) if result_summary else None
    lines = [
        "# Discriminative Whitelist + Concept-Profile CyCL Report",
        "",
        "## Setup",
        "",
        f"- Source bank: `{args.bank_dir}`.",
        f"- Embeddings: `{args.embeddings_dir}`.",
        f"- Top-k values: `{args.top_ks}`.",
        f"- M normalization: `{args.m_normalization}`.",
        f"- Seeds: `{args.seeds}`.",
        f"- Device requested: `{args.device}`.",
        f"- Lambda CyCL: `{args.lambda_cycl}`.",
        f"- Lambda align: `{args.lambda_align}`.",
        "",
        "## Step 1: Discriminative Score",
        "",
        "For each concept, the score is `own-class mean - hardest-negative mean` over train image embeddings.",
        "The full table is `concept_discriminative_scores.csv`.",
        "",
        "## Step 2: Top-k Whitelists",
        "",
    ]
    for row in whitelist_summaries:
        lines.append(
            f"- top{row['top_k']}: before={row['selected_before_cleaning']}, "
            f"after_clean_dedup={row['n_concepts']}, dropped={row['dropped_count']}, "
            f"shared_concepts={row['shared_concepts']}."
        )
    lines.extend(
        [
            "",
            "Shared concepts are retained and represented with multiple selected target classes instead of being removed.",
            "",
            "## Step 3: M Matrix",
            "",
            "Each row of M is a whitelist concept and each column is HGC/LGC/NTL/NST.",
            "M is initialized from empirical class-wise mean image-text similarities and is not one-hot.",
            "See `M_matrix_top*_normalized.csv` and `M_matrix_top*_heatmap.png`.",
            "",
            "Class profile cosine matrices are saved as `class_profile_similarity_top*.csv/png`.",
            "",
            "## Step 4-5: Minimal Model",
            "",
            "The trained model is: cached BioMedCLIP image embedding -> adapter MLP -> similarity to whitelist concept embeddings -> concept activation vector -> linear classifier.",
            "",
            "Loss:",
            "",
            "`L = L_cls + lambda_cycl * L_CyCL + lambda_align * L_align`",
            "",
            "- `L_cls`: weighted cross-entropy.",
            "- `L_CyCL`: weighted contrastive loss using pair weights `cosine(M[:, y_i], M[:, y_j])`.",
            "- `L_align`: MSE between per-image normalized concept activations and the class profile from M.",
            "",
            "## Model Results",
            "",
        ]
    )
    for row in result_summary:
        lines.append(
            f"- top{row['top_k']}: test Macro-F1={row['test_macro_f1_mean']:.4f} ± {row['test_macro_f1_std']:.4f}, "
            f"test Acc={row['test_accuracy_mean']:.4f} ± {row['test_accuracy_std']:.4f}, "
            f"test AUROC={row['test_macro_auroc_mean']:.4f} ± {row['test_macro_auroc_std']:.4f}."
        )
    if best:
        lines.extend(
            [
                "",
                f"Best by mean test Macro-F1: top{best['top_k']} with {best['test_macro_f1_mean']:.4f}.",
            ]
        )
    hgc_to_lgc = int(
        len(score_df[(score_df["target_class"] == "HGC") & (score_df["hardest_negative_class"] == "LGC")])
    )
    lgc_to_hgc = int(
        len(score_df[(score_df["target_class"] == "LGC") & (score_df["hardest_negative_class"] == "HGC")])
    )
    lines.extend(
        [
            "",
            "## HGC/LGC Hardest-Negative Finding",
            "",
            f"- HGC concepts whose hardest negative is LGC: `{hgc_to_lgc}`.",
            f"- LGC concepts whose hardest negative is HGC: `{lgc_to_hgc}`.",
            "",
            "This directly supports the CyCL design choice that HGC/LGC should not always be treated as pure negatives; their class profiles can be partial positives through M.",
        ]
    )
    (output_dir / "experiment_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)
        try:
            torch.set_num_interop_threads(args.num_threads)
        except RuntimeError:
            pass
    ensure_dir(args.output_dir)
    top_ks = parse_int_list(args.top_ks)
    seeds = parse_int_list(args.seeds)
    device = resolve_device(args.device)

    write_json(
        args.output_dir / "experiment_config.json",
        {
            **vars(args),
            "bank_dir": str(args.bank_dir),
            "embeddings_dir": str(args.embeddings_dir),
            "output_dir": str(args.output_dir),
            "resolved_device": str(device),
            "cuda_available": torch.cuda.is_available(),
        },
    )
    split_payloads = load_split_payloads(args.embeddings_dir)
    bank = load_top300_bank(args.bank_dir)
    score_df, class_means = compute_discriminative_scores(bank, split_payloads["train"])
    score_df.to_csv(args.output_dir / "concept_discriminative_scores.csv", index=False)
    build_top_tables(score_df, top_ks, args.output_dir)
    save_hardest_negative_analysis(score_df, args.output_dir)
    save_architecture_diagram(args.output_dir / "model_architecture.png")

    whitelist_summaries: list[dict[str, Any]] = []
    train_results: list[dict[str, Any]] = []
    for top_k in top_ks:
        whitelist_df, concept_embeddings, _raw_m, m_matrix = build_whitelist(
            score_df=score_df,
            bank=bank,
            class_means=class_means,
            top_k=top_k,
            output_dir=args.output_dir,
            near_duplicate_threshold=args.near_duplicate_threshold,
            normalization=args.m_normalization,
            softmax_temperature=args.softmax_temperature,
        )
        dropped_path = args.output_dir / f"whitelist_top{top_k}_dropped.csv"
        dropped_count = max(0, sum(1 for _ in dropped_path.open(encoding="utf-8")) - 1) if dropped_path.exists() else 0
        whitelist_summaries.append(
            {
                "top_k": top_k,
                "selected_before_cleaning": top_k * len(CLASSES),
                "n_concepts": int(len(whitelist_df)),
                "dropped_count": int(dropped_count),
                "shared_concepts": int((whitelist_df["source_selection_count"] > 1).sum()) if len(whitelist_df) else 0,
            }
        )
        for seed in seeds:
            result = train_one_setting(
                split_payloads=split_payloads,
                concept_embeddings=concept_embeddings,
                m_matrix=m_matrix,
                top_k=top_k,
                seed=seed,
                args=args,
                output_dir=args.output_dir,
            )
            train_results.append(result)

    write_csv(args.output_dir / "whitelist_summary.csv", whitelist_summaries)
    write_csv(args.output_dir / "training_seed_results.csv", train_results)
    result_summary = summarize_results(train_results)
    write_csv(args.output_dir / "training_results_summary.csv", result_summary)
    write_report(args.output_dir, args, score_df, whitelist_summaries, result_summary)
    print(json.dumps({"output_dir": str(args.output_dir), "results": result_summary}, indent=2))


if __name__ == "__main__":
    main()
