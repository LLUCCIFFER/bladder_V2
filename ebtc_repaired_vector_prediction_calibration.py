#!/usr/bin/env python3
"""Prediction and conservative calibration diagnostics for the repaired bank.

This script checks two follow-up ideas on the repaired z_margin concept bank:

1. Hard -1 MLP prediction:
   image embedding -> 40-d repaired-bank concept vector.
2. Conservative residual calibration:
   repaired cosine vector -> cosine vector + scale * tanh(residual).

The raw+repaired fusion diagnostic keeps the previously selected fixed fusion
weight and fuses raw original_top10 class-count evidence with the repaired
source produced by each method.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ebtc_concept_vector_fusion import DEFAULT_REPAIRED_BANK_DIR, load_repaired_bank, safe_stem, score_to_probs
from ebtc_cosine_topk_assignment import (
    ConceptBank,
    compute_topk_tables,
    dataframe_to_markdown,
    load_concept_bank,
    load_stage_split_payloads,
    predict_class_average,
    predict_majority_vote,
)
from ebtc_discriminative_whitelist_cycl import CLASSES, SplitPayload
from ebtc_project_paths import OFFICIAL_EMBEDDINGS_DIR, OUTPUT_ROOT


DEFAULT_REFINED_STAGE_DIR = OUTPUT_ROOT / "ebtc_embedding_refinement_stage_conservative_outputs"
DEFAULT_OUTPUT_DIR = OUTPUT_ROOT / "ebtc_repaired_vector_prediction_calibration_outputs"
CONCEPTS_PER_CLASS = 10
EPS = 1e-8


@dataclass
class SplitPrepared:
    raw_original_vectors: np.ndarray
    repaired_vectors: np.ndarray
    hard_targets: np.ndarray
    hard_weights: np.ndarray
    labels: np.ndarray


@dataclass
class TrainResult:
    source: str
    family: str
    seed: int
    best_epoch: int
    best_val_macro_f1: float
    best_val_accuracy: float
    best_val_loss: float
    predictions: dict[str, np.ndarray]


class HardVectorPredictor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class ResidualCalibrator(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float, residual_scale: float) -> None:
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, raw_vectors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual = self.residual_scale * torch.tanh(self.net(raw_vectors))
        return torch.clamp(raw_vectors + residual, min=-1.0, max=1.0), residual


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run repaired-bank hard prediction and conservative calibration diagnostics.")
    parser.add_argument("--refined-stage-dir", type=Path, default=DEFAULT_REFINED_STAGE_DIR)
    parser.add_argument("--embeddings-dir", type=Path, default=OFFICIAL_EMBEDDINGS_DIR)
    parser.add_argument("--repaired-bank-dir", type=Path, default=DEFAULT_REPAIRED_BANK_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--fusion-weight", type=float, default=0.85)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--hard-positive-weights", default="1.0,3.0")
    parser.add_argument("--residual-scales", default="0.01,0.02")
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--target-negative", type=float, default=-1.0)
    parser.add_argument("--residual-logit-scale", type=float, default=10.0)
    parser.add_argument("--residual-block-topk", type=int, default=3)
    parser.add_argument("--lambda-rank", type=float, default=0.5)
    parser.add_argument("--lambda-preserve", type=float, default=10.0)
    parser.add_argument("--lambda-residual", type=float, default=1.0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--num-threads", type=int, default=2)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def validate_bank_layout(bank: ConceptBank) -> None:
    expected = np.repeat(np.arange(len(CLASSES), dtype=np.int64), CONCEPTS_PER_CLASS)
    if bank.labels.shape[0] != expected.shape[0] or not np.array_equal(bank.labels, expected):
        raise RuntimeError(
            "Expected a 40-concept class-block bank ordered as HGC/LGC/NTL/NST, 10 concepts per class."
        )


def build_hard_targets(
    raw_vectors: np.ndarray,
    labels: np.ndarray,
    concept_labels: np.ndarray,
    negative_value: float,
    positive_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    targets = np.full_like(raw_vectors, fill_value=float(negative_value), dtype=np.float32)
    weights = np.ones_like(raw_vectors, dtype=np.float32)
    for class_idx in range(len(CLASSES)):
        rows = labels == class_idx
        cols = concept_labels == class_idx
        if bool(rows.any()):
            targets[np.ix_(rows, cols)] = raw_vectors[np.ix_(rows, cols)]
            weights[np.ix_(rows, cols)] = float(positive_weight)
    return targets.astype(np.float32), weights.astype(np.float32)


def prepare_splits(
    payloads: dict[str, SplitPayload],
    raw_bank: ConceptBank,
    repaired_bank: ConceptBank,
    target_negative: float,
    positive_weight: float,
) -> dict[str, SplitPrepared]:
    out: dict[str, SplitPrepared] = {}
    for split, payload in payloads.items():
        raw_original_vectors = (payload.embeddings @ raw_bank.embeddings.T).astype(np.float32)
        repaired_vectors = (payload.embeddings @ repaired_bank.embeddings.T).astype(np.float32)
        targets, weights = build_hard_targets(
            raw_vectors=repaired_vectors,
            labels=payload.labels,
            concept_labels=repaired_bank.labels,
            negative_value=target_negative,
            positive_weight=positive_weight,
        )
        out[split] = SplitPrepared(
            raw_original_vectors=raw_original_vectors,
            repaired_vectors=repaired_vectors,
            hard_targets=targets,
            hard_weights=weights,
            labels=payload.labels,
        )
    return out


def weighted_mse(predictions: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return ((predictions - targets).pow(2) * weights).mean()


def class_block_logits(vectors: torch.Tensor, concept_labels: torch.Tensor, block_topk: int) -> torch.Tensor:
    logits: list[torch.Tensor] = []
    for class_idx in range(len(CLASSES)):
        block = vectors[:, concept_labels == class_idx]
        k = min(block_topk, int(block.shape[1]))
        logits.append(torch.topk(block, k=k, dim=1).values.mean(dim=1))
    return torch.stack(logits, dim=1)


def residual_loss(
    corrected: torch.Tensor,
    residual: torch.Tensor,
    raw_vectors: torch.Tensor,
    labels: torch.Tensor,
    concept_labels: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    logits = class_block_logits(corrected, concept_labels, args.residual_block_topk)
    scaled_logits = logits * float(args.residual_logit_scale)
    ce_loss = F.cross_entropy(scaled_logits, labels)

    true_logits = logits.gather(1, labels[:, None]).squeeze(1)
    other_logits = logits.masked_fill(F.one_hot(labels, len(CLASSES)).bool(), -1e4)
    rank_loss = F.relu(0.05 - (true_logits[:, None] - other_logits)).mean()

    true_mask = concept_labels[None, :] == labels[:, None]
    preserve_loss = (corrected[true_mask] - raw_vectors[true_mask]).pow(2).mean()
    residual_penalty = residual.pow(2).mean()
    return ce_loss + args.lambda_rank * rank_loss + args.lambda_preserve * preserve_loss + args.lambda_residual * residual_penalty


def make_hard_loader(
    payload: SplitPayload,
    prepared: SplitPrepared,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(payload.embeddings, dtype=torch.float32),
        torch.tensor(prepared.hard_targets, dtype=torch.float32),
        torch.tensor(prepared.hard_weights, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def make_residual_loader(prepared: SplitPrepared, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(prepared.repaired_vectors, dtype=torch.float32),
        torch.tensor(prepared.labels, dtype=torch.long),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def predict_hard_vectors(
    model: HardVectorPredictor,
    payload: SplitPayload,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    parts: list[np.ndarray] = []
    loader = DataLoader(torch.tensor(payload.embeddings, dtype=torch.float32), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for features in loader:
            parts.append(model(features.to(device)).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(parts, axis=0)


def predict_residual_vectors(
    model: ResidualCalibrator,
    prepared: SplitPrepared,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    parts: list[np.ndarray] = []
    loader = DataLoader(torch.tensor(prepared.repaired_vectors, dtype=torch.float32), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for raw_vectors in loader:
            corrected, _residual = model(raw_vectors.to(device))
            parts.append(corrected.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(parts, axis=0)


def evaluate_vector_metrics(vectors: np.ndarray, labels: np.ndarray, bank: ConceptBank, top_k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _top_indices, counts, mean_scores, mean_scores_for_auc = compute_topk_tables(vectors, bank.labels, top_k=top_k)
    majority_preds = predict_majority_vote(counts, mean_scores)
    average_preds = predict_class_average(mean_scores)
    return counts, mean_scores, mean_scores_for_auc, majority_preds, average_preds


def vector_val_score(vectors: np.ndarray, labels: np.ndarray, bank: ConceptBank, top_k: int) -> tuple[float, float]:
    _counts, _mean_scores, _mean_scores_for_auc, preds, _average_preds = evaluate_vector_metrics(vectors, labels, bank, top_k)
    return (
        float(f1_score(labels, preds, average="macro", zero_division=0)),
        float(accuracy_score(labels, preds)),
    )


def evaluate_hard_loss(
    model: HardVectorPredictor,
    payload: SplitPayload,
    prepared: SplitPrepared,
    device: torch.device,
    batch_size: int,
) -> float:
    model.eval()
    loader = make_hard_loader(payload, prepared, batch_size=batch_size, shuffle=False)
    total = 0.0
    seen = 0
    with torch.no_grad():
        for features, targets, weights in loader:
            features = features.to(device)
            targets = targets.to(device)
            weights = weights.to(device)
            loss = weighted_mse(model(features), targets, weights)
            n = int(features.shape[0])
            total += float(loss.detach().cpu()) * n
            seen += n
    return total / max(1, seen)


def evaluate_residual_loss(
    model: ResidualCalibrator,
    prepared: SplitPrepared,
    concept_labels: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> float:
    model.eval()
    loader = make_residual_loader(prepared, batch_size=args.batch_size, shuffle=False)
    total = 0.0
    seen = 0
    with torch.no_grad():
        for raw_vectors, labels in loader:
            raw_vectors = raw_vectors.to(device)
            labels = labels.to(device)
            corrected, residual = model(raw_vectors)
            loss = residual_loss(corrected, residual, raw_vectors, labels, concept_labels, args)
            n = int(raw_vectors.shape[0])
            total += float(loss.detach().cpu()) * n
            seen += n
    return total / max(1, seen)


def train_hard_seed(
    seed: int,
    positive_weight: float,
    args: argparse.Namespace,
    payloads: dict[str, SplitPayload],
    prepared: dict[str, SplitPrepared],
    bank: ConceptBank,
    device: torch.device,
) -> TrainResult:
    set_seed(seed)
    model = HardVectorPredictor(
        input_dim=payloads["train"].embeddings.shape[1],
        output_dim=len(bank.labels),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = make_hard_loader(payloads["train"], prepared["train"], args.batch_size, shuffle=True)

    best_state: dict[str, torch.Tensor] | None = None
    best_val_macro = -1.0
    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    curve_rows: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        seen = 0
        for features, targets, weights in train_loader:
            features = features.to(device)
            targets = targets.to(device)
            weights = weights.to(device)
            loss = weighted_mse(model(features), targets, weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            n = int(features.shape[0])
            total += float(loss.detach().cpu()) * n
            seen += n

        val_vectors = predict_hard_vectors(model, payloads["val"], device, args.batch_size)
        val_macro, val_acc = vector_val_score(val_vectors, prepared["val"].labels, bank, args.top_k)
        val_loss = evaluate_hard_loss(model, payloads["val"], prepared["val"], device, args.batch_size)
        train_loss = total / max(1, seen)
        curve_rows.append(
            {
                "seed": seed,
                "positive_weight": positive_weight,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_macro_f1": val_macro,
                "val_accuracy": val_acc,
            }
        )
        improved = (val_macro > best_val_macro + 1e-8) or (
            abs(val_macro - best_val_macro) <= 1e-8 and val_acc > best_val_acc + 1e-8
        )
        if improved:
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_val_macro = val_macro
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= args.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    source = f"hard_mlp_pw{positive_weight:g}_seed_{seed}"
    train_dir = args.output_dir / "training" / safe_stem(source)
    ensure_dir(train_dir)
    pd.DataFrame(curve_rows).to_csv(train_dir / "training_curve.csv", index=False)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "seed": seed,
            "positive_weight": positive_weight,
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_val_macro,
            "best_val_accuracy": best_val_acc,
            "classes": CLASSES,
        },
        train_dir / "best_checkpoint.pt",
    )

    predictions = {split: predict_hard_vectors(model, payload, device, args.batch_size) for split, payload in payloads.items()}
    return TrainResult(
        source=source,
        family=f"hard_mlp_pw{positive_weight:g}",
        seed=seed,
        best_epoch=best_epoch,
        best_val_macro_f1=best_val_macro,
        best_val_accuracy=best_val_acc,
        best_val_loss=best_val_loss,
        predictions=predictions,
    )


def train_residual_seed(
    seed: int,
    residual_scale: float,
    args: argparse.Namespace,
    prepared: dict[str, SplitPrepared],
    bank: ConceptBank,
    device: torch.device,
) -> TrainResult:
    set_seed(seed)
    model = ResidualCalibrator(
        input_dim=len(bank.labels),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        residual_scale=residual_scale,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = make_residual_loader(prepared["train"], args.batch_size, shuffle=True)
    concept_labels = torch.tensor(bank.labels, dtype=torch.long, device=device)

    best_state: dict[str, torch.Tensor] | None = None
    best_val_macro = -1.0
    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    curve_rows: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        seen = 0
        for raw_vectors, labels in train_loader:
            raw_vectors = raw_vectors.to(device)
            labels = labels.to(device)
            corrected, residual = model(raw_vectors)
            loss = residual_loss(corrected, residual, raw_vectors, labels, concept_labels, args)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            n = int(raw_vectors.shape[0])
            total += float(loss.detach().cpu()) * n
            seen += n

        val_vectors = predict_residual_vectors(model, prepared["val"], device, args.batch_size)
        val_macro, val_acc = vector_val_score(val_vectors, prepared["val"].labels, bank, args.top_k)
        val_loss = evaluate_residual_loss(model, prepared["val"], concept_labels, args, device)
        train_loss = total / max(1, seen)
        curve_rows.append(
            {
                "seed": seed,
                "residual_scale": residual_scale,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_macro_f1": val_macro,
                "val_accuracy": val_acc,
            }
        )
        improved = (val_macro > best_val_macro + 1e-8) or (
            abs(val_macro - best_val_macro) <= 1e-8 and val_acc > best_val_acc + 1e-8
        )
        if improved:
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_val_macro = val_macro
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= args.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    source = f"residual_s{residual_scale:g}_seed_{seed}"
    train_dir = args.output_dir / "training" / safe_stem(source)
    ensure_dir(train_dir)
    pd.DataFrame(curve_rows).to_csv(train_dir / "training_curve.csv", index=False)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "seed": seed,
            "residual_scale": residual_scale,
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_val_macro,
            "best_val_accuracy": best_val_acc,
            "classes": CLASSES,
        },
        train_dir / "best_checkpoint.pt",
    )

    predictions = {split: predict_residual_vectors(model, data, device, args.batch_size) for split, data in prepared.items()}
    return TrainResult(
        source=source,
        family=f"residual_s{residual_scale:g}",
        seed=seed,
        best_epoch=best_epoch,
        best_val_macro_f1=best_val_macro,
        best_val_accuracy=best_val_acc,
        best_val_loss=best_val_loss,
        predictions=predictions,
    )


def metric_row(
    source: str,
    split: str,
    rule: str,
    labels: np.ndarray,
    preds: np.ndarray,
    scores: np.ndarray,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "source": source,
        "split": split,
        "rule": rule,
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
    }
    try:
        if scores.shape[1] == len(CLASSES) and np.all(scores >= 0):
            probs = score_to_probs(scores)
        else:
            stable = scores - np.max(scores, axis=1, keepdims=True)
            exp_scores = np.exp(stable)
            probs = exp_scores / np.clip(exp_scores.sum(axis=1, keepdims=True), EPS, None)
        row["macro_auroc"] = float(roc_auc_score(labels, probs, multi_class="ovr", average="macro", labels=list(range(len(CLASSES)))))
    except ValueError:
        row["macro_auroc"] = ""
    precision, recall, f1, support = precision_recall_fscore_support(labels, preds, labels=list(range(len(CLASSES))), zero_division=0)
    for class_idx, class_name in enumerate(CLASSES):
        row[f"{class_name}_precision"] = float(precision[class_idx])
        row[f"{class_name}_recall"] = float(recall[class_idx])
        row[f"{class_name}_f1"] = float(f1[class_idx])
        row[f"{class_name}_support"] = int(support[class_idx])
    return row


def save_confusion(output_dir: Path, source: str, split: str, rule: str, labels: np.ndarray, preds: np.ndarray) -> None:
    out_dir = output_dir / "confusion_matrices"
    ensure_dir(out_dir)
    matrix = confusion_matrix(labels, preds, labels=list(range(len(CLASSES))))
    base = out_dir / f"confusion_{safe_stem(source)}_{safe_stem(split)}_{safe_stem(rule)}"
    pd.DataFrame(matrix, index=CLASSES, columns=CLASSES).to_csv(base.with_suffix(".csv"))
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(CLASSES)), labels=CLASSES)
    ax.set_yticks(range(len(CLASSES)), labels=CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{source} {split} {rule}")
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            ax.text(col_idx, row_idx, str(int(matrix[row_idx, col_idx])), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(base.with_suffix(".png"), dpi=180)
    plt.close(fig)


def save_count_summary(output_dir: Path, source: str, split: str, labels: np.ndarray, scores: np.ndarray) -> None:
    out_dir = output_dir / "score_summaries"
    ensure_dir(out_dir)
    rows: list[dict[str, Any]] = []
    for true_idx, true_class in enumerate(CLASSES):
        mask = labels == true_idx
        means = scores[mask].mean(axis=0) if bool(mask.any()) else np.zeros(len(CLASSES), dtype=np.float32)
        row = {"source": source, "split": split, "true_image_class": true_class}
        for class_idx, class_name in enumerate(CLASSES):
            row[f"mean_score_{class_name}"] = float(means[class_idx])
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / f"scores_{safe_stem(source)}_{safe_stem(split)}.csv", index=False)


def evaluate_repaired_source(
    output_dir: Path,
    source: str,
    vectors_by_split: dict[str, np.ndarray],
    prepared: dict[str, SplitPrepared],
    bank: ConceptBank,
    top_k: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, vectors in vectors_by_split.items():
        labels = prepared[split].labels
        counts, _mean_scores, mean_scores_for_auc, majority_preds, average_preds = evaluate_vector_metrics(vectors, labels, bank, top_k)
        for rule, preds, scores in [
            ("majority_vote", majority_preds, counts),
            ("class_average", average_preds, mean_scores_for_auc),
        ]:
            rows.append(metric_row(source, split, rule, labels, preds, scores))
            save_confusion(output_dir, source, split, rule, labels, preds)
        save_count_summary(output_dir, f"{source}_topk_counts", split, labels, counts)
    return rows


def counts_for_vectors(vectors: np.ndarray, bank: ConceptBank, top_k: int) -> np.ndarray:
    _top_indices, counts, _mean_scores, _mean_scores_for_auc = compute_topk_tables(vectors, bank.labels, top_k=top_k)
    return counts


def evaluate_fusion_source(
    output_dir: Path,
    source: str,
    repaired_vectors_by_split: dict[str, np.ndarray],
    prepared: dict[str, SplitPrepared],
    raw_bank: ConceptBank,
    repaired_bank: ConceptBank,
    top_k: int,
    fusion_weight: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, repaired_vectors in repaired_vectors_by_split.items():
        labels = prepared[split].labels
        raw_counts = counts_for_vectors(prepared[split].raw_original_vectors, raw_bank, top_k)
        repaired_counts = counts_for_vectors(repaired_vectors, repaired_bank, top_k)
        scores = (1.0 - fusion_weight) * raw_counts + fusion_weight * repaired_counts
        preds = scores.argmax(axis=1).astype(np.int64)
        rows.append(metric_row(source, split, "fusion_majority_vote", labels, preds, scores))
        save_confusion(output_dir, source, split, "fusion_majority_vote", labels, preds)
        save_count_summary(output_dir, source, split, labels, scores)
    return rows


def add_source_evaluations(
    output_dir: Path,
    source: str,
    vectors_by_split: dict[str, np.ndarray],
    prepared: dict[str, SplitPrepared],
    raw_bank: ConceptBank,
    repaired_bank: ConceptBank,
    args: argparse.Namespace,
    result_rows: list[dict[str, Any]],
) -> None:
    result_rows.extend(evaluate_repaired_source(output_dir, source, vectors_by_split, prepared, repaired_bank, args.top_k))
    result_rows.extend(
        evaluate_fusion_source(
            output_dir=output_dir,
            source=f"fusion_raw_plus_{source}",
            repaired_vectors_by_split=vectors_by_split,
            prepared=prepared,
            raw_bank=raw_bank,
            repaired_bank=repaired_bank,
            top_k=args.top_k,
            fusion_weight=args.fusion_weight,
        )
    )


def ensemble_predictions(results: list[TrainResult], family: str) -> dict[str, np.ndarray]:
    family_results = [item for item in results if item.family == family]
    if not family_results:
        raise RuntimeError(f"No seed predictions for family {family}")
    splits = family_results[0].predictions.keys()
    return {
        split: np.mean([item.predictions[split] for item in family_results], axis=0).astype(np.float32)
        for split in splits
    }


def write_report(
    output_dir: Path,
    args: argparse.Namespace,
    training: pd.DataFrame,
    results: pd.DataFrame,
) -> None:
    metric_cols = {"accuracy", "macro_f1", "macro_auroc"}
    test_cols = ["source", "rule", "accuracy", "macro_f1", "macro_auroc", "HGC_f1", "LGC_f1", "NTL_f1", "NST_f1"]
    summary_sources = [
        "direct_repaired_cosine",
        "fusion_raw_plus_direct_repaired_cosine",
    ]
    summary_sources.extend(sorted(source for source in results["source"].unique() if source.endswith("_ensemble")))
    summary_sources.extend(
        sorted(
            source
            for source in results["source"].unique()
            if source.startswith("fusion_raw_plus_") and source.endswith("_ensemble")
        )
    )
    test_summary = results[(results["split"].eq("test")) & (results["source"].isin(summary_sources))].copy()
    val_top = results[results["split"].eq("val")].sort_values(["macro_f1", "accuracy"], ascending=False).head(16)
    test_top = results[results["split"].eq("test")].sort_values(["macro_f1", "accuracy"], ascending=False).head(16)

    lines = [
        "# Repaired Vector Prediction and Calibration",
        "",
        "## Setup",
        "",
        f"- Repaired bank: `{args.repaired_bank_dir}`.",
        f"- Fixed raw+repaired fusion weight: `{args.fusion_weight}`.",
        f"- Hard prediction target: true-class repaired-bank cosine values; all other positions set to `{args.target_negative}`.",
        f"- Conservative residual form: `corrected = raw + scale * tanh(MLP(raw))`.",
        f"- Residual scales: `{args.residual_scales}`.",
        f"- Seeds: `{args.seeds}`.",
        "",
        "## Training Summary",
        "",
        dataframe_to_markdown(
            training[["source", "family", "seed", "best_epoch", "best_val_macro_f1", "best_val_accuracy", "best_val_loss"]],
            {"best_val_macro_f1", "best_val_accuracy", "best_val_loss"},
        ),
        "",
        "## Main Frozen-Test Rows",
        "",
        dataframe_to_markdown(test_summary[test_cols], metric_cols | set(CLASSES)),
        "",
        "## Top Validation Rows",
        "",
        dataframe_to_markdown(val_top[test_cols], metric_cols | set(CLASSES)),
        "",
        "## Top Test Rows",
        "",
        dataframe_to_markdown(test_top[test_cols], metric_cols | set(CLASSES)),
        "",
        "## Output Files",
        "",
        f"- Metrics: `{output_dir / 'repaired_vector_prediction_calibration_results.csv'}`",
        f"- Training summary: `{output_dir / 'training_summary.csv'}`",
        f"- Confusion matrices: `{output_dir / 'confusion_matrices'}`",
        f"- Score summaries: `{output_dir / 'score_summaries'}`",
    ]
    (output_dir / "repaired_vector_prediction_calibration_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)
    ensure_dir(args.output_dir)
    device = resolve_device(args.device)

    seeds = parse_int_list(args.seeds)
    positive_weights = parse_float_list(args.hard_positive_weights)
    residual_scales = parse_float_list(args.residual_scales)

    payloads = load_stage_split_payloads(args.refined_stage_dir, args.embeddings_dir, "refined_vectors_original_top10")
    raw_bank = load_concept_bank(args.refined_stage_dir, "refined_vectors_original_top10")
    repaired_bank = load_repaired_bank(args.repaired_bank_dir)
    validate_bank_layout(raw_bank)
    validate_bank_layout(repaired_bank)

    write_json(
        args.output_dir / "experiment_config.json",
        {
            "refined_stage_dir": str(args.refined_stage_dir),
            "embeddings_dir": str(args.embeddings_dir),
            "repaired_bank_dir": str(args.repaired_bank_dir),
            "output_dir": str(args.output_dir),
            "top_k": args.top_k,
            "fusion_weight": args.fusion_weight,
            "seeds": seeds,
            "hard_positive_weights": positive_weights,
            "residual_scales": residual_scales,
            "epochs": args.epochs,
            "patience": args.patience,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "target_negative": args.target_negative,
            "device": str(device),
        },
    )

    result_rows: list[dict[str, Any]] = []
    training_results: list[TrainResult] = []
    training_rows: list[dict[str, Any]] = []

    base_prepared = prepare_splits(payloads, raw_bank, repaired_bank, args.target_negative, positive_weight=1.0)
    direct_vectors = {split: data.repaired_vectors for split, data in base_prepared.items()}
    add_source_evaluations(args.output_dir, "direct_repaired_cosine", direct_vectors, base_prepared, raw_bank, repaired_bank, args, result_rows)

    for positive_weight in positive_weights:
        prepared = prepare_splits(payloads, raw_bank, repaired_bank, args.target_negative, positive_weight=positive_weight)
        family = f"hard_mlp_pw{positive_weight:g}"
        for seed in seeds:
            train_result = train_hard_seed(seed, positive_weight, args, payloads, prepared, repaired_bank, device)
            training_results.append(train_result)
            training_rows.append(
                {
                    "source": train_result.source,
                    "family": train_result.family,
                    "seed": train_result.seed,
                    "best_epoch": train_result.best_epoch,
                    "best_val_macro_f1": train_result.best_val_macro_f1,
                    "best_val_accuracy": train_result.best_val_accuracy,
                    "best_val_loss": train_result.best_val_loss,
                }
            )
            add_source_evaluations(args.output_dir, train_result.source, train_result.predictions, prepared, raw_bank, repaired_bank, args, result_rows)
        ensemble = ensemble_predictions(training_results, family)
        add_source_evaluations(args.output_dir, f"{family}_ensemble", ensemble, prepared, raw_bank, repaired_bank, args, result_rows)

    for residual_scale in residual_scales:
        family = f"residual_s{residual_scale:g}"
        for seed in seeds:
            train_result = train_residual_seed(seed, residual_scale, args, base_prepared, repaired_bank, device)
            training_results.append(train_result)
            training_rows.append(
                {
                    "source": train_result.source,
                    "family": train_result.family,
                    "seed": train_result.seed,
                    "best_epoch": train_result.best_epoch,
                    "best_val_macro_f1": train_result.best_val_macro_f1,
                    "best_val_accuracy": train_result.best_val_accuracy,
                    "best_val_loss": train_result.best_val_loss,
                }
            )
            add_source_evaluations(args.output_dir, train_result.source, train_result.predictions, base_prepared, raw_bank, repaired_bank, args, result_rows)
        ensemble = ensemble_predictions(training_results, family)
        add_source_evaluations(args.output_dir, f"{family}_ensemble", ensemble, base_prepared, raw_bank, repaired_bank, args, result_rows)

    training = pd.DataFrame(training_rows)
    results = pd.DataFrame(result_rows)
    training.to_csv(args.output_dir / "training_summary.csv", index=False)
    results.to_csv(args.output_dir / "repaired_vector_prediction_calibration_results.csv", index=False)
    write_report(args.output_dir, args, training, results)
    print(json.dumps({"output_dir": str(args.output_dir), "rows": int(len(results))}, indent=2))


if __name__ == "__main__":
    main()
