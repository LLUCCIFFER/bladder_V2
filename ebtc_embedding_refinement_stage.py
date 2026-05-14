#!/usr/bin/env python3
"""Image/text embedding refinement before EBTC whitelist-CBM training.

This stage keeps the existing filtered_top300 concept bank fixed, trains
lightweight residual MLP adapters on cached BioMedCLIP image/text embeddings,
then recomputes concept similarities, whitelist concepts, and CBM results.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ebtc_discriminative_whitelist_cycl import (
    BankPayload,
    CLASSES,
    CLASS_TO_INDEX,
    EPS,
    SplitPayload,
    build_top_tables,
    build_whitelist,
    compute_class_means,
    compute_discriminative_scores,
    load_split_payloads,
    load_top300_bank,
    normalize_m,
    parse_int_list,
    save_hardest_negative_analysis,
    summarize_results,
    train_one_setting,
    write_csv,
    write_json,
)
from ebtc_project_paths import FILTERED_TOP300_BANK_DIR, OFFICIAL_EMBEDDINGS_DIR, OUTPUT_ROOT


DEFAULT_OUTPUT_DIR = OUTPUT_ROOT / "ebtc_embedding_refinement_stage_outputs"


@dataclass
class RefinementPayload:
    split_payloads: dict[str, SplitPayload]
    text_embeddings: np.ndarray
    concept_labels: np.ndarray


class ResidualAdapter(nn.Module):
    """Small residual adapter that preserves the original embedding space."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.normalize(inputs + self.net(inputs), dim=-1)


class ImageTextRefinementModel(nn.Module):
    """Residual MLP adapters for frozen BioMedCLIP image/text embeddings."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.image_adapter = ResidualAdapter(dim, hidden_dim, dropout)
        self.text_adapter = ResidualAdapter(dim, hidden_dim, dropout)

    def encode_images(self, image_embeddings: torch.Tensor) -> torch.Tensor:
        return self.image_adapter(image_embeddings)

    def encode_texts(self, text_embeddings: torch.Tensor) -> torch.Tensor:
        return self.text_adapter(text_embeddings)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train image/text adapters and rerun EBTC whitelist-CBM.")
    parser.add_argument("--bank-dir", type=Path, default=FILTERED_TOP300_BANK_DIR)
    parser.add_argument("--embeddings-dir", type=Path, default=OFFICIAL_EMBEDDINGS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--num-threads", type=int, default=2)

    parser.add_argument("--adapter-hidden-dim", type=int, default=256)
    parser.add_argument("--adapter-dropout", type=float, default=0.1)
    parser.add_argument("--refine-epochs", type=int, default=20)
    parser.add_argument("--refine-patience", type=int, default=5)
    parser.add_argument("--refine-batch-size", type=int, default=128)
    parser.add_argument("--refine-lr", type=float, default=5e-4)
    parser.add_argument("--refine-weight-decay", type=float, default=1e-4)
    parser.add_argument("--refine-tau", type=float, default=0.07)
    parser.add_argument("--lambda-t2i", type=float, default=0.5)
    parser.add_argument("--refine-seed", type=int, default=42)
    parser.add_argument("--disable-balanced-refine-sampler", action="store_true")

    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.995)
    parser.add_argument("--m-normalization", choices=["minmax", "softmax"], default="minmax")
    parser.add_argument("--softmax-temperature", type=float, default=0.02)
    parser.add_argument("--retrieval-k", type=int, default=10)

    parser.add_argument("--run-cbm", action="store_true", help="Run original-control and refined CBM comparisons.")
    parser.add_argument("--cbm-seeds", default="42,43,44")
    parser.add_argument("--cbm-epochs", type=int, default=40)
    parser.add_argument("--cbm-patience", type=int, default=8)
    parser.add_argument("--cbm-batch-size", type=int, default=128)
    parser.add_argument("--cbm-hidden-dim", type=int, default=128)
    parser.add_argument("--cbm-dropout", type=float, default=0.1)
    parser.add_argument("--cbm-lr", type=float, default=1e-3)
    parser.add_argument("--cbm-weight-decay", type=float, default=1e-4)
    parser.add_argument("--cbm-lambda-cycl", type=float, default=0.0)
    parser.add_argument("--cbm-lambda-align", type=float, default=0.05)
    parser.add_argument("--cbm-tau", type=float, default=0.1)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def labels_from_bank(bank: BankPayload) -> np.ndarray:
    return np.array([CLASS_TO_INDEX[str(item)] for item in bank.rows["primary_class"].tolist()], dtype=np.int64)


def multipositive_nce(logits: torch.Tensor, positive_mask: torch.Tensor) -> torch.Tensor:
    """Multi-positive InfoNCE over rows of logits."""

    valid = positive_mask.any(dim=1)
    if not bool(valid.any()):
        return logits.new_tensor(0.0)
    logits_valid = logits[valid]
    mask_valid = positive_mask[valid]
    positive_logits = logits_valid.masked_fill(~mask_valid, -1e9)
    return -(torch.logsumexp(positive_logits, dim=1) - torch.logsumexp(logits_valid, dim=1)).mean()


def make_refine_loader(split_payload: SplitPayload, batch_size: int, balanced: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(split_payload.embeddings, dtype=torch.float32),
        torch.tensor(split_payload.labels, dtype=torch.long),
    )
    if not balanced:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    counts = np.bincount(split_payload.labels, minlength=len(CLASSES)).astype(np.float32)
    sample_weights = 1.0 / np.clip(counts[split_payload.labels], EPS, None)
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


def compute_class_similarity_matrix(
    image_embeddings: np.ndarray,
    image_labels: np.ndarray,
    text_embeddings: np.ndarray,
    concept_labels: np.ndarray,
) -> np.ndarray:
    sims = image_embeddings @ text_embeddings.T
    matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.float32)
    for image_class_idx in range(len(CLASSES)):
        image_mask = image_labels == image_class_idx
        for concept_class_idx in range(len(CLASSES)):
            concept_mask = concept_labels == concept_class_idx
            matrix[image_class_idx, concept_class_idx] = float(sims[np.ix_(image_mask, concept_mask)].mean())
    return matrix


def matrix_summary(stage: str, split: str, matrix: np.ndarray) -> dict[str, Any]:
    diag = np.diag(matrix)
    off_mask = ~np.eye(len(CLASSES), dtype=bool)
    off = matrix[off_mask]
    row: dict[str, Any] = {
        "stage": stage,
        "split": split,
        "diagonal_mean": float(diag.mean()),
        "off_diagonal_mean": float(off.mean()),
        "diag_minus_offdiag": float(diag.mean() - off.mean()),
    }
    for idx, class_name in enumerate(CLASSES):
        row[f"{class_name}_row_margin"] = float(matrix[idx, idx] - np.max(np.delete(matrix[idx], idx)))
        row[f"{class_name}_col_margin"] = float(matrix[idx, idx] - np.mean(np.delete(matrix[:, idx], idx)))
    return row


def save_matrix_csv(path: Path, matrix: np.ndarray) -> None:
    rows: list[dict[str, Any]] = []
    for idx, class_name in enumerate(CLASSES):
        row = {"image_class": class_name}
        for col_idx, concept_class in enumerate(CLASSES):
            row[concept_class] = float(matrix[idx, col_idx])
        rows.append(row)
    write_csv(path, rows)


def save_heatmap(path: Path, matrix: np.ndarray, title: str) -> None:
    ensure_dir(path.parent)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="viridis")
    ax.set_xticks(range(len(CLASSES)), labels=CLASSES)
    ax.set_yticks(range(len(CLASSES)), labels=CLASSES)
    ax.set_xlabel("Concept class")
    ax.set_ylabel("Image class")
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", color="white")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def evaluate_majority_vote(
    image_embeddings: np.ndarray,
    labels: np.ndarray,
    text_embeddings: np.ndarray,
    concept_labels: np.ndarray,
    top_k: int,
) -> dict[str, Any]:
    sims = image_embeddings @ text_embeddings.T
    top_indices = np.argpartition(-sims, kth=min(top_k - 1, sims.shape[1] - 1), axis=1)[:, :top_k]
    preds: list[int] = []
    margins: list[int] = []
    for row in top_indices:
        counts = np.bincount(concept_labels[row], minlength=len(CLASSES))
        order = np.argsort(-counts)
        preds.append(int(order[0]))
        margins.append(int(counts[order[0]] - counts[order[1]]))
    pred_array = np.array(preds, dtype=np.int64)
    probs = np.zeros((len(labels), len(CLASSES)), dtype=np.float32)
    for idx, row in enumerate(top_indices):
        counts = np.bincount(concept_labels[row], minlength=len(CLASSES)).astype(np.float32)
        probs[idx] = counts / max(1, top_k)
    result: dict[str, Any] = {
        "accuracy": float(accuracy_score(labels, pred_array)),
        "macro_f1": float(f1_score(labels, pred_array, average="macro", zero_division=0)),
        "top_k": top_k,
        "mean_vote_margin": float(np.mean(margins)),
        "confusion_matrix": confusion_matrix(labels, pred_array, labels=list(range(len(CLASSES)))).tolist(),
    }
    try:
        result["macro_auroc"] = float(roc_auc_score(labels, probs, multi_class="ovr", average="macro", labels=list(range(len(CLASSES)))))
    except ValueError:
        result["macro_auroc"] = ""
    precision, recall, f1, support = precision_recall_fscore_support(labels, pred_array, labels=list(range(len(CLASSES))), zero_division=0)
    for idx, class_name in enumerate(CLASSES):
        result[f"{class_name}_precision"] = float(precision[idx])
        result[f"{class_name}_recall"] = float(recall[idx])
        result[f"{class_name}_f1"] = float(f1[idx])
        result[f"{class_name}_support"] = int(support[idx])
    return result


def save_confusion_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Render a small dataframe as a GitHub-style markdown table without tabulate."""

    if df.empty:
        return ""
    columns = [str(column) for column in df.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in df.iterrows():
        values: list[str] = []
        for column in df.columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def compute_refined_payload(
    model: ImageTextRefinementModel,
    split_payloads: dict[str, SplitPayload],
    text_embeddings: np.ndarray,
    concept_labels: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> RefinementPayload:
    model.eval()
    refined_splits: dict[str, SplitPayload] = {}
    with torch.no_grad():
        text_tensor = torch.tensor(text_embeddings, dtype=torch.float32, device=device)
        refined_text = model.encode_texts(text_tensor).cpu().numpy().astype(np.float32)
        for split, payload in split_payloads.items():
            chunks: list[np.ndarray] = []
            for start in range(0, payload.embeddings.shape[0], batch_size):
                batch = torch.tensor(payload.embeddings[start : start + batch_size], dtype=torch.float32, device=device)
                chunks.append(model.encode_images(batch).cpu().numpy().astype(np.float32))
            refined_splits[split] = SplitPayload(
                embeddings=np.concatenate(chunks, axis=0),
                labels=payload.labels.copy(),
                rows=payload.rows,
            )
    return RefinementPayload(split_payloads=refined_splits, text_embeddings=refined_text, concept_labels=concept_labels.copy())


def save_refined_cache(output_dir: Path, payload: RefinementPayload, bank: BankPayload) -> None:
    cache_dir = output_dir / "adapter_refinement"
    ensure_dir(cache_dir)
    for split, split_payload in payload.split_payloads.items():
        np.save(cache_dir / f"refined_image_embeddings_{split}.npy", split_payload.embeddings)
    np.savez_compressed(
        cache_dir / "refined_filtered_top300_text_embeddings.npz",
        concepts=bank.concepts,
        concept_embeddings=payload.text_embeddings,
        concept_labels=payload.concept_labels,
        classes=np.array(CLASSES, dtype=object),
    )


def evaluate_matrices(
    output_dir: Path,
    stage: str,
    split_payloads: dict[str, SplitPayload],
    text_embeddings: np.ndarray,
    concept_labels: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    matrix_dir = output_dir / "matrix_verification"
    for split, payload in split_payloads.items():
        matrix = compute_class_similarity_matrix(payload.embeddings, payload.labels, text_embeddings, concept_labels)
        save_matrix_csv(matrix_dir / f"{stage}_matrix_{split}.csv", matrix)
        save_heatmap(matrix_dir / f"{stage}_matrix_{split}.png", matrix, title=f"{stage} image-text matrix ({split})")
        rows.append(matrix_summary(stage=stage, split=split, matrix=matrix))
    return rows


def train_refinement(
    args: argparse.Namespace,
    split_payloads: dict[str, SplitPayload],
    bank: BankPayload,
    concept_labels: np.ndarray,
    device: torch.device,
) -> tuple[ImageTextRefinementModel, list[dict[str, Any]]]:
    set_seed(args.refine_seed)
    dim = int(bank.embeddings.shape[1])
    model = ImageTextRefinementModel(dim=dim, hidden_dim=args.adapter_hidden_dim, dropout=args.adapter_dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.refine_lr, weight_decay=args.refine_weight_decay)
    train_loader = make_refine_loader(
        split_payloads["train"],
        args.refine_batch_size,
        balanced=not args.disable_balanced_refine_sampler,
    )
    text_base = torch.tensor(bank.embeddings, dtype=torch.float32, device=device)
    concept_label_tensor = torch.tensor(concept_labels, dtype=torch.long, device=device)

    best_state: dict[str, torch.Tensor] | None = None
    best_val_f1 = -math.inf
    patience = 0
    rows: list[dict[str, Any]] = []

    for epoch in range(1, args.refine_epochs + 1):
        model.train()
        running = {"total": 0.0, "i2t": 0.0, "t2i": 0.0}
        seen = 0
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            image_z = model.encode_images(features)
            text_z = model.encode_texts(text_base)
            logits_i2t = image_z @ text_z.T / args.refine_tau
            pos_i2t = labels[:, None] == concept_label_tensor[None, :]
            loss_i2t = multipositive_nce(logits_i2t, pos_i2t)

            logits_t2i = text_z @ image_z.T / args.refine_tau
            pos_t2i = concept_label_tensor[:, None] == labels[None, :]
            loss_t2i = multipositive_nce(logits_t2i, pos_t2i)
            loss = loss_i2t + args.lambda_t2i * loss_t2i

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_n = int(labels.numel())
            seen += batch_n
            running["total"] += float(loss.detach().cpu()) * batch_n
            running["i2t"] += float(loss_i2t.detach().cpu()) * batch_n
            running["t2i"] += float(loss_t2i.detach().cpu()) * batch_n

        refined_payload = compute_refined_payload(model, split_payloads, bank.embeddings, concept_labels, device, args.refine_batch_size)
        val_vote = evaluate_majority_vote(
            refined_payload.split_payloads["val"].embeddings,
            refined_payload.split_payloads["val"].labels,
            refined_payload.text_embeddings,
            concept_labels,
            top_k=args.retrieval_k,
        )
        val_matrix = compute_class_similarity_matrix(
            refined_payload.split_payloads["val"].embeddings,
            refined_payload.split_payloads["val"].labels,
            refined_payload.text_embeddings,
            concept_labels,
        )
        summary = matrix_summary("refined_epoch", "val", val_matrix)
        row = {
            "epoch": epoch,
            "train_total_loss": running["total"] / max(1, seen),
            "train_i2t_loss": running["i2t"] / max(1, seen),
            "train_t2i_loss": running["t2i"] / max(1, seen),
            "val_retrieval_accuracy": val_vote["accuracy"],
            "val_retrieval_macro_f1": val_vote["macro_f1"],
            "val_retrieval_macro_auroc": val_vote["macro_auroc"],
            "val_diag_minus_offdiag": summary["diag_minus_offdiag"],
            "val_diagonal_mean": summary["diagonal_mean"],
            "val_off_diagonal_mean": summary["off_diagonal_mean"],
        }
        rows.append(row)
        if float(val_vote["macro_f1"]) > best_val_f1:
            best_val_f1 = float(val_vote["macro_f1"])
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience >= args.refine_patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, rows


def build_stage_whitelist(
    output_dir: Path,
    stage_name: str,
    bank: BankPayload,
    split_payloads: dict[str, SplitPayload],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    stage_dir = output_dir / stage_name
    ensure_dir(stage_dir)
    score_df, class_means = compute_discriminative_scores(bank, split_payloads["train"])
    score_df.to_csv(stage_dir / "concept_discriminative_scores.csv", index=False)
    build_top_tables(score_df, [args.top_k], stage_dir)
    save_hardest_negative_analysis(score_df, stage_dir)
    whitelist_df, concept_embeddings, _raw_m, m_matrix = build_whitelist(
        score_df=score_df,
        bank=bank,
        class_means=class_means,
        top_k=args.top_k,
        output_dir=stage_dir,
        near_duplicate_threshold=args.near_duplicate_threshold,
        normalization=args.m_normalization,
        softmax_temperature=args.softmax_temperature,
    )
    return whitelist_df, concept_embeddings, m_matrix, score_df


def run_majority_diagnostics(
    output_dir: Path,
    stage_name: str,
    split_payloads: dict[str, SplitPayload],
    text_embeddings: np.ndarray,
    concept_labels: np.ndarray,
    top_k: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    confusion_payload: dict[str, Any] = {}
    for split, payload in split_payloads.items():
        result = evaluate_majority_vote(payload.embeddings, payload.labels, text_embeddings, concept_labels, top_k=top_k)
        row = {
            "stage": stage_name,
            "split": split,
            "top_k": top_k,
            "accuracy": result["accuracy"],
            "macro_f1": result["macro_f1"],
            "macro_auroc": result["macro_auroc"],
            "mean_vote_margin": result["mean_vote_margin"],
        }
        for class_name in CLASSES:
            row[f"{class_name}_precision"] = result[f"{class_name}_precision"]
            row[f"{class_name}_recall"] = result[f"{class_name}_recall"]
            row[f"{class_name}_f1"] = result[f"{class_name}_f1"]
            row[f"{class_name}_support"] = result[f"{class_name}_support"]
        rows.append(row)
        confusion_payload[split] = result["confusion_matrix"]
    save_confusion_json(output_dir / "retrieval_majority_vote" / f"{stage_name}_confusion_matrices.json", confusion_payload)
    return rows


def make_cbm_args(args: argparse.Namespace, output_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        device=args.device,
        lr=args.cbm_lr,
        weight_decay=args.cbm_weight_decay,
        hidden_dim=args.cbm_hidden_dim,
        dropout=args.cbm_dropout,
        lambda_cycl=args.cbm_lambda_cycl,
        lambda_align=args.cbm_lambda_align,
        tau=args.cbm_tau,
        batch_size=args.cbm_batch_size,
        epochs=args.cbm_epochs,
        patience=args.cbm_patience,
        output_dir=output_dir,
    )


def run_cbm_stage(
    output_dir: Path,
    stage_name: str,
    split_payloads: dict[str, SplitPayload],
    concept_embeddings: np.ndarray,
    m_matrix: np.ndarray,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    stage_dir = output_dir / "cbm_training" / stage_name
    ensure_dir(stage_dir)
    cbm_args = make_cbm_args(args, stage_dir)
    rows: list[dict[str, Any]] = []
    for seed in parse_int_list(args.cbm_seeds):
        result = train_one_setting(
            split_payloads=split_payloads,
            concept_embeddings=concept_embeddings,
            m_matrix=m_matrix,
            top_k=args.top_k,
            seed=seed,
            args=cbm_args,
            output_dir=stage_dir,
        )
        result["stage"] = stage_name
        rows.append(result)
    write_csv(stage_dir / "training_seed_results.csv", rows)
    summary = summarize_results(rows)
    for row in summary:
        row["stage"] = stage_name
    write_csv(stage_dir / "training_results_summary.csv", summary)
    return rows


def save_refinement_report(
    output_dir: Path,
    args: argparse.Namespace,
    matrix_rows: list[dict[str, Any]],
    majority_rows: list[dict[str, Any]],
    cbm_rows: list[dict[str, Any]],
) -> None:
    matrix_df = pd.DataFrame(matrix_rows)
    majority_df = pd.DataFrame(majority_rows)
    cbm_df = pd.DataFrame(cbm_rows) if cbm_rows else pd.DataFrame()
    lines = [
        "# Embedding Refinement Stage Report",
        "",
        "## Setup",
        "",
        f"- Input bank: `{args.bank_dir}`.",
        f"- Starting bank: filtered_top300 concepts.",
        f"- Adapter hidden dim: `{args.adapter_hidden_dim}`.",
        f"- Refinement epochs requested: `{args.refine_epochs}`.",
        f"- Refinement loss: image-to-text multi-positive InfoNCE + `{args.lambda_t2i}` * text-to-image InfoNCE.",
        f"- Whitelist rule after refinement: hardest-negative margin top `{args.top_k}` per class.",
        f"- CBM loss: `L_cls + {args.cbm_lambda_cycl} * L_CyCL + {args.cbm_lambda_align} * L_align`.",
        "",
        "## Matrix Summary",
        "",
    ]
    if not matrix_df.empty:
        compact = matrix_df[["stage", "split", "diagonal_mean", "off_diagonal_mean", "diag_minus_offdiag"]].copy()
        lines.append(dataframe_to_markdown(compact))
    lines.extend(["", "## Majority-Vote Retrieval Summary", ""])
    if not majority_df.empty:
        compact = majority_df[["stage", "split", "accuracy", "macro_f1", "macro_auroc", "mean_vote_margin"]].copy()
        lines.append(dataframe_to_markdown(compact))
    if not cbm_df.empty:
        lines.extend(["", "## CBM Training Summary", ""])
        summary = []
        for stage, group in cbm_df.groupby("stage"):
            row: dict[str, Any] = {"stage": stage, "n_runs": len(group)}
            for metric in ["val_accuracy", "val_macro_f1", "val_macro_auroc", "test_accuracy", "test_macro_f1", "test_macro_auroc"]:
                values = pd.to_numeric(group[metric], errors="coerce")
                row[f"{metric}_mean"] = values.mean()
                row[f"{metric}_std"] = values.std(ddof=1) if len(values) > 1 else 0.0
            summary.append(row)
        lines.append(dataframe_to_markdown(pd.DataFrame(summary)))
    lines.extend(
        [
            "",
            "## Interpretation Guide",
            "",
            "- If refined matrix margin increases, the adapters made image-text similarity more class-structured.",
            "- If majority-vote retrieval improves, the refined similarity vectors are cleaner before CBM training.",
            "- If refined CBM improves over original-control CBM, the refined vectors help the downstream concept bottleneck model.",
            "- If validation improves but test drops, the adapters likely overfit the train/val category structure.",
        ]
    )
    (output_dir / "refinement_stage_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)
        try:
            torch.set_num_interop_threads(args.num_threads)
        except RuntimeError:
            pass
    ensure_dir(args.output_dir)
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
            "cuda_device_count": torch.cuda.device_count(),
        },
    )

    split_payloads = load_split_payloads(args.embeddings_dir)
    bank = load_top300_bank(args.bank_dir)
    concept_labels = labels_from_bank(bank)

    original_matrix_rows = evaluate_matrices(args.output_dir, "original", split_payloads, bank.embeddings, concept_labels)
    original_majority_rows = run_majority_diagnostics(
        args.output_dir,
        "original_filtered_top300",
        split_payloads,
        bank.embeddings,
        concept_labels,
        top_k=args.retrieval_k,
    )
    original_whitelist_df, original_concept_embeddings, original_m_matrix, _ = build_stage_whitelist(
        args.output_dir,
        "original_filtering",
        bank,
        split_payloads,
        args,
    )
    original_whitelist_labels = np.array([CLASS_TO_INDEX[str(item)] for item in original_whitelist_df["primary_target_class"].tolist()], dtype=np.int64)
    original_whitelist_majority_rows = run_majority_diagnostics(
        args.output_dir,
        "original_whitelist_top10",
        split_payloads,
        original_concept_embeddings,
        original_whitelist_labels,
        top_k=args.retrieval_k,
    )

    model, train_rows = train_refinement(args, split_payloads, bank, concept_labels, device)
    refinement_dir = args.output_dir / "adapter_refinement"
    ensure_dir(refinement_dir)
    write_csv(refinement_dir / "refinement_train_log.csv", train_rows)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "classes": CLASSES,
        },
        refinement_dir / "best_refinement_adapters.pt",
    )
    refined_payload = compute_refined_payload(model, split_payloads, bank.embeddings, concept_labels, device, args.refine_batch_size)
    save_refined_cache(args.output_dir, refined_payload, bank)

    refined_matrix_rows = evaluate_matrices(
        args.output_dir,
        "refined",
        refined_payload.split_payloads,
        refined_payload.text_embeddings,
        concept_labels,
    )
    refined_bank = BankPayload(rows=bank.rows, concepts=bank.concepts, embeddings=refined_payload.text_embeddings)
    _refined_score_df_for_fixed, refined_class_means_for_fixed = compute_discriminative_scores(
        refined_bank,
        refined_payload.split_payloads["train"],
    )
    original_indices = original_whitelist_df["representative_concept_index"].astype(int).to_numpy(dtype=np.int64)
    refined_original_concept_embeddings = refined_payload.text_embeddings[original_indices].astype(np.float32)
    refined_original_m_matrix = normalize_m(
        refined_class_means_for_fixed[original_indices].astype(np.float32),
        normalization=args.m_normalization,
        temperature=args.softmax_temperature,
    )
    fixed_refined_dir = args.output_dir / "refined_vectors_original_whitelist"
    ensure_dir(fixed_refined_dir)
    original_whitelist_df.to_csv(fixed_refined_dir / f"whitelist_top{args.top_k}.csv", index=False)
    np.savez_compressed(
        fixed_refined_dir / f"whitelist_top{args.top_k}.npz",
        concepts=original_whitelist_df["concept"].to_numpy(dtype=object),
        concept_embeddings=refined_original_concept_embeddings,
        raw_M=refined_class_means_for_fixed[original_indices].astype(np.float32),
        M=refined_original_m_matrix,
        classes=np.array(CLASSES, dtype=object),
    )
    refined_whitelist_df, refined_concept_embeddings, refined_m_matrix, _ = build_stage_whitelist(
        args.output_dir,
        "refined_filtering",
        refined_bank,
        refined_payload.split_payloads,
        args,
    )
    refined_whitelist_labels = np.array([CLASS_TO_INDEX[str(item)] for item in refined_whitelist_df["primary_target_class"].tolist()], dtype=np.int64)

    refined_majority_rows = run_majority_diagnostics(
        args.output_dir,
        "refined_filtered_top300",
        refined_payload.split_payloads,
        refined_payload.text_embeddings,
        concept_labels,
        top_k=args.retrieval_k,
    )
    refined_original_whitelist_majority_rows = run_majority_diagnostics(
        args.output_dir,
        "refined_vectors_original_whitelist_top10",
        refined_payload.split_payloads,
        refined_original_concept_embeddings,
        original_whitelist_labels,
        top_k=args.retrieval_k,
    )
    refined_whitelist_majority_rows = run_majority_diagnostics(
        args.output_dir,
        "refined_whitelist_top10",
        refined_payload.split_payloads,
        refined_concept_embeddings,
        refined_whitelist_labels,
        top_k=args.retrieval_k,
    )

    matrix_rows = original_matrix_rows + refined_matrix_rows
    write_csv(args.output_dir / "matrix_verification" / "matrix_summary.csv", matrix_rows)
    majority_rows = (
        original_majority_rows
        + original_whitelist_majority_rows
        + refined_majority_rows
        + refined_original_whitelist_majority_rows
        + refined_whitelist_majority_rows
    )
    write_csv(args.output_dir / "retrieval_majority_vote" / "majority_vote_results.csv", majority_rows)

    cbm_rows: list[dict[str, Any]] = []
    if args.run_cbm:
        cbm_rows.extend(
            run_cbm_stage(
                args.output_dir,
                "original_control_top10",
                split_payloads,
                original_concept_embeddings,
                original_m_matrix,
                args,
            )
        )
        cbm_rows.extend(
            run_cbm_stage(
                args.output_dir,
                "refined_vectors_original_whitelist_top10",
                refined_payload.split_payloads,
                refined_original_concept_embeddings,
                refined_original_m_matrix,
                args,
            )
        )
        cbm_rows.extend(
            run_cbm_stage(
                args.output_dir,
                "refined_top10",
                refined_payload.split_payloads,
                refined_concept_embeddings,
                refined_m_matrix,
                args,
            )
        )
        write_csv(args.output_dir / "cbm_training" / "combined_seed_results.csv", cbm_rows)
        summary_rows: list[dict[str, Any]] = []
        for stage, group in pd.DataFrame(cbm_rows).groupby("stage"):
            for top_k, top_group in group.groupby("top_k"):
                row = {"stage": stage, "top_k": int(top_k), "n_runs": int(len(top_group))}
                for metric in ["val_accuracy", "val_macro_f1", "val_macro_auroc", "test_accuracy", "test_macro_f1", "test_macro_auroc"]:
                    values = pd.to_numeric(top_group[metric], errors="coerce")
                    row[f"{metric}_mean"] = float(values.mean())
                    row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
                summary_rows.append(row)
        write_csv(args.output_dir / "cbm_training" / "combined_results_summary.csv", summary_rows)

    save_refinement_report(args.output_dir, args, matrix_rows, majority_rows, cbm_rows)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "matrix_summary": matrix_rows,
                "majority_vote": majority_rows,
                "cbm_rows": cbm_rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
