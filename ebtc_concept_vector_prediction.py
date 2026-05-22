#!/usr/bin/env python3
"""Predict cleaned 40-d concept activation vectors from image embeddings.

The target vector keeps the raw cosine activation values at concepts belonging
to the image's true class and sets all other concept positions to -1. The
trained MLP is then evaluated with the same top-k majority-vote and class-
average rules used by the cosine-only assignment diagnostic.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

from ebtc_cosine_topk_assignment import (
    STAGE_TO_WHITELIST_DIR,
    ConceptBank,
    compute_topk_tables,
    dataframe_to_markdown,
    load_concept_bank,
    load_stage_split_payloads,
    metric_payload,
    per_class_rows,
    predict_class_average,
    predict_majority_vote,
    save_confusion_outputs,
    save_topk_count_matrix,
)
from ebtc_discriminative_whitelist_cycl import CLASSES, SplitPayload
from ebtc_project_paths import OFFICIAL_EMBEDDINGS_DIR, OUTPUT_ROOT


DEFAULT_REFINED_STAGE_DIR = OUTPUT_ROOT / "ebtc_embedding_refinement_stage_conservative_outputs"
DEFAULT_OUTPUT_DIR = OUTPUT_ROOT / "ebtc_concept_vector_prediction_outputs"


@dataclass
class SplitVectors:
    raw_vectors: np.ndarray
    targets: np.ndarray
    target_weights: np.ndarray
    labels: np.ndarray


class ConceptVectorPredictor(nn.Module):
    """Small MLP that maps an image embedding directly to a concept vector."""

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

    def forward(self, image_embeddings: torch.Tensor) -> torch.Tensor:
        return self.net(image_embeddings)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an MLP to predict cleaned 40-d concept vectors.")
    parser.add_argument("--refined-stage-dir", type=Path, default=DEFAULT_REFINED_STAGE_DIR)
    parser.add_argument("--embeddings-dir", type=Path, default=OFFICIAL_EMBEDDINGS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stage", choices=sorted(STAGE_TO_WHITELIST_DIR), default="refined_vectors_original_top10")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--target-negative", type=float, default=-1.0)
    parser.add_argument(
        "--positive-weight",
        type=float,
        default=1.0,
        help="MSE weight for true-class concept positions; non-true positions use weight 1.",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--num-threads", type=int, default=2)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def clean_targets(
    raw_vectors: np.ndarray,
    labels: np.ndarray,
    concept_labels: np.ndarray,
    negative_value: float,
    positive_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    targets = np.full_like(raw_vectors, fill_value=float(negative_value), dtype=np.float32)
    weights = np.ones_like(raw_vectors, dtype=np.float32)
    for class_idx in range(len(CLASSES)):
        row_mask = labels == class_idx
        col_mask = concept_labels == class_idx
        if bool(row_mask.any()) and bool(col_mask.any()):
            targets[np.ix_(row_mask, col_mask)] = raw_vectors[np.ix_(row_mask, col_mask)]
            weights[np.ix_(row_mask, col_mask)] = float(positive_weight)
    return targets.astype(np.float32), weights.astype(np.float32)


def build_split_vectors(
    payloads: dict[str, SplitPayload],
    bank: ConceptBank,
    negative_value: float,
    positive_weight: float,
) -> dict[str, SplitVectors]:
    out: dict[str, SplitVectors] = {}
    for split, payload in payloads.items():
        raw_vectors = (payload.embeddings @ bank.embeddings.T).astype(np.float32)
        targets, weights = clean_targets(
            raw_vectors=raw_vectors,
            labels=payload.labels,
            concept_labels=bank.labels,
            negative_value=negative_value,
            positive_weight=positive_weight,
        )
        out[split] = SplitVectors(
            raw_vectors=raw_vectors,
            targets=targets,
            target_weights=weights,
            labels=payload.labels,
        )
    return out


def make_loader(payload: SplitPayload, split_vectors: SplitVectors, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(payload.embeddings, dtype=torch.float32),
        torch.tensor(split_vectors.targets, dtype=torch.float32),
        torch.tensor(split_vectors.target_weights, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def weighted_mse(predictions: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return ((predictions - targets).pow(2) * weights).mean()


def evaluate_loss(
    model: ConceptVectorPredictor,
    payload: SplitPayload,
    split_vectors: SplitVectors,
    device: torch.device,
    batch_size: int,
) -> float:
    loader = make_loader(payload, split_vectors, batch_size=batch_size, shuffle=False)
    model.eval()
    total = 0.0
    seen = 0
    with torch.no_grad():
        for features, targets, weights in loader:
            features = features.to(device)
            targets = targets.to(device)
            weights = weights.to(device)
            preds = model(features)
            loss = weighted_mse(preds, targets, weights)
            batch_n = int(features.shape[0])
            total += float(loss.detach().cpu()) * batch_n
            seen += batch_n
    return total / max(1, seen)


def predict_vectors(
    model: ConceptVectorPredictor,
    payload: SplitPayload,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    parts: list[np.ndarray] = []
    loader = DataLoader(torch.tensor(payload.embeddings, dtype=torch.float32), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for features in loader:
            preds = model(features.to(device)).detach().cpu().numpy().astype(np.float32)
            parts.append(preds)
    return np.concatenate(parts, axis=0)


def train_seed(
    seed: int,
    args: argparse.Namespace,
    payloads: dict[str, SplitPayload],
    split_vectors: dict[str, SplitVectors],
    bank: ConceptBank,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    set_seed(seed)
    model = ConceptVectorPredictor(
        input_dim=payloads["train"].embeddings.shape[1],
        output_dim=len(bank.labels),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = make_loader(payloads["train"], split_vectors["train"], batch_size=args.batch_size, shuffle=True)

    best_state: dict[str, torch.Tensor] | None = None
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    curve_rows: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for features, targets, weights in train_loader:
            features = features.to(device)
            targets = targets.to(device)
            weights = weights.to(device)
            preds = model(features)
            loss = weighted_mse(preds, targets, weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_n = int(features.shape[0])
            running += float(loss.detach().cpu()) * batch_n
            seen += batch_n

        train_loss = running / max(1, seen)
        val_loss = evaluate_loss(model, payloads["val"], split_vectors["val"], device=device, batch_size=args.batch_size)
        curve_rows.append({"seed": seed, "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= args.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    seed_dir = args.output_dir / "training" / f"seed_{seed}"
    ensure_dir(seed_dir)
    pd.DataFrame(curve_rows).to_csv(seed_dir / "training_curve.csv", index=False)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "seed": seed,
            "classes": CLASSES,
            "concept_labels": bank.labels,
            "args": vars(args),
        },
        seed_dir / "best_checkpoint.pt",
    )

    predictions = {
        split: predict_vectors(model, payload, device=device, batch_size=args.batch_size)
        for split, payload in payloads.items()
    }
    summary = {
        "source": f"predicted_seed_{seed}",
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "train_loss": evaluate_loss(model, payloads["train"], split_vectors["train"], device=device, batch_size=args.batch_size),
        "test_loss": evaluate_loss(model, payloads["test"], split_vectors["test"], device=device, batch_size=args.batch_size),
    }
    write_json(seed_dir / "metrics.json", summary)
    return summary, predictions


def vector_quality_rows(
    source: str,
    vectors_by_split: dict[str, np.ndarray],
    split_vectors: dict[str, SplitVectors],
    concept_labels: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, vectors in vectors_by_split.items():
        targets = split_vectors[split].targets
        labels = split_vectors[split].labels
        true_mask = np.zeros_like(vectors, dtype=bool)
        for class_idx in range(len(CLASSES)):
            true_mask[np.ix_(labels == class_idx, concept_labels == class_idx)] = True
        false_mask = ~true_mask
        rows.append(
            {
                "source": source,
                "split": split,
                "target_mse_all": float(np.mean((vectors - targets) ** 2)),
                "target_mse_true_positions": float(np.mean((vectors[true_mask] - targets[true_mask]) ** 2)),
                "target_mse_false_positions": float(np.mean((vectors[false_mask] - targets[false_mask]) ** 2)),
                "mean_true_position_value": float(np.mean(vectors[true_mask])),
                "mean_false_position_value": float(np.mean(vectors[false_mask])),
            }
        )
    return rows


def evaluate_vector_source(
    output_dir: Path,
    source: str,
    vectors_by_split: dict[str, np.ndarray],
    payloads: dict[str, SplitPayload],
    bank: ConceptBank,
    top_k: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metric_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    ensure_dir(output_dir / "topk_class_count_matrices")
    for split, payload in payloads.items():
        vectors = vectors_by_split[split]
        top_indices, counts, mean_scores, mean_scores_for_auc = compute_topk_tables(vectors, bank.labels, top_k=top_k)
        majority_preds = predict_majority_vote(counts, mean_scores)
        average_preds = predict_class_average(mean_scores)
        rules = {
            "majority_vote": (majority_preds, counts),
            "class_average": (average_preds, mean_scores_for_auc),
        }
        for rule, (preds, scores) in rules.items():
            metric_rows.append(metric_payload(payload.labels, preds, scores, source, split, rule, top_k))
            class_rows.extend(per_class_rows(payload.labels, preds, counts, source, split, rule))
            cm = confusion_matrix(payload.labels, preds, labels=list(range(len(CLASSES))))
            save_confusion_outputs(output_dir, cm, source, split, rule)
        save_topk_count_matrix(output_dir, payload.labels, counts, source, split)
    return metric_rows, class_rows


def write_report(
    output_dir: Path,
    args: argparse.Namespace,
    training_summary: pd.DataFrame,
    results: pd.DataFrame,
    per_class: pd.DataFrame,
    quality: pd.DataFrame,
) -> None:
    val_test = results[results["split"].isin(["val", "test"])].copy()
    test = results[results["split"] == "test"].copy()
    metric_cols = {"accuracy", "macro_f1", "macro_auroc"}
    best = test.sort_values(["macro_f1", "accuracy"], ascending=False).head(1)
    test_per_class = (
        per_class[per_class["split"] == "test"]
        .pivot_table(index=["stage", "rule"], columns="class", values="f1", aggfunc="first")
        .reset_index()
    )
    quality_test = quality[quality["split"] == "test"].copy()

    lines = [
        "# Concept Vector Prediction",
        "",
        "## Setup",
        "",
        f"- Stage: `{args.stage}`.",
        f"- Top-k value: `{args.top_k}`.",
        f"- Target rule: keep true-class concept cosine values; set all other class concept positions to `{args.target_negative}`.",
        f"- MLP: image embedding -> hidden `{args.hidden_dim}` -> hidden `{args.hidden_dim}` -> 40-d concept vector.",
        f"- Seeds: `{args.seeds}`.",
        "- Evaluation repeats the existing top10 `majority_vote` and `class_average` rules on the predicted vector.",
        "",
        "## Training Summary",
        "",
        dataframe_to_markdown(training_summary, {"best_val_loss", "train_loss", "test_loss"}),
        "",
        "## Main Val/Test Results",
        "",
        dataframe_to_markdown(val_test[["stage", "split", "rule", "accuracy", "macro_f1", "macro_auroc"]], metric_cols),
        "",
        "## Best Test Row",
        "",
        dataframe_to_markdown(best[["stage", "rule", "accuracy", "macro_f1", "macro_auroc"]], metric_cols),
        "",
        "## Test Per-Class F1",
        "",
        dataframe_to_markdown(test_per_class, set(CLASSES)),
        "",
        "## Test Vector Quality",
        "",
        dataframe_to_markdown(
            quality_test[
                [
                    "source",
                    "target_mse_all",
                    "target_mse_true_positions",
                    "target_mse_false_positions",
                    "mean_true_position_value",
                    "mean_false_position_value",
                ]
            ],
            {
                "target_mse_all",
                "target_mse_true_positions",
                "target_mse_false_positions",
                "mean_true_position_value",
                "mean_false_position_value",
            },
        ),
        "",
        "## Output Files",
        "",
        f"- Metrics: `{output_dir / 'concept_vector_prediction_results.csv'}`",
        f"- Per-class metrics: `{output_dir / 'concept_vector_prediction_per_class.csv'}`",
        f"- Vector quality: `{output_dir / 'concept_vector_prediction_quality.csv'}`",
        f"- Training checkpoints: `{output_dir / 'training'}`",
        "",
        "## Interpretation Notes",
        "",
        "- This is a supervised vector-cleaning diagnostic. If predicted-vector top-k scores improve strongly over raw cosine activation, the noisy activation vector is a real bottleneck.",
        "- Because the MLP learns from class labels through the cleaned target, high top-k accuracy should be interpreted as evidence that the image embedding contains recoverable class signal, not as a purely unsupervised concept readout.",
    ]
    (output_dir / "concept_vector_prediction_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)
    ensure_dir(args.output_dir)
    device = resolve_device(args.device)

    bank = load_concept_bank(args.refined_stage_dir, args.stage)
    payloads = load_stage_split_payloads(args.refined_stage_dir, args.embeddings_dir, args.stage)
    split_vectors = build_split_vectors(
        payloads=payloads,
        bank=bank,
        negative_value=args.target_negative,
        positive_weight=args.positive_weight,
    )

    write_json(
        args.output_dir / "experiment_config.json",
        {
            "refined_stage_dir": str(args.refined_stage_dir),
            "embeddings_dir": str(args.embeddings_dir),
            "output_dir": str(args.output_dir),
            "stage": args.stage,
            "top_k": args.top_k,
            "seeds": parse_int_list(args.seeds),
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "target_negative": args.target_negative,
            "positive_weight": args.positive_weight,
            "device": str(device),
            "classes": CLASSES,
            "n_concepts": int(len(bank.labels)),
            **{f"n_{class_name}_concepts": int((bank.labels == idx).sum()) for idx, class_name in enumerate(CLASSES)},
        },
    )
    bank.metadata.to_csv(args.output_dir / "concept_metadata.csv", index=False)

    raw_source = "raw_cosine_activation"
    raw_vectors = {split: vectors.raw_vectors for split, vectors in split_vectors.items()}
    metric_rows, class_rows = evaluate_vector_source(args.output_dir, raw_source, raw_vectors, payloads, bank, args.top_k)
    quality_rows = vector_quality_rows(raw_source, raw_vectors, split_vectors, bank.labels)

    seed_predictions: list[dict[str, np.ndarray]] = []
    training_rows: list[dict[str, Any]] = []
    for seed in parse_int_list(args.seeds):
        summary, predictions = train_seed(seed, args, payloads, split_vectors, bank, device)
        seed_predictions.append(predictions)
        training_rows.append(summary)
        rows, per_class = evaluate_vector_source(
            args.output_dir,
            f"predicted_seed_{seed}",
            predictions,
            payloads,
            bank,
            args.top_k,
        )
        metric_rows.extend(rows)
        class_rows.extend(per_class)
        quality_rows.extend(vector_quality_rows(f"predicted_seed_{seed}", predictions, split_vectors, bank.labels))

    ensemble_predictions: dict[str, np.ndarray] = {}
    for split in payloads:
        ensemble_predictions[split] = np.mean([preds[split] for preds in seed_predictions], axis=0).astype(np.float32)
    rows, per_class = evaluate_vector_source(
        args.output_dir,
        "predicted_ensemble_mean",
        ensemble_predictions,
        payloads,
        bank,
        args.top_k,
    )
    metric_rows.extend(rows)
    class_rows.extend(per_class)
    quality_rows.extend(vector_quality_rows("predicted_ensemble_mean", ensemble_predictions, split_vectors, bank.labels))

    training_summary = pd.DataFrame(training_rows)
    results = pd.DataFrame(metric_rows)
    per_class = pd.DataFrame(class_rows)
    quality = pd.DataFrame(quality_rows)
    training_summary.to_csv(args.output_dir / "concept_vector_prediction_training_summary.csv", index=False)
    results.to_csv(args.output_dir / "concept_vector_prediction_results.csv", index=False)
    per_class.to_csv(args.output_dir / "concept_vector_prediction_per_class.csv", index=False)
    quality.to_csv(args.output_dir / "concept_vector_prediction_quality.csv", index=False)
    write_report(args.output_dir, args, training_summary, results, per_class, quality)
    print(f"Wrote concept-vector prediction outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
