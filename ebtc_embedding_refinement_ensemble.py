#!/usr/bin/env python3
"""Evaluate checkpoint ensembles produced by ebtc_embedding_refinement_stage.py."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score

from ebtc_discriminative_whitelist_cycl import AdapterConceptCBM, CLASSES, load_split_payloads
from ebtc_project_paths import OFFICIAL_EMBEDDINGS_DIR, OUTPUT_ROOT


DEFAULT_STAGE_DIR = OUTPUT_ROOT / "ebtc_embedding_refinement_stage_conservative_outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an ensemble for an embedding-refinement CBM stage.")
    parser.add_argument(
        "--stage-output-dir",
        type=Path,
        action="append",
        default=None,
        help="One or more output directories from ebtc_embedding_refinement_stage.py.",
    )
    parser.add_argument(
        "--stage",
        choices=["original_control_top10", "refined_vectors_original_whitelist_top10", "refined_top10"],
        required=True,
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--embeddings-dir", type=Path, default=OFFICIAL_EMBEDDINGS_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-8, None)


def load_stage_split_payloads(stage_output_dir: Path, stage: str, embeddings_dir: Path):
    split_payloads = load_split_payloads(embeddings_dir)
    if stage == "original_control_top10":
        return split_payloads
    refined_dir = stage_output_dir / "adapter_refinement"
    for split in ["train", "val", "test"]:
        split_payloads[split].embeddings = np.load(refined_dir / f"refined_image_embeddings_{split}.npy").astype(np.float32)
    return split_payloads


def whitelist_path_for_stage(stage_output_dir: Path, stage: str, top_k: int) -> Path:
    if stage == "original_control_top10":
        return stage_output_dir / "original_filtering" / f"whitelist_top{top_k}.npz"
    if stage == "refined_vectors_original_whitelist_top10":
        return stage_output_dir / "refined_vectors_original_whitelist" / f"whitelist_top{top_k}.npz"
    return stage_output_dir / "refined_filtering" / f"whitelist_top{top_k}.npz"


def load_models(stage_output_dirs: list[Path], stage: str, top_k: int) -> tuple[list[AdapterConceptCBM], int, list[Path]]:
    models: list[AdapterConceptCBM] = []
    checkpoint_paths: list[Path] = []
    n_concepts: int | None = None
    for stage_output_dir in stage_output_dirs:
        whitelist_path = whitelist_path_for_stage(stage_output_dir, stage, top_k)
        payload = np.load(whitelist_path, allow_pickle=True)
        concept_embeddings = payload["concept_embeddings"].astype(np.float32)
        if n_concepts is None:
            n_concepts = int(concept_embeddings.shape[0])
        elif n_concepts != int(concept_embeddings.shape[0]):
            raise ValueError(f"Inconsistent concept count in {whitelist_path}")

        stage_checkpoint_paths = sorted((stage_output_dir / "cbm_training" / stage / "training" / f"top{top_k}").glob("seed_*/best_checkpoint.pt"))
        if not stage_checkpoint_paths:
            raise FileNotFoundError(f"No checkpoints found for {stage} under {stage_output_dir}")
        for checkpoint_path in stage_checkpoint_paths:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            checkpoint_args = checkpoint.get("args", {})
            model = AdapterConceptCBM(
                concept_embeddings=concept_embeddings,
                hidden_dim=int(checkpoint_args.get("hidden_dim", 128)),
                dropout=float(checkpoint_args.get("dropout", 0.1)),
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            models.append(model)
            checkpoint_paths.append(checkpoint_path)
    if n_concepts is None:
        raise ValueError("No stage output dirs were provided.")
    return models, n_concepts, checkpoint_paths


def evaluate(models: list[AdapterConceptCBM], embeddings: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    features = torch.tensor(embeddings, dtype=torch.float32)
    all_probs: list[np.ndarray] = []
    with torch.no_grad():
        for model in models:
            logits = model(features)["logits"].numpy()
            all_probs.append(softmax(logits))
    probs = np.mean(all_probs, axis=0)
    preds = probs.argmax(axis=1)
    result: dict[str, Any] = {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "macro_auroc": float(roc_auc_score(labels, probs, multi_class="ovr", average="macro", labels=list(range(len(CLASSES))))),
        "confusion_matrix": confusion_matrix(labels, preds, labels=list(range(len(CLASSES)))).tolist(),
    }
    precision, recall, f1, support = precision_recall_fscore_support(labels, preds, labels=list(range(len(CLASSES))), zero_division=0)
    for idx, class_name in enumerate(CLASSES):
        result[f"{class_name}_precision"] = float(precision[idx])
        result[f"{class_name}_recall"] = float(recall[idx])
        result[f"{class_name}_f1"] = float(f1[idx])
        result[f"{class_name}_support"] = int(support[idx])
    return result


def main() -> None:
    args = parse_args()
    stage_output_dirs = args.stage_output_dir or [DEFAULT_STAGE_DIR]
    output_dir = args.output_dir or (stage_output_dirs[0] / "cbm_training" / args.stage / "ensemble")
    output_dir.mkdir(parents=True, exist_ok=True)
    split_payloads = load_stage_split_payloads(stage_output_dirs[0], args.stage, args.embeddings_dir)
    models, n_concepts, checkpoint_paths = load_models(stage_output_dirs, args.stage, args.top_k)
    rows: list[dict[str, Any]] = []
    confusion_payload: dict[str, Any] = {}
    for split in ["val", "test"]:
        metrics = evaluate(models, split_payloads[split].embeddings, split_payloads[split].labels)
        row = {
            "stage": args.stage,
            "split": split,
            "top_k": args.top_k,
            "n_models": len(models),
            "n_concepts": n_concepts,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "macro_auroc": metrics["macro_auroc"],
        }
        for class_name in CLASSES:
            row[f"{class_name}_precision"] = metrics[f"{class_name}_precision"]
            row[f"{class_name}_recall"] = metrics[f"{class_name}_recall"]
            row[f"{class_name}_f1"] = metrics[f"{class_name}_f1"]
            row[f"{class_name}_support"] = metrics[f"{class_name}_support"]
        rows.append(row)
        confusion_payload[split] = metrics["confusion_matrix"]
    write_csv(output_dir / "ensemble_metrics.csv", rows)
    (output_dir / "ensemble_confusion_matrices.json").write_text(json.dumps(confusion_payload, indent=2), encoding="utf-8")
    (output_dir / "ensemble_manifest.json").write_text(
        json.dumps(
            {
                "stage_output_dirs": [str(path) for path in stage_output_dirs],
                "stage": args.stage,
                "top_k": args.top_k,
                "n_models": len(models),
                "n_concepts": n_concepts,
                "checkpoint_paths": [str(path) for path in checkpoint_paths],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
