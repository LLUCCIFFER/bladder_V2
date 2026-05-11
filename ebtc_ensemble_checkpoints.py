#!/usr/bin/env python3
"""Evaluate an ensemble of saved discriminative-whitelist CBM checkpoints."""

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
from ebtc_project_paths import OFFICIAL_EMBEDDINGS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ensemble predictions from saved whitelist-CBM checkpoints.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Directory produced by ebtc_discriminative_whitelist_cycl.py")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--embeddings-dir", type=Path, default=OFFICIAL_EMBEDDINGS_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-8, None)


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


def load_models(run_dir: Path, top_k: int) -> tuple[list[AdapterConceptCBM], np.ndarray, list[Path]]:
    whitelist_path = run_dir / f"whitelist_top{top_k}.npz"
    payload = np.load(whitelist_path, allow_pickle=True)
    concept_embeddings = payload["concept_embeddings"].astype(np.float32)
    checkpoint_paths = sorted((run_dir / "training" / f"top{top_k}").glob("seed_*/best_checkpoint.pt"))
    if not checkpoint_paths:
        raise FileNotFoundError(f"No checkpoints found under {run_dir / 'training' / f'top{top_k}'}")

    models: list[AdapterConceptCBM] = []
    for checkpoint_path in checkpoint_paths:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint_args = checkpoint.get("args", {})
        model = AdapterConceptCBM(
            concept_embeddings=concept_embeddings,
            hidden_dim=int(checkpoint_args.get("hidden_dim", 256)),
            dropout=float(checkpoint_args.get("dropout", 0.1)),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        models.append(model)
    return models, concept_embeddings, checkpoint_paths


def evaluate_split(models: list[AdapterConceptCBM], split_embeddings: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    features = torch.tensor(split_embeddings, dtype=torch.float32)
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
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        preds,
        labels=list(range(len(CLASSES))),
        zero_division=0,
    )
    for index, class_name in enumerate(CLASSES):
        result[f"{class_name}_precision"] = float(precision[index])
        result[f"{class_name}_recall"] = float(recall[index])
        result[f"{class_name}_f1"] = float(f1[index])
        result[f"{class_name}_support"] = int(support[index])
    return result


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.run_dir / "ensemble")
    output_dir.mkdir(parents=True, exist_ok=True)

    split_payloads = load_split_payloads(args.embeddings_dir)
    models, concept_embeddings, checkpoint_paths = load_models(args.run_dir, args.top_k)

    rows: list[dict[str, Any]] = []
    confusion_payload: dict[str, Any] = {}
    for split in ["val", "test"]:
        payload = split_payloads[split]
        metrics = evaluate_split(models, payload.embeddings, payload.labels)
        row = {
            "split": split,
            "run_dir": str(args.run_dir),
            "top_k": args.top_k,
            "n_models": len(models),
            "n_concepts": int(concept_embeddings.shape[0]),
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
    (output_dir / "ensemble_confusion_matrices.json").write_text(
        json.dumps(confusion_payload, indent=2),
        encoding="utf-8",
    )
    (output_dir / "ensemble_manifest.json").write_text(
        json.dumps(
            {
                "run_dir": str(args.run_dir),
                "top_k": args.top_k,
                "n_models": len(models),
                "checkpoint_paths": [str(path) for path in checkpoint_paths],
                "n_concepts": int(concept_embeddings.shape[0]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
