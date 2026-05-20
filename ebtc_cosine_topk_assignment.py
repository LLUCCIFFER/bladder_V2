"""Cosine-only top-k concept assignment diagnostics.

This script implements the two non-training classification rules requested for
the concept activation vector:

1. Top-k majority vote over concept source classes.
2. Top-k class-average cosine similarity.

It intentionally does not train a model. It reads cached image embeddings and a
fixed top-k concept bank, computes image-concept cosine similarities, and writes
metrics, confusion matrices, per-sample predictions, and a short report.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from ebtc_discriminative_whitelist_cycl import CLASSES, CLASS_TO_INDEX, SplitPayload, l2_normalize, load_split_payloads
from ebtc_project_paths import OFFICIAL_EMBEDDINGS_DIR, OUTPUT_ROOT


DEFAULT_REFINED_STAGE_DIR = OUTPUT_ROOT / "ebtc_embedding_refinement_stage_conservative_outputs"
DEFAULT_OUTPUT_DIR = OUTPUT_ROOT / "ebtc_cosine_topk_assignment_outputs"

STAGE_TO_WHITELIST_DIR = {
    "original_vectors_original_top10": "original_filtering",
    "refined_vectors_original_top10": "refined_vectors_original_whitelist",
    "refined_vectors_refined_top10": "refined_filtering",
}


@dataclass
class ConceptBank:
    stage: str
    concepts: np.ndarray
    embeddings: np.ndarray
    labels: np.ndarray
    metadata: pd.DataFrame
    whitelist_npz: Path
    whitelist_csv: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cosine-only top-k concept assignment diagnostics.")
    parser.add_argument("--refined-stage-dir", type=Path, default=DEFAULT_REFINED_STAGE_DIR)
    parser.add_argument("--embeddings-dir", type=Path, default=OFFICIAL_EMBEDDINGS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--stages",
        default="refined_vectors_original_top10,original_vectors_original_top10,refined_vectors_refined_top10",
        help="Comma-separated stages to evaluate.",
    )
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_stage_list(text: str) -> list[str]:
    stages = [item.strip() for item in text.split(",") if item.strip()]
    unknown = [stage for stage in stages if stage not in STAGE_TO_WHITELIST_DIR]
    if unknown:
        raise ValueError(f"Unknown stage(s): {unknown}. Available: {sorted(STAGE_TO_WHITELIST_DIR)}")
    return stages


def load_stage_split_payloads(refined_stage_dir: Path, embeddings_dir: Path, stage: str) -> dict[str, SplitPayload]:
    """Load original or refined image embeddings for a stage."""

    payloads = load_split_payloads(embeddings_dir)
    if stage.startswith("refined_vectors_"):
        adapter_dir = refined_stage_dir / "adapter_refinement"
        for split in ["train", "val", "test"]:
            refined_path = adapter_dir / f"refined_image_embeddings_{split}.npy"
            payloads[split].embeddings = l2_normalize(np.load(refined_path).astype(np.float32))
    return payloads


def load_concept_bank(refined_stage_dir: Path, stage: str) -> ConceptBank:
    """Load the stage-specific top10 concept bank and class provenance."""

    subdir = STAGE_TO_WHITELIST_DIR[stage]
    bank_dir = refined_stage_dir / subdir
    whitelist_npz = bank_dir / "whitelist_top10.npz"
    whitelist_csv = bank_dir / "whitelist_top10.csv"
    if not whitelist_npz.exists() or not whitelist_csv.exists():
        raise FileNotFoundError(f"Missing whitelist files under {bank_dir}")

    payload = np.load(whitelist_npz, allow_pickle=True)
    metadata = pd.read_csv(whitelist_csv)
    if "primary_target_class" not in metadata.columns:
        raise RuntimeError(f"{whitelist_csv} does not contain primary_target_class.")
    labels = np.array([CLASS_TO_INDEX[str(item)] for item in metadata["primary_target_class"].tolist()], dtype=np.int64)
    concepts = payload["concepts"].astype(object)
    embeddings = l2_normalize(payload["concept_embeddings"].astype(np.float32))
    if len(concepts) != len(labels) or embeddings.shape[0] != len(labels):
        raise RuntimeError(
            f"Concept metadata mismatch for {stage}: concepts={len(concepts)} embeddings={embeddings.shape[0]} labels={len(labels)}"
        )
    return ConceptBank(
        stage=stage,
        concepts=concepts,
        embeddings=embeddings,
        labels=labels,
        metadata=metadata,
        whitelist_npz=whitelist_npz,
        whitelist_csv=whitelist_csv,
    )


def softmax_scores(scores: np.ndarray) -> np.ndarray:
    stable = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(stable)
    return exp_scores / np.clip(exp_scores.sum(axis=1, keepdims=True), 1e-12, None)


def compute_topk_tables(
    similarities: np.ndarray,
    concept_labels: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return sorted top-k indices, class counts, class means, and class mean scores."""

    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    if top_k > similarities.shape[1]:
        raise ValueError(f"top_k={top_k} exceeds number of concepts={similarities.shape[1]}.")

    top_indices = np.argsort(-similarities, axis=1)[:, :top_k]
    counts = np.zeros((similarities.shape[0], len(CLASSES)), dtype=np.float32)
    mean_scores = np.full((similarities.shape[0], len(CLASSES)), -np.inf, dtype=np.float32)
    mean_scores_for_auc = np.zeros((similarities.shape[0], len(CLASSES)), dtype=np.float32)

    for row_idx, concept_indices in enumerate(top_indices):
        top_sims = similarities[row_idx, concept_indices]
        sample_min = float(np.min(top_sims)) - 1e-6
        mean_scores_for_auc[row_idx, :] = sample_min
        for class_idx in range(len(CLASSES)):
            mask = concept_labels[concept_indices] == class_idx
            counts[row_idx, class_idx] = float(mask.sum())
            if mask.any():
                mean_value = float(top_sims[mask].mean())
                mean_scores[row_idx, class_idx] = mean_value
                mean_scores_for_auc[row_idx, class_idx] = mean_value
    return top_indices, counts, mean_scores, mean_scores_for_auc


def predict_majority_vote(counts: np.ndarray, mean_scores: np.ndarray) -> np.ndarray:
    """Predict by majority class in top-k, using top-k mean similarity as tie-break."""

    preds: list[int] = []
    for row_idx in range(counts.shape[0]):
        max_count = counts[row_idx].max()
        candidates = np.flatnonzero(counts[row_idx] == max_count)
        if len(candidates) == 1:
            preds.append(int(candidates[0]))
            continue
        candidate_scores = mean_scores[row_idx, candidates]
        preds.append(int(candidates[int(np.argmax(candidate_scores))]))
    return np.array(preds, dtype=np.int64)


def predict_class_average(mean_scores: np.ndarray) -> np.ndarray:
    """Predict by highest mean cosine similarity among classes present in top-k."""

    return np.argmax(mean_scores, axis=1).astype(np.int64)


def metric_payload(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    score_matrix: np.ndarray,
    stage: str,
    split: str,
    rule: str,
    top_k: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "stage": stage,
        "split": split,
        "rule": rule,
        "top_k": int(top_k),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    try:
        if rule == "majority_vote":
            probs = score_matrix / np.clip(score_matrix.sum(axis=1, keepdims=True), 1e-12, None)
        else:
            probs = softmax_scores(score_matrix)
        row["macro_auroc"] = float(
            roc_auc_score(y_true, probs, multi_class="ovr", average="macro", labels=list(range(len(CLASSES))))
        )
    except ValueError:
        row["macro_auroc"] = ""
    return row


def per_class_rows(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    counts: np.ndarray,
    stage: str,
    split: str,
    rule: str,
) -> list[dict[str, Any]]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(CLASSES))),
        zero_division=0,
    )
    rows: list[dict[str, Any]] = []
    for class_idx, class_name in enumerate(CLASSES):
        rows.append(
            {
                "stage": stage,
                "split": split,
                "rule": rule,
                "class": class_name,
                "precision": float(precision[class_idx]),
                "recall": float(recall[class_idx]),
                "f1": float(f1[class_idx]),
                "support": int(support[class_idx]),
                "mean_topk_count_for_concept_class": float(counts[:, class_idx].mean()),
            }
        )
    return rows


def save_confusion_outputs(output_dir: Path, cm: np.ndarray, stage: str, split: str, rule: str) -> None:
    cm_dir = output_dir / "confusion_matrices"
    ensure_dir(cm_dir)
    df = pd.DataFrame(cm, index=CLASSES, columns=CLASSES)
    df.to_csv(cm_dir / f"confusion_{stage}_{split}_{rule}.csv")

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES)), labels=CLASSES)
    ax.set_yticks(range(len(CLASSES)), labels=CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{stage} {split} {rule}")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(cm_dir / f"confusion_{stage}_{split}_{rule}.png", dpi=180)
    plt.close(fig)


def save_topk_count_matrix(output_dir: Path, labels: np.ndarray, counts: np.ndarray, stage: str, split: str) -> None:
    rows: list[dict[str, Any]] = []
    matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.float32)
    for true_idx, true_class in enumerate(CLASSES):
        mask = labels == true_idx
        if mask.any():
            matrix[true_idx] = counts[mask].mean(axis=0)
        row = {"stage": stage, "split": split, "true_image_class": true_class}
        for concept_idx, concept_class in enumerate(CLASSES):
            row[f"mean_topk_count_{concept_class}_concepts"] = float(matrix[true_idx, concept_idx])
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / "topk_class_count_matrices" / f"topk_counts_{stage}_{split}.csv", index=False)


def prediction_rows(
    payload: SplitPayload,
    stage: str,
    split: str,
    bank: ConceptBank,
    similarities: np.ndarray,
    top_indices: np.ndarray,
    counts: np.ndarray,
    mean_scores: np.ndarray,
    majority_preds: np.ndarray,
    average_preds: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, manifest_row in enumerate(payload.rows):
        top = top_indices[idx]
        row: dict[str, Any] = {
            "stage": stage,
            "split": split,
            "image_id": manifest_row.get("image_id", ""),
            "image_path": manifest_row.get("image_path", ""),
            "true_class": CLASSES[int(payload.labels[idx])],
            "majority_vote_pred": CLASSES[int(majority_preds[idx])],
            "class_average_pred": CLASSES[int(average_preds[idx])],
            "topk_concept_classes": ";".join(CLASSES[int(bank.labels[item])] for item in top),
            "topk_concepts": ";".join(str(bank.concepts[item]) for item in top),
            "topk_similarities": ";".join(f"{float(similarities[idx, item]):.6f}" for item in top),
        }
        for class_idx, class_name in enumerate(CLASSES):
            row[f"{class_name}_topk_count"] = int(counts[idx, class_idx])
            value = mean_scores[idx, class_idx]
            row[f"{class_name}_topk_mean_similarity"] = "" if not np.isfinite(value) else f"{float(value):.6f}"
        rows.append(row)
    return rows


def run_stage(
    output_dir: Path,
    refined_stage_dir: Path,
    embeddings_dir: Path,
    stage: str,
    top_k: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    bank = load_concept_bank(refined_stage_dir, stage)
    payloads = load_stage_split_payloads(refined_stage_dir, embeddings_dir, stage)

    bank_manifest = {
        "stage": stage,
        "whitelist_npz": str(bank.whitelist_npz),
        "whitelist_csv": str(bank.whitelist_csv),
        "n_concepts": int(len(bank.concepts)),
        **{f"n_{class_name}_concepts": int((bank.labels == idx).sum()) for idx, class_name in enumerate(CLASSES)},
    }
    ensure_dir(output_dir / "bank_manifests")
    (output_dir / "bank_manifests" / f"{stage}.json").write_text(json.dumps(bank_manifest, indent=2) + "\n", encoding="utf-8")
    bank.metadata.to_csv(output_dir / "bank_manifests" / f"{stage}_concept_metadata.csv", index=False)

    metric_rows: list[dict[str, Any]] = []
    per_class_metric_rows: list[dict[str, Any]] = []
    all_predictions: list[dict[str, Any]] = []
    ensure_dir(output_dir / "topk_class_count_matrices")
    ensure_dir(output_dir / "predictions")

    for split, payload in payloads.items():
        similarities = payload.embeddings @ bank.embeddings.T
        top_indices, counts, mean_scores, mean_scores_for_auc = compute_topk_tables(similarities, bank.labels, top_k=top_k)
        majority_preds = predict_majority_vote(counts, mean_scores)
        class_average_preds = predict_class_average(mean_scores)

        rules = {
            "majority_vote": (majority_preds, counts),
            "class_average": (class_average_preds, mean_scores_for_auc),
        }
        for rule, (preds, scores) in rules.items():
            metric_rows.append(metric_payload(payload.labels, preds, scores, stage, split, rule, top_k))
            per_class_metric_rows.extend(per_class_rows(payload.labels, preds, counts, stage, split, rule))
            cm = confusion_matrix(payload.labels, preds, labels=list(range(len(CLASSES))))
            save_confusion_outputs(output_dir, cm, stage, split, rule)

        save_topk_count_matrix(output_dir, payload.labels, counts, stage, split)
        all_predictions.extend(
            prediction_rows(
                payload=payload,
                stage=stage,
                split=split,
                bank=bank,
                similarities=similarities,
                top_indices=top_indices,
                counts=counts,
                mean_scores=mean_scores,
                majority_preds=majority_preds,
                average_preds=class_average_preds,
            )
        )

    pd.DataFrame(all_predictions).to_csv(output_dir / "predictions" / f"predictions_{stage}.csv", index=False)
    return metric_rows, per_class_metric_rows


def dataframe_to_markdown(df: pd.DataFrame, float_cols: set[str] | None = None) -> str:
    if df.empty:
        return ""
    float_cols = float_cols or set()
    lines = ["| " + " | ".join(df.columns) + " |", "| " + " | ".join("---" for _ in df.columns) + " |"]
    for _, row in df.iterrows():
        values: list[str] = []
        for col in df.columns:
            value = row[col]
            if col in float_cols and pd.notna(value) and value != "":
                values.append(f"{float(value):.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(output_dir: Path, stages: list[str], top_k: int, results: pd.DataFrame, per_class: pd.DataFrame) -> None:
    metric_cols = ["accuracy", "macro_f1", "macro_auroc"]
    val_test = results[results["split"].isin(["val", "test"])].copy()
    test = results[results["split"] == "test"].copy()
    best = test.sort_values(["macro_f1", "accuracy"], ascending=False).head(1)

    lines = [
        "# Cosine-Only Top-k Concept Assignment",
        "",
        "## Setup",
        "",
        f"- Top-k value: `{top_k}`.",
        f"- Evaluated stages: `{', '.join(stages)}`.",
        "- No classifier is trained in this experiment.",
        "- Rule 1, `majority_vote`: choose the class that appears most often among the global top-k concepts; ties are broken by within-top-k mean cosine.",
        "- Rule 2, `class_average`: among the global top-k concepts, average cosine similarities by concept class and choose the largest class mean.",
        "",
        "## Main Val/Test Results",
        "",
        dataframe_to_markdown(val_test[["stage", "split", "rule", "accuracy", "macro_f1", "macro_auroc"]], set(metric_cols)),
        "",
        "## Best Test Row",
        "",
        dataframe_to_markdown(best[["stage", "rule", "accuracy", "macro_f1", "macro_auroc"]], set(metric_cols)),
        "",
        "## Test Per-Class F1",
        "",
    ]
    test_per_class = per_class[per_class["split"] == "test"].pivot_table(
        index=["stage", "rule"], columns="class", values="f1", aggfunc="first"
    )
    test_per_class = test_per_class.reset_index()
    lines.append(dataframe_to_markdown(test_per_class, set(CLASSES)))
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            f"- Summary metrics: `{output_dir / 'cosine_topk_assignment_results.csv'}`",
            f"- Per-class metrics: `{output_dir / 'cosine_topk_assignment_per_class.csv'}`",
            f"- Confusion matrices: `{output_dir / 'confusion_matrices'}`",
            f"- Per-sample predictions and top-k concepts: `{output_dir / 'predictions'}`",
            f"- Top-k concept class count matrices: `{output_dir / 'topk_class_count_matrices'}`",
            "",
            "## Interpretation Notes",
            "",
            "- These results test whether the concept activation vector itself is class-structured enough for rule-based classification.",
            "- If these results are weaker than CBM/CyCL, the learned classifier is compensating for noisy or shared concepts rather than simply reading off the majority class.",
        ]
    )
    (output_dir / "cosine_topk_assignment_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    stages = parse_stage_list(args.stages)
    ensure_dir(args.output_dir)
    config = {
        "refined_stage_dir": str(args.refined_stage_dir),
        "embeddings_dir": str(args.embeddings_dir),
        "output_dir": str(args.output_dir),
        "stages": stages,
        "top_k": args.top_k,
        "classes": CLASSES,
    }
    (args.output_dir / "experiment_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    metric_rows: list[dict[str, Any]] = []
    per_class_rows_all: list[dict[str, Any]] = []
    for stage in stages:
        rows, class_rows = run_stage(
            output_dir=args.output_dir,
            refined_stage_dir=args.refined_stage_dir,
            embeddings_dir=args.embeddings_dir,
            stage=stage,
            top_k=args.top_k,
        )
        metric_rows.extend(rows)
        per_class_rows_all.extend(class_rows)

    results = pd.DataFrame(metric_rows)
    per_class = pd.DataFrame(per_class_rows_all)
    results.to_csv(args.output_dir / "cosine_topk_assignment_results.csv", index=False)
    per_class.to_csv(args.output_dir / "cosine_topk_assignment_per_class.csv", index=False)
    write_report(args.output_dir, stages, args.top_k, results, per_class)
    print(f"Wrote cosine-only top-k assignment outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
