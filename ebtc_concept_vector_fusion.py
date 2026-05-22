#!/usr/bin/env python3
"""Fuse raw original_top10 and repaired concept-vector decisions.

The repaired z_margin bank fixes LGC collapse but weakens HGC/NTL. The raw
original_top10 bank has the opposite behavior. This script keeps the evaluation
rule interpretable by fusing only top10 class-count evidence from the two banks.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ebtc_cosine_topk_assignment import (
    ConceptBank,
    compute_topk_tables,
    dataframe_to_markdown,
    load_concept_bank,
    load_stage_split_payloads,
    predict_majority_vote,
)
from ebtc_discriminative_whitelist_cycl import CLASSES, l2_normalize
from ebtc_project_paths import OFFICIAL_EMBEDDINGS_DIR, OUTPUT_ROOT


DEFAULT_REFINED_STAGE_DIR = OUTPUT_ROOT / "ebtc_embedding_refinement_stage_conservative_outputs"
DEFAULT_REPAIRED_BANK_DIR = OUTPUT_ROOT / "ebtc_confusion_aware_bank_repair_outputs" / "banks" / "z_margin"
DEFAULT_OUTPUT_DIR = OUTPUT_ROOT / "ebtc_concept_vector_fusion_outputs"
EPS = 1e-8


@dataclass
class CountPayload:
    counts: np.ndarray
    mean_scores: np.ndarray
    preds: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fuse raw and repaired top10 concept-vector evidence.")
    parser.add_argument("--refined-stage-dir", type=Path, default=DEFAULT_REFINED_STAGE_DIR)
    parser.add_argument("--embeddings-dir", type=Path, default=OFFICIAL_EMBEDDINGS_DIR)
    parser.add_argument("--repaired-bank-dir", type=Path, default=DEFAULT_REPAIRED_BANK_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--fusion-weights", default="0.70,0.75,0.80,0.85,0.90,0.95,1.00")
    parser.add_argument("--gate-thresholds", default="0,1,2")
    parser.add_argument(
        "--selection-macro-tolerance",
        type=float,
        default=0.005,
        help="Among rows within this validation Macro-F1 of the best, select the highest validation accuracy.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def safe_stem(text: str) -> str:
    return (
        text.replace(".", "p")
        .replace("-", "m")
        .replace("+", "p")
        .replace("/", "_")
        .replace(" ", "_")
    )


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def load_repaired_bank(bank_dir: Path) -> ConceptBank:
    npz_path = bank_dir / "whitelist_top10.npz"
    csv_path = bank_dir / "whitelist_top10.csv"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing repaired bank npz: {npz_path}")
    payload = np.load(npz_path, allow_pickle=True)
    metadata = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()
    return ConceptBank(
        stage=bank_dir.name,
        concepts=payload["concepts"].astype(object),
        embeddings=l2_normalize(payload["concept_embeddings"].astype(np.float32)),
        labels=payload["labels"].astype(np.int64),
        metadata=metadata,
        whitelist_npz=npz_path,
        whitelist_csv=csv_path,
    )


def compute_count_payloads(
    bank: ConceptBank,
    split_payloads: dict[str, Any],
    top_k: int,
) -> dict[str, CountPayload]:
    out: dict[str, CountPayload] = {}
    for split, payload in split_payloads.items():
        vectors = payload.embeddings @ bank.embeddings.T
        _top_indices, counts, mean_scores, _mean_scores_for_auc = compute_topk_tables(vectors, bank.labels, top_k=top_k)
        preds = predict_majority_vote(counts, mean_scores)
        out[split] = CountPayload(counts=counts, mean_scores=mean_scores, preds=preds)
    return out


def score_to_probs(scores: np.ndarray) -> np.ndarray:
    scores = np.clip(scores.astype(np.float64), 0.0, None)
    return scores / np.clip(scores.sum(axis=1, keepdims=True), EPS, None)


def evaluate_predictions(
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
        row["macro_auroc"] = float(roc_auc_score(labels, score_to_probs(scores), multi_class="ovr", average="macro", labels=list(range(len(CLASSES)))))
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
    cm_dir = output_dir / "confusion_matrices"
    ensure_dir(cm_dir)
    matrix = confusion_matrix(labels, preds, labels=list(range(len(CLASSES))))
    base = cm_dir / f"confusion_{safe_stem(source)}_{safe_stem(split)}_{safe_stem(rule)}"
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


def save_count_matrix(output_dir: Path, source: str, split: str, labels: np.ndarray, scores: np.ndarray) -> None:
    out_dir = output_dir / "topk_class_count_matrices"
    ensure_dir(out_dir)
    rows: list[dict[str, Any]] = []
    for true_idx, true_class in enumerate(CLASSES):
        mask = labels == true_idx
        means = scores[mask].mean(axis=0) if bool(mask.any()) else np.zeros(len(CLASSES), dtype=np.float32)
        row = {"source": source, "split": split, "true_image_class": true_class}
        for class_idx, class_name in enumerate(CLASSES):
            row[f"mean_score_{class_name}"] = float(means[class_idx])
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / f"topk_counts_{safe_stem(source)}_{safe_stem(split)}.csv", index=False)


def fusion_outputs(
    mode: str,
    raw_payload: CountPayload,
    repaired_payload: CountPayload,
    parameter: float,
) -> tuple[np.ndarray, np.ndarray]:
    if mode == "raw_majority":
        return raw_payload.preds, raw_payload.counts
    if mode == "repaired_majority":
        return repaired_payload.preds, repaired_payload.counts
    if mode == "count_fusion":
        scores = (1.0 - parameter) * raw_payload.counts + parameter * repaired_payload.counts
        return scores.argmax(axis=1).astype(np.int64), scores
    if mode == "lgc_confidence_gate":
        preds = raw_payload.preds.copy()
        scores = raw_payload.counts.copy()
        use_repaired = (repaired_payload.counts[:, CLASSES.index("LGC")] - repaired_payload.counts[:, CLASSES.index("HGC")]) >= parameter
        preds[use_repaired] = repaired_payload.preds[use_repaired]
        scores[use_repaired] = repaired_payload.counts[use_repaired]
        return preds, scores
    raise ValueError(f"Unsupported fusion mode: {mode}")


def select_strategy(results: pd.DataFrame, macro_tolerance: float) -> tuple[str, str, float]:
    candidates = results[results["split"].eq("val")].copy()
    best_macro = float(candidates["macro_f1"].max())
    candidates = candidates[candidates["macro_f1"] >= best_macro - macro_tolerance]
    candidates = candidates.sort_values(["accuracy", "macro_f1", "LGC_f1", "NTL_f1"], ascending=False)
    row = candidates.iloc[0]
    return str(row["source"]), str(row["rule"]), float(row["parameter"])


def write_report(
    output_dir: Path,
    args: argparse.Namespace,
    results: pd.DataFrame,
    selected_source: str,
) -> None:
    metric_cols = {"accuracy", "macro_f1", "macro_auroc", "HGC_f1", "LGC_f1", "NTL_f1", "NST_f1", "parameter"}
    val_rows = results[results["split"].eq("val")].sort_values(["macro_f1", "accuracy"], ascending=False).head(12)
    test_selected = results[(results["split"].eq("test")) & (results["source"].isin(["raw_majority", "repaired_majority", selected_source]))]
    lines = [
        "# Concept Vector Fusion",
        "",
        "## Setup",
        "",
        "- Raw source: current refined `original_top10` bank.",
        f"- Repaired source: `{args.repaired_bank_dir}`.",
        "- Evidence type: top10 class-count vectors only.",
        "- Strategy selection: validation Macro-F1 with an accuracy tie-break inside the configured tolerance.",
        f"- Macro-F1 tolerance: `{args.selection_macro_tolerance}`.",
        "",
        "## Validation Selection",
        "",
        dataframe_to_markdown(val_rows[["source", "rule", "parameter", "accuracy", "macro_f1", "macro_auroc", "HGC_f1", "LGC_f1", "NTL_f1", "NST_f1"]], metric_cols),
        "",
        f"Selected source: `{selected_source}`.",
        "",
        "## Test Results",
        "",
        dataframe_to_markdown(test_selected[["source", "rule", "parameter", "accuracy", "macro_f1", "macro_auroc", "HGC_f1", "LGC_f1", "NTL_f1", "NST_f1"]], metric_cols),
        "",
        "## Output Files",
        "",
        f"- Results: `{output_dir / 'concept_vector_fusion_results.csv'}`",
        f"- Confusion matrices: `{output_dir / 'confusion_matrices'}`",
        f"- Count matrices: `{output_dir / 'topk_class_count_matrices'}`",
    ]
    (output_dir / "concept_vector_fusion_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    weights = parse_float_list(args.fusion_weights)
    thresholds = parse_float_list(args.gate_thresholds)
    write_json(
        args.output_dir / "experiment_config.json",
        {
            "refined_stage_dir": str(args.refined_stage_dir),
            "embeddings_dir": str(args.embeddings_dir),
            "repaired_bank_dir": str(args.repaired_bank_dir),
            "output_dir": str(args.output_dir),
            "top_k": args.top_k,
            "fusion_weights": weights,
            "gate_thresholds": thresholds,
            "selection_macro_tolerance": args.selection_macro_tolerance,
        },
    )

    payloads = load_stage_split_payloads(args.refined_stage_dir, args.embeddings_dir, "refined_vectors_original_top10")
    raw_bank = load_concept_bank(args.refined_stage_dir, "refined_vectors_original_top10")
    repaired_bank = load_repaired_bank(args.repaired_bank_dir)
    raw_counts = compute_count_payloads(raw_bank, payloads, args.top_k)
    repaired_counts = compute_count_payloads(repaired_bank, payloads, args.top_k)

    result_rows: list[dict[str, Any]] = []
    for mode, parameters in [
        ("raw_majority", [0.0]),
        ("repaired_majority", [1.0]),
        ("count_fusion", weights),
        ("lgc_confidence_gate", thresholds),
    ]:
        for parameter in parameters:
            source = mode if mode in {"raw_majority", "repaired_majority"} else f"{mode}_{parameter:g}"
            for split, payload in payloads.items():
                preds, scores = fusion_outputs(mode, raw_counts[split], repaired_counts[split], parameter)
                row = evaluate_predictions(source, split, "majority_vote", payload.labels, preds, scores)
                row["parameter"] = float(parameter)
                result_rows.append(row)
                save_confusion(args.output_dir, source, split, "majority_vote", payload.labels, preds)
                save_count_matrix(args.output_dir, source, split, payload.labels, scores)

    results = pd.DataFrame(result_rows)
    selected_source, _selected_rule, selected_parameter = select_strategy(results, args.selection_macro_tolerance)
    results.to_csv(args.output_dir / "concept_vector_fusion_results.csv", index=False)
    write_json(
        args.output_dir / "selected_strategy.json",
        {
            "selected_source": selected_source,
            "selected_parameter": selected_parameter,
            "selection_macro_tolerance": args.selection_macro_tolerance,
        },
    )
    write_report(args.output_dir, args, results, selected_source)
    print(json.dumps({"output_dir": str(args.output_dir), "selected_source": selected_source}, indent=2))


if __name__ == "__main__":
    main()
