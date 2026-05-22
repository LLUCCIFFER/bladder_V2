#!/usr/bin/env python3
"""Confusion-aware repair for the fixed 4x10 EBTC concept bank.

This script keeps the same 40-d concept-vector format as the current
original_top10 bank, but replaces concepts from the filtered_top300 candidate
pool using train-only class-separation statistics. The repaired bank is then
evaluated with the same top10 majority-vote and class-average rules used by the
cosine-only diagnostics.
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
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ebtc_cosine_topk_assignment import (
    ConceptBank,
    compute_topk_tables,
    dataframe_to_markdown,
    load_concept_bank,
    load_stage_split_payloads,
    predict_class_average,
    predict_majority_vote,
    softmax_scores,
)
from ebtc_discriminative_whitelist_cycl import CLASSES, CLASS_TO_INDEX, SplitPayload, l2_normalize
from ebtc_project_paths import FILTERED_TOP300_BANK_DIR, OFFICIAL_EMBEDDINGS_DIR, OUTPUT_ROOT


DEFAULT_REFINED_STAGE_DIR = OUTPUT_ROOT / "ebtc_embedding_refinement_stage_conservative_outputs"
DEFAULT_OUTPUT_DIR = OUTPUT_ROOT / "ebtc_confusion_aware_bank_repair_outputs"
DEFAULT_STRATEGIES = "hard_margin,z_margin,pair_z,hybrid_z_margin,own_rank,own_minus_hgc_for_lgc"
CONCEPTS_PER_CLASS = 10
EPS = 1e-8


@dataclass
class CandidateBank:
    concepts: np.ndarray
    embeddings: np.ndarray
    labels: np.ndarray
    metadata: pd.DataFrame


@dataclass
class TrainStats:
    class_means: np.ndarray
    class_stds: np.ndarray
    train_similarities: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run confusion-aware concept-bank repair diagnostics.")
    parser.add_argument("--refined-stage-dir", type=Path, default=DEFAULT_REFINED_STAGE_DIR)
    parser.add_argument("--embeddings-dir", type=Path, default=OFFICIAL_EMBEDDINGS_DIR)
    parser.add_argument("--candidate-bank-dir", type=Path, default=FILTERED_TOP300_BANK_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--strategies", default=DEFAULT_STRATEGIES)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--concepts-per-class", type=int, default=CONCEPTS_PER_CLASS)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_csv_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def load_refined_split_payloads(refined_stage_dir: Path, embeddings_dir: Path) -> dict[str, SplitPayload]:
    payloads = load_stage_split_payloads(refined_stage_dir, embeddings_dir, "refined_vectors_original_top10")
    return payloads


def load_candidate_bank(refined_stage_dir: Path, candidate_bank_dir: Path) -> CandidateBank:
    metadata_path = candidate_bank_dir / "merged_concepts.csv"
    refined_text_path = refined_stage_dir / "adapter_refinement" / "refined_filtered_top300_text_embeddings.npz"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing candidate metadata: {metadata_path}")
    if not refined_text_path.exists():
        raise FileNotFoundError(f"Missing refined text embeddings: {refined_text_path}")

    metadata = pd.read_csv(metadata_path)
    payload = np.load(refined_text_path, allow_pickle=True)
    concepts = payload["concepts"].astype(object)
    embeddings = l2_normalize(payload["concept_embeddings"].astype(np.float32))
    labels = payload["concept_labels"].astype(np.int64)

    if len(metadata) != embeddings.shape[0] or len(concepts) != embeddings.shape[0]:
        raise RuntimeError(
            f"Candidate bank mismatch: metadata={len(metadata)}, concepts={len(concepts)}, embeddings={embeddings.shape[0]}"
        )
    metadata_labels = np.array([CLASS_TO_INDEX[str(item)] for item in metadata["primary_class"].tolist()], dtype=np.int64)
    if not np.array_equal(labels, metadata_labels):
        raise RuntimeError("Refined text concept labels do not match candidate metadata primary_class labels.")

    return CandidateBank(concepts=concepts, embeddings=embeddings, labels=labels, metadata=metadata)


def compute_train_stats(train_payload: SplitPayload, bank: CandidateBank) -> TrainStats:
    sims = train_payload.embeddings @ bank.embeddings.T
    class_means = np.zeros((bank.embeddings.shape[0], len(CLASSES)), dtype=np.float32)
    class_stds = np.zeros((bank.embeddings.shape[0], len(CLASSES)), dtype=np.float32)
    for class_idx in range(len(CLASSES)):
        mask = train_payload.labels == class_idx
        class_means[:, class_idx] = sims[mask].mean(axis=0)
        class_stds[:, class_idx] = sims[mask].std(axis=0) + EPS
    return TrainStats(class_means=class_means, class_stds=class_stds, train_similarities=sims.astype(np.float32))


def score_candidates(strategy: str, bank: CandidateBank, stats: TrainStats) -> np.ndarray:
    scores = np.zeros(bank.embeddings.shape[0], dtype=np.float32)
    means = stats.class_means
    stds = stats.class_stds
    for concept_idx, class_idx in enumerate(bank.labels):
        own = float(means[concept_idx, class_idx])
        neg_indices = [idx for idx in range(len(CLASSES)) if idx != int(class_idx)]
        hard_neg = max(float(means[concept_idx, idx]) for idx in neg_indices)
        hard_margin = own - hard_neg

        if strategy == "hard_margin":
            score = hard_margin
        elif strategy == "own_rank":
            score = own
        elif strategy == "z_margin":
            neg_mean = float(np.mean([means[concept_idx, idx] for idx in neg_indices]))
            neg_std = float(np.mean([stds[concept_idx, idx] for idx in neg_indices]))
            score = (own - neg_mean) / max(float(stds[concept_idx, class_idx]) + neg_std, EPS)
        elif strategy == "pair_z":
            if class_idx == CLASS_TO_INDEX["LGC"]:
                score = pairwise_z(concept_idx, class_idx, CLASS_TO_INDEX["HGC"], means, stds)
            elif class_idx == CLASS_TO_INDEX["NTL"]:
                score = min(
                    pairwise_z(concept_idx, class_idx, CLASS_TO_INDEX["HGC"], means, stds),
                    pairwise_z(concept_idx, class_idx, CLASS_TO_INDEX["LGC"], means, stds),
                )
            elif class_idx == CLASS_TO_INDEX["HGC"]:
                score = pairwise_z(concept_idx, class_idx, CLASS_TO_INDEX["LGC"], means, stds)
            else:
                neg_std = float(np.mean([stds[concept_idx, idx] for idx in neg_indices]))
                score = hard_margin / max(float(stds[concept_idx, class_idx]) + neg_std, EPS)
        elif strategy == "hybrid_z_margin":
            neg_mean = float(np.mean([means[concept_idx, idx] for idx in neg_indices]))
            neg_std = float(np.mean([stds[concept_idx, idx] for idx in neg_indices]))
            z_margin = (own - neg_mean) / max(float(stds[concept_idx, class_idx]) + neg_std, EPS)
            if class_idx == CLASS_TO_INDEX["LGC"]:
                score = z_margin + 1.5 * pairwise_z(concept_idx, class_idx, CLASS_TO_INDEX["HGC"], means, stds)
            else:
                score = z_margin
        elif strategy == "own_minus_hgc_for_lgc":
            if class_idx == CLASS_TO_INDEX["LGC"]:
                score = 2.0 * (own - float(means[concept_idx, CLASS_TO_INDEX["HGC"]])) + hard_margin + 0.1 * own
            elif class_idx == CLASS_TO_INDEX["NTL"]:
                score = (
                    2.0
                    * (own - max(float(means[concept_idx, CLASS_TO_INDEX["HGC"]]), float(means[concept_idx, CLASS_TO_INDEX["LGC"]])))
                    + hard_margin
                    + 0.1 * own
                )
            elif class_idx == CLASS_TO_INDEX["HGC"]:
                score = (
                    1.5 * (own - float(means[concept_idx, CLASS_TO_INDEX["LGC"]]))
                    + 0.5 * (own - float(means[concept_idx, CLASS_TO_INDEX["NTL"]]))
                    + hard_margin
                    + 0.1 * own
                )
            else:
                score = hard_margin + 0.2 * own
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")
        scores[concept_idx] = float(score)
    return scores


def pairwise_z(concept_idx: int, own_class: int, negative_class: int, means: np.ndarray, stds: np.ndarray) -> float:
    return float(means[concept_idx, own_class] - means[concept_idx, negative_class]) / max(
        float(stds[concept_idx, own_class] + stds[concept_idx, negative_class]),
        EPS,
    )


def select_bank(strategy: str, bank: CandidateBank, scores: np.ndarray, concepts_per_class: int) -> np.ndarray:
    selected: list[int] = []
    for class_idx in range(len(CLASSES)):
        class_candidates = np.flatnonzero(bank.labels == class_idx)
        ordered = class_candidates[np.argsort(-scores[class_candidates])]
        selected.extend(ordered[:concepts_per_class].tolist())
    selected_array = np.array(selected, dtype=np.int64)
    expected_labels = np.repeat(np.arange(len(CLASSES), dtype=np.int64), concepts_per_class)
    if not np.array_equal(bank.labels[selected_array], expected_labels):
        raise RuntimeError(f"Selected bank for {strategy} does not preserve 4 class blocks.")
    return selected_array


def save_selected_bank(
    output_dir: Path,
    strategy: str,
    bank: CandidateBank,
    selected_indices: np.ndarray,
    scores: np.ndarray,
    stats: TrainStats,
) -> ConceptBank:
    bank_dir = output_dir / "banks" / strategy
    ensure_dir(bank_dir)
    metadata = bank.metadata.iloc[selected_indices].copy().reset_index(drop=True)
    metadata.insert(0, "whitelist_index", np.arange(len(metadata)))
    metadata["repair_strategy"] = strategy
    metadata["repair_score"] = scores[selected_indices]
    for class_idx, class_name in enumerate(CLASSES):
        metadata[f"train_mu_{class_name}"] = stats.class_means[selected_indices, class_idx]
        metadata[f"train_std_{class_name}"] = stats.class_stds[selected_indices, class_idx]
    metadata.to_csv(bank_dir / "whitelist_top10.csv", index=False)
    np.savez_compressed(
        bank_dir / "whitelist_top10.npz",
        concepts=bank.concepts[selected_indices],
        concept_embeddings=bank.embeddings[selected_indices].astype(np.float32),
        labels=bank.labels[selected_indices].astype(np.int64),
        selected_indices=selected_indices.astype(np.int64),
        scores=scores[selected_indices].astype(np.float32),
        classes=np.array(CLASSES, dtype=object),
    )
    return ConceptBank(
        stage=strategy,
        concepts=bank.concepts[selected_indices],
        embeddings=bank.embeddings[selected_indices].astype(np.float32),
        labels=bank.labels[selected_indices].astype(np.int64),
        metadata=metadata,
        whitelist_npz=bank_dir / "whitelist_top10.npz",
        whitelist_csv=bank_dir / "whitelist_top10.csv",
    )


def evaluate_bank(
    output_dir: Path,
    source_name: str,
    bank: ConceptBank,
    payloads: dict[str, SplitPayload],
    top_k: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metric_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    ensure_dir(output_dir / "confusion_matrices")
    ensure_dir(output_dir / "topk_class_count_matrices")
    for split, payload in payloads.items():
        vectors = payload.embeddings @ bank.embeddings.T
        top_indices, counts, mean_scores, mean_scores_for_auc = compute_topk_tables(vectors, bank.labels, top_k=top_k)
        predictions = {
            "majority_vote": (predict_majority_vote(counts, mean_scores), counts),
            "class_average": (predict_class_average(mean_scores), mean_scores_for_auc),
        }
        save_topk_count_matrix(output_dir, source_name, split, payload.labels, counts)
        for rule, (preds, scores) in predictions.items():
            metric_rows.append(metric_payload(source_name, split, rule, top_k, payload.labels, preds, scores))
            per_class_rows.extend(per_class_metric_rows(source_name, split, rule, payload.labels, preds, counts))
            save_confusion_matrix(output_dir, source_name, split, rule, payload.labels, preds)
    return metric_rows, per_class_rows


def metric_payload(
    source_name: str,
    split: str,
    rule: str,
    top_k: int,
    labels: np.ndarray,
    preds: np.ndarray,
    scores: np.ndarray,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "source": source_name,
        "split": split,
        "rule": rule,
        "top_k": top_k,
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
    }
    try:
        probs = scores / np.clip(scores.sum(axis=1, keepdims=True), EPS, None) if rule == "majority_vote" else softmax_scores(scores)
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


def per_class_metric_rows(
    source_name: str,
    split: str,
    rule: str,
    labels: np.ndarray,
    preds: np.ndarray,
    counts: np.ndarray,
) -> list[dict[str, Any]]:
    precision, recall, f1, support = precision_recall_fscore_support(labels, preds, labels=list(range(len(CLASSES))), zero_division=0)
    rows: list[dict[str, Any]] = []
    for class_idx, class_name in enumerate(CLASSES):
        rows.append(
            {
                "source": source_name,
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


def save_confusion_matrix(output_dir: Path, source_name: str, split: str, rule: str, labels: np.ndarray, preds: np.ndarray) -> None:
    matrix = confusion_matrix(labels, preds, labels=list(range(len(CLASSES))))
    path_base = output_dir / "confusion_matrices" / f"confusion_{source_name}_{split}_{rule}"
    pd.DataFrame(matrix, index=CLASSES, columns=CLASSES).to_csv(path_base.with_suffix(".csv"))

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(CLASSES)), labels=CLASSES)
    ax.set_yticks(range(len(CLASSES)), labels=CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{source_name} {split} {rule}")
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            ax.text(col_idx, row_idx, str(int(matrix[row_idx, col_idx])), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path_base.with_suffix(".png"), dpi=180)
    plt.close(fig)


def save_topk_count_matrix(output_dir: Path, source_name: str, split: str, labels: np.ndarray, counts: np.ndarray) -> None:
    rows: list[dict[str, Any]] = []
    for true_idx, true_class in enumerate(CLASSES):
        mask = labels == true_idx
        mean_counts = counts[mask].mean(axis=0) if bool(mask.any()) else np.zeros(len(CLASSES), dtype=np.float32)
        row = {"source": source_name, "split": split, "true_image_class": true_class}
        for class_idx, class_name in enumerate(CLASSES):
            row[f"mean_topk_count_{class_name}_concepts"] = float(mean_counts[class_idx])
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / "topk_class_count_matrices" / f"topk_counts_{source_name}_{split}.csv", index=False)


def select_best_strategy(results: pd.DataFrame) -> str:
    candidates = results[(results["split"] == "val") & (results["rule"] == "majority_vote")].copy()
    candidates = candidates[~candidates["source"].isin(["raw_original_top10"])]
    if candidates.empty:
        raise RuntimeError("No validation candidate rows available for strategy selection.")
    candidates = candidates.sort_values(["macro_f1", "accuracy", "LGC_f1"], ascending=False)
    return str(candidates.iloc[0]["source"])


def write_report(
    output_dir: Path,
    args: argparse.Namespace,
    results: pd.DataFrame,
    per_class: pd.DataFrame,
    best_strategy: str,
) -> None:
    metric_cols = {"accuracy", "macro_f1", "macro_auroc", "HGC_f1", "LGC_f1", "NTL_f1", "NST_f1"}
    val_rows = results[(results["split"] == "val") & (results["rule"] == "majority_vote")].copy()
    test_rows = results[(results["split"] == "test")].copy()
    final_rows = test_rows[
        (test_rows["source"].isin(["raw_original_top10", best_strategy]))
        | ((test_rows["source"] == best_strategy) & test_rows["rule"].isin(["majority_vote", "class_average"]))
    ][["source", "rule", "accuracy", "macro_f1", "macro_auroc", "HGC_f1", "LGC_f1", "NTL_f1", "NST_f1"]]
    best_counts_path = output_dir / "topk_class_count_matrices" / f"topk_counts_{best_strategy}_test.csv"
    raw_counts_path = output_dir / "topk_class_count_matrices" / "topk_counts_raw_original_top10_test.csv"

    lines = [
        "# Confusion-Aware Concept Bank Repair",
        "",
        "## Setup",
        "",
        f"- Candidate bank: `{args.candidate_bank_dir}`.",
        f"- Refined stage: `{args.refined_stage_dir}`.",
        "- Candidate embeddings: refined filtered_top300 text embeddings.",
        "- Image embeddings: refined BioMedCLIP image embeddings.",
        "- Output format remains fixed: 4 classes x 10 concepts = 40 concepts.",
        "- Strategy selection uses validation top10 majority-vote Macro-F1.",
        "",
        "## Validation Strategy Selection",
        "",
        dataframe_to_markdown(
            val_rows[["source", "accuracy", "macro_f1", "macro_auroc", "HGC_f1", "LGC_f1", "NTL_f1", "NST_f1"]],
            metric_cols,
        ),
        "",
        f"Selected strategy: `{best_strategy}`.",
        "",
        "## Test Results",
        "",
        dataframe_to_markdown(final_rows, metric_cols),
        "",
        "## Top10 Concept Class Counts",
        "",
        "Raw original_top10 test counts:",
        "",
        dataframe_to_markdown(pd.read_csv(raw_counts_path), set()),
        "",
        f"Selected `{best_strategy}` test counts:",
        "",
        dataframe_to_markdown(pd.read_csv(best_counts_path), set()),
        "",
        "## Output Files",
        "",
        f"- Results: `{output_dir / 'confusion_aware_bank_repair_results.csv'}`",
        f"- Per-class metrics: `{output_dir / 'confusion_aware_bank_repair_per_class.csv'}`",
        f"- Selected banks: `{output_dir / 'banks'}`",
        f"- Confusion matrices: `{output_dir / 'confusion_matrices'}`",
        f"- Top10 count matrices: `{output_dir / 'topk_class_count_matrices'}`",
        "",
        "## Interpretation",
        "",
        "- The repair is successful if the selected bank improves validation Macro-F1 and also improves frozen-test top10 Macro-F1 over raw original_top10.",
        "- LGC should no longer collapse to zero F1.",
        "- Remaining NTL weakness should be interpreted separately from HGC/LGC confusion.",
    ]
    (output_dir / "confusion_aware_bank_repair_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    strategies = parse_csv_list(args.strategies)
    ensure_dir(args.output_dir)
    write_json(
        args.output_dir / "experiment_config.json",
        {
            "refined_stage_dir": str(args.refined_stage_dir),
            "embeddings_dir": str(args.embeddings_dir),
            "candidate_bank_dir": str(args.candidate_bank_dir),
            "output_dir": str(args.output_dir),
            "strategies": strategies,
            "top_k": args.top_k,
            "concepts_per_class": args.concepts_per_class,
            "classes": CLASSES,
        },
    )

    payloads = load_refined_split_payloads(args.refined_stage_dir, args.embeddings_dir)
    candidate_bank = load_candidate_bank(args.refined_stage_dir, args.candidate_bank_dir)
    stats = compute_train_stats(payloads["train"], candidate_bank)
    raw_bank = load_concept_bank(args.refined_stage_dir, "refined_vectors_original_top10")

    metric_rows: list[dict[str, Any]] = []
    per_class_rows_all: list[dict[str, Any]] = []
    raw_rows, raw_class_rows = evaluate_bank(args.output_dir, "raw_original_top10", raw_bank, payloads, top_k=args.top_k)
    metric_rows.extend(raw_rows)
    per_class_rows_all.extend(raw_class_rows)

    strategy_summary_rows: list[dict[str, Any]] = []
    for strategy in strategies:
        scores = score_candidates(strategy, candidate_bank, stats)
        selected_indices = select_bank(strategy, candidate_bank, scores, concepts_per_class=args.concepts_per_class)
        repaired_bank = save_selected_bank(args.output_dir, strategy, candidate_bank, selected_indices, scores, stats)
        rows, class_rows = evaluate_bank(args.output_dir, strategy, repaired_bank, payloads, top_k=args.top_k)
        metric_rows.extend(rows)
        per_class_rows_all.extend(class_rows)
        strategy_summary_rows.append(
            {
                "strategy": strategy,
                "selected_indices": "|".join(str(int(item)) for item in selected_indices),
                "selected_concepts": "|".join(str(item) for item in repaired_bank.concepts.tolist()),
            }
        )

    results = pd.DataFrame(metric_rows)
    per_class = pd.DataFrame(per_class_rows_all)
    strategy_summary = pd.DataFrame(strategy_summary_rows)
    best_strategy = select_best_strategy(results)
    results.to_csv(args.output_dir / "confusion_aware_bank_repair_results.csv", index=False)
    per_class.to_csv(args.output_dir / "confusion_aware_bank_repair_per_class.csv", index=False)
    strategy_summary.to_csv(args.output_dir / "strategy_selected_concepts.csv", index=False)
    write_json(args.output_dir / "selected_strategy.json", {"selected_strategy": best_strategy})
    write_report(args.output_dir, args, results, per_class, best_strategy)
    print(json.dumps({"output_dir": str(args.output_dir), "selected_strategy": best_strategy}, indent=2))


if __name__ == "__main__":
    main()
