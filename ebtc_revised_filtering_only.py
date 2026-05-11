#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

from ebtc_project_paths import EXPERIMENT_DIR, FILTERING_ONLY_OUTPUT_DIR, PROJECT_ROOT

matplotlib.use("Agg")
import matplotlib.pyplot as plt


if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))
NEWCODE_DIR = PROJECT_ROOT
if str(NEWCODE_DIR) not in sys.path:
    sys.path.insert(0, str(NEWCODE_DIR))

from ebtc_cycl_retrieval_stage_revised import (  # noqa: E402
    CLASS_TO_INDEX,
    LABEL_CODES,
    ConceptCandidate,
    RetrievalBank,
    DEFAULT_BACKUP_EMBEDDINGS_DIR,
    DEFAULT_EMBEDDINGS_DIR,
    DEFAULT_FILTERED_TOP300_DIR,
    ensure_dir,
    ensure_official_split_embedding_cache,
    l2_normalize,
    load_cached_split_payloads,
    load_filtered_top300,
    merge_selected_rows,
    save_retrieval_bank,
    write_csv,
    write_json,
)


DEFAULT_OUTPUT_DIR = FILTERING_ONLY_OUTPUT_DIR

DEFAULT_CLASS_WEIGHTS: dict[str, dict[str, float]] = {
    "HGC": {"LGC": 0.5, "NTL": 0.3, "NST": 0.2},
    "LGC": {"HGC": 0.4, "NTL": 0.4, "NST": 0.2},
    "NTL": {"LGC": 0.45, "HGC": 0.35, "NST": 0.2},
    "NST": {"HGC": 0.3, "LGC": 0.3, "NTL": 0.4},
}

BUDGETS: dict[str, dict[str, int]] = {
    "sym10": {"HGC": 10, "LGC": 10, "NTL": 10, "NST": 10},
    "asym_small": {"HGC": 10, "LGC": 20, "NTL": 20, "NST": 10},
    "asym_ntl_heavy": {"HGC": 10, "LGC": 20, "NTL": 30, "NST": 10},
    "sym20": {"HGC": 20, "LGC": 20, "NTL": 20, "NST": 20},
}

METHODS = ("mean_cosine", "disc_mean_diff", "max_negative_margin", "class_weighted_negative")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filtering-only EBTC concept bank ranking and 4x4 heatmap repair.")
    parser.add_argument("--filtered-top300-dir", type=Path, default=DEFAULT_FILTERED_TOP300_DIR)
    parser.add_argument("--embeddings-dir", type=Path, default=DEFAULT_EMBEDDINGS_DIR)
    parser.add_argument("--backup-embeddings-dir", type=Path, default=DEFAULT_BACKUP_EMBEDDINGS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--weights-json", type=Path, default=None)
    return parser.parse_args()


def load_weight_config(path: Path | None) -> dict[str, dict[str, float]]:
    if path is None:
        return DEFAULT_CLASS_WEIGHTS
    payload = json.loads(path.read_text(encoding="utf-8"))
    weights: dict[str, dict[str, float]] = {}
    for class_name in LABEL_CODES:
        class_weights = {other: float(payload[class_name][other]) for other in LABEL_CODES if other != class_name}
        total = sum(class_weights.values())
        if total <= 0:
            raise ValueError(f"Invalid class weights for {class_name}: {class_weights}")
        weights[class_name] = {other: value / total for other, value in class_weights.items()}
    return weights


def compute_classwise_mean_similarities(
    candidates_by_class: dict[str, list[ConceptCandidate]],
    split_payloads: dict[str, dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    train_rows = split_payloads["train"]["rows"]
    train_embeddings = split_payloads["train"]["embeddings"]
    train_labels = np.array([CLASS_TO_INDEX[str(row["class_name"])] for row in train_rows], dtype=np.int64)

    rows_by_class: dict[str, list[dict[str, Any]]] = {}
    all_rows: list[dict[str, Any]] = []
    for concept_class in LABEL_CODES:
        class_rows: list[dict[str, Any]] = []
        for candidate in candidates_by_class[concept_class]:
            means: dict[str, float] = {}
            for image_class in LABEL_CODES:
                image_mask = train_labels == CLASS_TO_INDEX[image_class]
                means[image_class] = float((train_embeddings[image_mask] @ candidate.embedding).mean())

            row: dict[str, Any] = {
                "concept_id": candidate.concept_id,
                "concept_text": candidate.concept,
                "concept_class": concept_class,
                "source_rank_in_filtered_top300": candidate.source_rank_in_filtered_top300,
                "embedding_index": candidate.embedding_index,
                "embedding": candidate.embedding,
            }
            for image_class in LABEL_CODES:
                row[f"mu_{image_class}"] = means[image_class]
            class_rows.append(row)
        rows_by_class[concept_class] = class_rows
        all_rows.extend({key: value for key, value in row.items() if key != "embedding"} for row in class_rows)
    return rows_by_class, all_rows


def compute_ranking_scores(
    classwise_rows: dict[str, list[dict[str, Any]]],
    method: str,
    class_weights: dict[str, dict[str, float]],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    if method not in METHODS:
        raise ValueError(f"Unsupported method: {method}")

    ranked_by_class: dict[str, list[dict[str, Any]]] = {}
    diagnostic_rows: list[dict[str, Any]] = []
    for target_class in LABEL_CODES:
        ranked_rows: list[dict[str, Any]] = []
        for base_row in classwise_rows[target_class]:
            row = dict(base_row)
            mu_same = float(row[f"mu_{target_class}"])
            negative_means = {
                class_name: float(row[f"mu_{class_name}"])
                for class_name in LABEL_CODES
                if class_name != target_class
            }
            hardest_negative_class = max(negative_means, key=negative_means.get)
            hardest_negative_value = float(negative_means[hardest_negative_class])
            mu_diff = float(np.mean(list(negative_means.values())))
            weighted_negative = float(
                sum(class_weights[target_class][class_name] * negative_means[class_name] for class_name in negative_means)
            )

            if method == "mean_cosine":
                ranking_score = mu_same
            elif method == "disc_mean_diff":
                ranking_score = mu_same - mu_diff
            elif method == "max_negative_margin":
                ranking_score = mu_same - hardest_negative_value
            elif method == "class_weighted_negative":
                ranking_score = mu_same - weighted_negative
            else:
                raise ValueError(f"Unsupported method: {method}")

            row.update(
                {
                    "target_class": target_class,
                    "class_name": target_class,
                    "concept": row["concept_text"],
                    "mu_same": mu_same,
                    "mu_diff": mu_diff,
                    "hardest_negative_class": hardest_negative_class,
                    "hardest_negative_value": hardest_negative_value,
                    "weighted_negative": weighted_negative,
                    "ranking_score": float(ranking_score),
                    "ranking_method": method,
                    "mu_diff_max": hardest_negative_value,
                    "margin_vs_rest": mu_same - mu_diff,
                    "margin_vs_max_other": mu_same - hardest_negative_value,
                    "auroc_ovr": float("nan"),
                }
            )
            ranked_rows.append(row)

        ranked_rows.sort(
            key=lambda item: (
                -float(item["ranking_score"]),
                -float(item["mu_same"]),
                int(item["source_rank_in_filtered_top300"]),
                str(item["concept_text"]),
            )
        )
        for rank, row in enumerate(ranked_rows, start=1):
            row["retrieval_rank_in_class"] = rank
            diagnostic_rows.append({key: value for key, value in row.items() if key != "embedding"})
        ranked_by_class[target_class] = ranked_rows
    return ranked_by_class, diagnostic_rows


def build_bank_from_budget(
    ranked_by_class: dict[str, list[dict[str, Any]]],
    bank_name: str,
    budget: dict[str, int],
) -> RetrievalBank:
    selected = {
        class_name: [dict(item) for item in ranked_by_class[class_name][: budget[class_name]]]
        for class_name in LABEL_CODES
    }
    return merge_selected_rows(bank_name, selected)


def compute_class_similarity_matrix(
    bank: RetrievalBank,
    split_payloads: dict[str, dict[str, Any]],
    split_name: str,
) -> np.ndarray:
    rows = split_payloads[split_name]["rows"]
    embeddings = split_payloads[split_name]["embeddings"]
    matrix = np.zeros((len(LABEL_CODES), len(LABEL_CODES)), dtype=np.float32)
    per_class_text: dict[str, np.ndarray] = {}
    for class_name in LABEL_CODES:
        per_class_text[class_name] = l2_normalize(
            np.stack([row["embedding"] for row in bank.per_class_rows[class_name]], axis=0).astype(np.float32)
        )

    for image_class in LABEL_CODES:
        image_mask = np.array([str(row["class_name"]) == image_class for row in rows], dtype=bool)
        image_embeddings = embeddings[image_mask]
        for concept_class in LABEL_CODES:
            matrix[CLASS_TO_INDEX[image_class], CLASS_TO_INDEX[concept_class]] = float(
                (image_embeddings @ per_class_text[concept_class].T).mean()
            )
    return matrix


def compute_row_and_column_margins(matrix: np.ndarray) -> dict[str, Any]:
    diag = np.diag(matrix)
    off_diag = matrix[~np.eye(matrix.shape[0], dtype=bool)]
    result: dict[str, Any] = {
        "mean_diagonal": float(diag.mean()),
        "mean_off_diagonal": float(off_diag.mean()),
        "diagonal_minus_off_diagonal": float(diag.mean() - off_diag.mean()),
    }
    for idx, class_name in enumerate(LABEL_CODES):
        row_others = np.delete(matrix[idx, :], idx)
        col_others = np.delete(matrix[:, idx], idx)
        result[f"row_margin_{class_name}"] = float(matrix[idx, idx] - row_others.max())
        result[f"row_argmax_{class_name}"] = LABEL_CODES[int(matrix[idx, :].argmax())]
        result[f"col_margin_{class_name}"] = float(matrix[idx, idx] - col_others.mean())
        result[f"col_argmax_{class_name}"] = LABEL_CODES[int(matrix[:, idx].argmax())]
    return result


def save_matrix_csv(matrix: np.ndarray, path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for row_idx, image_class in enumerate(LABEL_CODES):
        row: dict[str, Any] = {"image_class": image_class}
        for col_idx, concept_class in enumerate(LABEL_CODES):
            row[concept_class] = float(matrix[row_idx, col_idx])
        rows.append(row)
    write_csv(path, ["image_class", *LABEL_CODES], rows)


def plot_heatmap(matrix: np.ndarray, path: Path, title: str, vmin: float | None = None, vmax: float | None = None) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 4.9))
    image = ax.imshow(matrix, cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(LABEL_CODES)))
    ax.set_yticks(np.arange(len(LABEL_CODES)))
    ax.set_xticklabels(LABEL_CODES)
    ax.set_yticklabels(LABEL_CODES)
    ax.set_xlabel("Concept Class")
    ax.set_ylabel("Image Class")
    ax.set_title(title)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, f"{matrix[row, col]:.3f}", ha="center", va="center", color="white")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Mean cosine")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_overview_heatmaps(matrices: dict[tuple[str, str], np.ndarray], bank_order: list[str], output_path: Path) -> None:
    all_values = np.concatenate([matrix.reshape(-1) for matrix in matrices.values()])
    vmin = float(all_values.min())
    vmax = float(all_values.max())
    fig, axes = plt.subplots(nrows=len(bank_order), ncols=2, figsize=(11, max(4, len(bank_order) * 3.2)))
    if len(bank_order) == 1:
        axes = np.expand_dims(axes, axis=0)
    fig.suptitle("Revised class-wise image-text similarity", fontsize=18, y=0.995)
    last_image = None
    for row_idx, bank_name in enumerate(bank_order):
        for col_idx, split_name in enumerate(["train", "val"]):
            ax = axes[row_idx, col_idx]
            matrix = matrices[(bank_name, split_name)]
            last_image = ax.imshow(matrix, cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set_xticks(np.arange(len(LABEL_CODES)))
            ax.set_yticks(np.arange(len(LABEL_CODES)))
            ax.set_xticklabels(LABEL_CODES, fontsize=9)
            ax.set_yticklabels(LABEL_CODES, fontsize=9)
            if row_idx == 0:
                ax.set_title(split_name, fontsize=12)
            ax.set_ylabel(bank_name if col_idx == 0 else "Image class", fontsize=9)
            ax.set_xlabel("Concept class", fontsize=9)
            metrics = compute_row_and_column_margins(matrix)
            ax.text(
                0.5,
                1.02,
                f"margin={metrics['diagonal_minus_off_diagonal']:.3f} | LGC col={metrics['col_margin_LGC']:.3f} | NTL row={metrics['row_margin_NTL']:.3f}",
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=8,
            )
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", color="white", fontsize=8)
    fig.subplots_adjust(left=0.16, right=0.88, top=0.98, bottom=0.03, hspace=0.48, wspace=0.18)
    cbar_ax = fig.add_axes([0.905, 0.12, 0.018, 0.76])
    fig.colorbar(last_image, cax=cbar_ax, label="Mean cosine")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def write_bank_csv(bank: RetrievalBank, path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for class_name in LABEL_CODES:
        for row in bank.per_class_rows[class_name]:
            rows.append(
                {
                    "bank_name": bank.name,
                    "concept_class": class_name,
                    "concept_id": row["concept_id"],
                    "concept_text": row["concept_text"],
                    "retrieval_rank_in_class": int(row["retrieval_rank_in_class"]),
                    "ranking_method": row["ranking_method"],
                    "ranking_score": float(row["ranking_score"]),
                    "mu_HGC": float(row["mu_HGC"]),
                    "mu_LGC": float(row["mu_LGC"]),
                    "mu_NTL": float(row["mu_NTL"]),
                    "mu_NST": float(row["mu_NST"]),
                    "mu_same": float(row["mu_same"]),
                    "mu_diff": float(row["mu_diff"]),
                    "hardest_negative_class": row["hardest_negative_class"],
                    "hardest_negative_value": float(row["hardest_negative_value"]),
                    "weighted_negative": float(row["weighted_negative"]),
                }
            )
    write_csv(
        path,
        [
            "bank_name",
            "concept_class",
            "concept_id",
            "concept_text",
            "retrieval_rank_in_class",
            "ranking_method",
            "ranking_score",
            "mu_HGC",
            "mu_LGC",
            "mu_NTL",
            "mu_NST",
            "mu_same",
            "mu_diff",
            "hardest_negative_class",
            "hardest_negative_value",
            "weighted_negative",
        ],
        rows,
    )


def answer_report(
    output_dir: Path,
    summary_rows: list[dict[str, Any]],
    baseline_lookup: dict[tuple[str, str], dict[str, Any]],
    best_val_bank: dict[str, Any],
) -> None:
    def val_row(bank_name: str) -> dict[str, Any]:
        return next(row for row in summary_rows if row["bank_name"] == bank_name and row["split"] == "val")

    filtered_val = baseline_lookup[("filtered_top300", "val")]
    best_val = val_row(str(best_val_bank["bank_name"]))
    lines = [
        "# Revised Filtering-Only Report",
        "",
        "## Scope",
        "",
        "This run only revises concept ranking, per-class bank budgets, and 4x4 class-wise image-text similarity verification.",
        "CBM/CyCL models, losses, pair definitions, and training schedules were not modified or run.",
        "",
        "## Ranking Methods",
        "",
        "- `mean_cosine`: `score = mu_same`.",
        "- `disc_mean_diff`: `score = mu_same - mean(non-target class means)`.",
        "- `max_negative_margin`: `score = mu_same - max_{d != c} mu_d`.",
        "- `class_weighted_negative`: `score = mu_same - sum_d w(c,d) * mu_d` with configurable class-pair weights.",
        "",
        "## Budgets",
        "",
        "- `sym10`: HGC=10, LGC=10, NTL=10, NST=10.",
        "- `asym_small`: HGC=10, LGC=20, NTL=20, NST=10.",
        "- `asym_ntl_heavy`: HGC=10, LGC=20, NTL=30, NST=10.",
        "- `sym20`: HGC=20, LGC=20, NTL=20, NST=20.",
        "",
        "## Main Findings",
        "",
        f"- Baseline `filtered_top300` val margin: `{filtered_val['diagonal_minus_off_diagonal']:.4f}`.",
        f"- Best val bank by primary score: `{best_val_bank['bank_name']}`.",
        f"- Best val margin: `{best_val['diagonal_minus_off_diagonal']:.4f}`.",
        f"- Best val LGC column margin: `{best_val['col_margin_LGC']:.4f}`.",
        f"- Best val NTL row margin: `{best_val['row_margin_NTL']:.4f}`.",
        "",
        "## Answers",
        "",
    ]

    margin_improved = float(best_val["diagonal_minus_off_diagonal"]) > float(filtered_val["diagonal_minus_off_diagonal"])
    lgc_improved = float(best_val["col_margin_LGC"]) > float(filtered_val["col_margin_LGC"])
    ntl_improved = float(best_val["row_margin_NTL"]) > float(filtered_val["row_margin_NTL"])
    budget_name = str(best_val_bank["budget_name"])
    method_name = str(best_val_bank["ranking_method"])

    lines.extend(
        [
            f"1. New ranking improved diagonal structure: `{margin_improved}`.",
            f"2. LGC column selectivity improved versus filtered_top300: `{lgc_improved}`.",
            f"3. NTL row diagonal advantage improved versus filtered_top300: `{ntl_improved}`.",
            f"4. Best current budget: `{budget_name}`.",
            f"5. Best current ranking method: `{method_name}`.",
            f"6. Recommended next CyCL input bank: `{best_val_bank['bank_name']}`.",
            "",
            "## Important Caveat",
            "",
            "The selected best bank is based only on train/val 4x4 image-text structure. It is not yet validated by CBM/CyCL training.",
            "",
            "## Key Outputs",
            "",
            "- `outputs/revised_ranking_scores.csv`",
            "- `outputs/revised_bank_summary.csv`",
            "- `outputs/revised_class_similarity_summary.csv`",
            "- `outputs/revised_confusion_diagnostics.csv`",
            "- `outputs/revised_class_similarity_heatmaps.png`",
        ]
    )
    (output_dir / "outputs" / "revised_filtering_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    ensure_dir(args.output_dir / "outputs")
    ensure_dir(args.output_dir / "banks")
    ensure_official_split_embedding_cache(args.embeddings_dir, args.backup_embeddings_dir)
    class_weights = load_weight_config(args.weights_json)
    write_json(args.output_dir / "outputs" / "class_weighted_negative_weights.json", class_weights)

    split_payloads = load_cached_split_payloads(args.embeddings_dir)
    candidates = load_filtered_top300(args.filtered_top300_dir, args.embeddings_dir)
    classwise_rows, _ = compute_classwise_mean_similarities(candidates, split_payloads)

    ranking_rows: list[dict[str, Any]] = []
    ranked_by_method: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for method in METHODS:
        ranked, diagnostics = compute_ranking_scores(classwise_rows, method, class_weights)
        ranked_by_method[method] = ranked
        ranking_rows.extend(diagnostics)

    ranking_fieldnames = [
        "concept_id",
        "concept_text",
        "concept",
        "concept_class",
        "target_class",
        "class_name",
        "source_rank_in_filtered_top300",
        "embedding_index",
        "mu_HGC",
        "mu_LGC",
        "mu_NTL",
        "mu_NST",
        "mu_same",
        "mu_diff",
        "hardest_negative_class",
        "hardest_negative_value",
        "weighted_negative",
        "mu_diff_max",
        "margin_vs_rest",
        "margin_vs_max_other",
        "auroc_ovr",
        "ranking_score",
        "ranking_method",
        "retrieval_rank_in_class",
    ]
    write_csv(args.output_dir / "outputs" / "revised_ranking_scores.csv", ranking_fieldnames, ranking_rows)

    banks: dict[str, RetrievalBank] = {}
    bank_order: list[str] = []
    for method in ["max_negative_margin", "class_weighted_negative"]:
        method_prefix = "maxneg" if method == "max_negative_margin" else "weighted"
        for budget_name, budget in BUDGETS.items():
            bank_name = f"bank_{method_prefix}_{budget_name}"
            bank = build_bank_from_budget(ranked_by_method[method], bank_name, budget)
            banks[bank_name] = bank
            bank_order.append(bank_name)
            save_retrieval_bank(bank, args.output_dir / "banks" / bank_name)
            write_bank_csv(bank, args.output_dir / "outputs" / f"{bank_name}.csv")

    # Keep baselines in the diagnostics but do not re-export them as revised banks.
    baseline_banks = {
        "filtered_top300": merge_selected_rows(
            "filtered_top300",
            {class_name: ranked_by_method["mean_cosine"][class_name] for class_name in LABEL_CODES},
        ),
        "old_disc_sym10": build_bank_from_budget(
            ranked_by_method["disc_mean_diff"],
            "old_disc_sym10",
            BUDGETS["sym10"],
        ),
    }

    summary_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    matrices_for_overview: dict[tuple[str, str], np.ndarray] = {}
    all_eval_banks = {**baseline_banks, **banks}
    for bank_name, bank in all_eval_banks.items():
        for split_name in ["train", "val"]:
            matrix = compute_class_similarity_matrix(bank, split_payloads, split_name)
            if bank_name in banks:
                matrices_for_overview[(bank_name, split_name)] = matrix
            csv_path = args.output_dir / "outputs" / f"revised_class_similarity_{bank_name}_{split_name}.csv"
            png_path = args.output_dir / "outputs" / f"revised_class_similarity_{bank_name}_{split_name}.png"
            save_matrix_csv(matrix, csv_path)
            plot_heatmap(matrix, png_path, f"{bank_name} image-text similarity ({split_name})")
            metrics = compute_row_and_column_margins(matrix)
            summary_rows.append(
                {
                    "bank_name": bank_name,
                    "split": split_name,
                    "matrix_csv": str(csv_path),
                    "matrix_png": str(png_path),
                    **metrics,
                }
            )
            confusion_rows.append(
                {
                    "bank_name": bank_name,
                    "split": split_name,
                    "LGC_column_margin": metrics["col_margin_LGC"],
                    "LGC_column_argmax": metrics["col_argmax_LGC"],
                    "LGC_row_margin": metrics["row_margin_LGC"],
                    "LGC_row_argmax": metrics["row_argmax_LGC"],
                    "NTL_column_margin": metrics["col_margin_NTL"],
                    "NTL_column_argmax": metrics["col_argmax_NTL"],
                    "NTL_row_margin": metrics["row_margin_NTL"],
                    "NTL_row_argmax": metrics["row_argmax_NTL"],
                    "global_margin": metrics["diagonal_minus_off_diagonal"],
                }
            )

    summary_fields = [
        "bank_name",
        "split",
        "matrix_csv",
        "matrix_png",
        "mean_diagonal",
        "mean_off_diagonal",
        "diagonal_minus_off_diagonal",
        *[f"row_margin_{class_name}" for class_name in LABEL_CODES],
        *[f"row_argmax_{class_name}" for class_name in LABEL_CODES],
        *[f"col_margin_{class_name}" for class_name in LABEL_CODES],
        *[f"col_argmax_{class_name}" for class_name in LABEL_CODES],
    ]
    write_csv(args.output_dir / "outputs" / "revised_class_similarity_summary.csv", summary_fields, summary_rows)
    write_csv(
        args.output_dir / "outputs" / "revised_confusion_diagnostics.csv",
        [
            "bank_name",
            "split",
            "LGC_column_margin",
            "LGC_column_argmax",
            "LGC_row_margin",
            "LGC_row_argmax",
            "NTL_column_margin",
            "NTL_column_argmax",
            "NTL_row_margin",
            "NTL_row_argmax",
            "global_margin",
        ],
        confusion_rows,
    )

    bank_summary_rows: list[dict[str, Any]] = []
    for bank_name, bank in banks.items():
        method = "max_negative_margin" if bank_name.startswith("bank_maxneg") else "class_weighted_negative"
        budget_name = bank_name.replace("bank_maxneg_", "").replace("bank_weighted_", "")
        val_metrics = next(row for row in summary_rows if row["bank_name"] == bank_name and row["split"] == "val")
        bank_summary_rows.append(
            {
                "bank_name": bank_name,
                "ranking_method": method,
                "budget_name": budget_name,
                "total_before_dedup": bank.total_before_dedup,
                "total_after_dedup": bank.total_after_dedup,
                "dedup_removed": bank.dedup_removed,
                **{f"{class_name}_budget": BUDGETS[budget_name][class_name] for class_name in LABEL_CODES},
                "val_global_margin": val_metrics["diagonal_minus_off_diagonal"],
                "val_LGC_col_margin": val_metrics["col_margin_LGC"],
                "val_NTL_row_margin": val_metrics["row_margin_NTL"],
                "selection_score": float(val_metrics["diagonal_minus_off_diagonal"])
                + 0.5 * float(val_metrics["col_margin_LGC"])
                + 0.5 * float(val_metrics["row_margin_NTL"]),
            }
        )
    write_csv(
        args.output_dir / "outputs" / "revised_bank_summary.csv",
        [
            "bank_name",
            "ranking_method",
            "budget_name",
            "total_before_dedup",
            "total_after_dedup",
            "dedup_removed",
            "HGC_budget",
            "LGC_budget",
            "NTL_budget",
            "NST_budget",
            "val_global_margin",
            "val_LGC_col_margin",
            "val_NTL_row_margin",
            "selection_score",
        ],
        bank_summary_rows,
    )

    plot_overview_heatmaps(matrices_for_overview, bank_order, args.output_dir / "outputs" / "revised_class_similarity_heatmaps.png")
    best_val_bank = max(bank_summary_rows, key=lambda row: float(row["selection_score"]))
    baseline_lookup = {(row["bank_name"], row["split"]): row for row in summary_rows}
    answer_report(args.output_dir, summary_rows, baseline_lookup, best_val_bank)

    write_json(
        args.output_dir / "outputs" / "stage_config.json",
        {
            "filtered_top300_dir": str(args.filtered_top300_dir),
            "embeddings_dir": str(args.embeddings_dir),
            "output_dir": str(args.output_dir),
            "methods": list(METHODS),
            "budgets": BUDGETS,
            "class_weighted_negative_weights": class_weights,
            "best_bank_by_selection_score": best_val_bank,
        },
    )
    print(
        f"[done] output_dir={args.output_dir} banks={len(banks)} best={best_val_bank['bank_name']} "
        f"score={best_val_bank['selection_score']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
