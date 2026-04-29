#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


EXPERIMENT_DIR = Path("/home/kunet.ae/100069491/experiment")
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))
NEWCODE_DIR = Path("/home/kunet.ae/100069491/newcode")
if str(NEWCODE_DIR) not in sys.path:
    sys.path.insert(0, str(NEWCODE_DIR))

from ebtc_cycl_retrieval_stage_revised import (  # noqa: E402
    DEFAULT_BACKUP_EMBEDDINGS_DIR,
    DEFAULT_EMBEDDINGS_DIR,
    LABEL_CODES,
    RetrievalBank,
    ensure_dir,
    ensure_official_split_embedding_cache,
    l2_normalize,
    load_biomedclip,
    load_cached_split_payloads,
    load_json,
    read_csv_rows,
    run_cbm_seed,
    seed_dir_name,
    write_csv,
    write_json,
)


DEFAULT_OUTPUT_DIR = Path(
    "/home/kunet.ae/100069491/newcode/ebtc_revised_bank_formal_compare_outputs"
)

BANK_PATH_CANDIDATES: dict[str, list[Path]] = {
    "filtered_top300": [
        Path("/home/kunet.ae/100069491/newcode/ebtc_cycl_retrieval_stage_revised_outputs/banks/filtered_top300"),
        Path("/home/kunet.ae/100069491/newcode/ebtc_cycl_retrieval_stage_multiseed_outputs/banks/filtered_top300"),
    ],
    "retrieval_top10_per_class": [
        Path("/home/kunet.ae/100069491/newcode/ebtc_cycl_retrieval_stage_revised_outputs/banks/retrieval_top10_per_class"),
        Path("/home/kunet.ae/100069491/newcode/ebtc_cycl_retrieval_stage_multiseed_outputs/banks/retrieval_top10_per_class"),
        Path("/home/kunet.ae/100069491/newcode/ebtc_cycl_retrieval_stage_outputs/banks/retrieval_top10_per_class"),
    ],
    "old_disc_sym10": [
        Path("/home/kunet.ae/100069491/newcode/ebtc_cycl_retrieval_stage_revised_outputs/banks/retrieval_top10_per_class_disc"),
        Path("/home/kunet.ae/100069491/newcode/ebtc_cycl_retrieval_stage_revised_shortdebug_outputs/banks/retrieval_top10_per_class_disc"),
    ],
    "weighted_sym10": [
        Path("/home/kunet.ae/100069491/newcode/ebtc_revised_filtering_only_outputs/banks/bank_weighted_sym10"),
    ],
}


@dataclass
class FixedBank:
    alias: str
    path: Path
    bank: RetrievalBank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Formal revised CBM/CyCL comparison across fixed EBTC banks.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--embeddings-dir", type=Path, default=DEFAULT_EMBEDDINGS_DIR)
    parser.add_argument("--backup-embeddings-dir", type=Path, default=DEFAULT_BACKUP_EMBEDDINGS_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--run-banks", default="filtered_top300,old_disc_sym10,weighted_sym10")
    parser.add_argument("--run-models", default="cbm_only,cycl")
    parser.add_argument("--schedule", default="formal_compare")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--baseline-epochs", type=int, default=0)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--joint-epochs", type=int, default=15)
    parser.add_argument("--baseline-lr", type=float, default=1e-3)
    parser.add_argument("--main-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-concept", type=float, default=1.0)
    parser.add_argument("--lambda-align", type=float, default=0.3)
    parser.add_argument("--lambda-cycl", type=float, default=0.1)
    parser.add_argument("--lambda-img-txt", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=0.07)
    parser.add_argument("--tau-img-txt", type=float, default=0.07)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--proj-dim", type=int, default=128)
    parser.add_argument("--num-views", type=int, default=2)
    parser.add_argument("--ranking-method", default="fixed_bank_formal_compare")
    parser.add_argument("--use-color-aug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-geom-aug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-image-text-pairing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_seed_list(value: str) -> list[int]:
    seeds = [int(item) for item in parse_csv_list(value)]
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def resolve_bank_path(alias: str) -> Path:
    for path in BANK_PATH_CANDIDATES[alias]:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not locate fixed bank alias={alias}; tried {BANK_PATH_CANDIDATES[alias]}")


def load_exported_bank(alias: str, bank_dir: Path) -> RetrievalBank:
    payload = np.load(bank_dir / "filtered_concept_text_embeddings.npz", allow_pickle=True)
    concepts = [str(item) for item in payload["concepts"].tolist()]
    embeddings = l2_normalize(payload["concept_embeddings"].astype(np.float32))
    concept_to_indices: dict[str, list[int]] = {}
    for index, concept in enumerate(concepts):
        concept_to_indices.setdefault(concept, []).append(index)

    per_class_rows: dict[str, list[dict[str, Any]]] = {}
    for class_name in LABEL_CODES:
        rows: list[dict[str, Any]] = []
        for row in read_csv_rows(bank_dir / f"{class_name}.csv"):
            concept = str(row["concept"])
            embedding_index = concept_to_indices[concept][0]
            rows.append(
                {
                    **row,
                    "bank_name": alias,
                    "class_name": class_name,
                    "concept": concept,
                    "concept_id": str(row["concept_id"]),
                    "source_rank_in_filtered_top300": int(row.get("source_rank_in_filtered_top300") or 0),
                    "retrieval_rank_in_class": int(row.get("retrieval_rank_in_class") or 0),
                    "ranking_score": float(row.get("ranking_score") or 0.0),
                    "ranking_method": str(row.get("ranking_method") or "unknown"),
                    "embedding": embeddings[embedding_index].astype(np.float32),
                }
            )
        per_class_rows[class_name] = rows

    merged_rows: list[dict[str, Any]] = []
    for index, concept in enumerate(concepts):
        source_classes: list[str] = []
        source_concept_ids: list[str] = []
        source_retrieval_ranks: list[int] = []
        source_filtered_ranks: list[int] = []
        primary_class = None
        primary_concept_id = None
        primary_source_rank = None
        primary_retrieval_rank = None
        primary_ranking_score = None
        for class_name in LABEL_CODES:
            for row in per_class_rows[class_name]:
                if str(row["concept"]) == concept:
                    source_classes.append(class_name)
                    source_concept_ids.append(str(row["concept_id"]))
                    source_retrieval_ranks.append(int(row["retrieval_rank_in_class"]))
                    source_filtered_ranks.append(int(row["source_rank_in_filtered_top300"]))
                    if primary_class is None:
                        primary_class = class_name
                        primary_concept_id = str(row["concept_id"])
                        primary_source_rank = int(row["source_rank_in_filtered_top300"])
                        primary_retrieval_rank = int(row["retrieval_rank_in_class"])
                        primary_ranking_score = float(row["ranking_score"])
        if primary_class is None:
            raise RuntimeError(f"Merged concept not found in per-class rows: {concept}")
        merged_rows.append(
            {
                "bank_name": alias,
                "merged_index": index,
                "concept": concept,
                "primary_class": primary_class,
                "primary_concept_id": primary_concept_id,
                "primary_source_rank_in_filtered_top300": primary_source_rank,
                "primary_retrieval_rank_in_class": primary_retrieval_rank,
                "primary_ranking_score": primary_ranking_score,
                "source_classes": source_classes,
                "source_concept_ids": source_concept_ids,
                "source_retrieval_ranks": source_retrieval_ranks,
                "source_filtered_top300_ranks": source_filtered_ranks,
                "source_class_count": len(set(source_classes)),
                "embedding": embeddings[index].astype(np.float32),
            }
        )

    before_count = sum(len(rows) for rows in per_class_rows.values())
    return RetrievalBank(
        name=alias,
        per_class_rows=per_class_rows,
        merged_rows=merged_rows,
        merged_concepts=concepts,
        merged_embeddings=embeddings,
        total_before_dedup=before_count,
        total_after_dedup=len(concepts),
        dedup_removed=before_count - len(concepts),
    )


def resolve_fixed_banks(requested_aliases: list[str]) -> list[FixedBank]:
    fixed: list[FixedBank] = []
    for alias in requested_aliases:
        if alias not in BANK_PATH_CANDIDATES:
            raise ValueError(f"Unsupported fixed bank alias: {alias}")
        path = resolve_bank_path(alias)
        fixed.append(FixedBank(alias=alias, path=path, bank=load_exported_bank(alias, path)))
    return fixed


def bank_manifest_rows(fixed_banks: list[FixedBank]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in fixed_banks:
        rows.append(
            {
                "bank": item.alias,
                "path": str(item.path),
                "total_before_dedup": item.bank.total_before_dedup,
                "total_after_dedup": item.bank.total_after_dedup,
                "dedup_removed": item.bank.dedup_removed,
                **{f"{class_name}_concept_count": len(item.bank.per_class_rows[class_name]) for class_name in LABEL_CODES},
            }
        )
    return rows


def row_from_existing(
    bank: RetrievalBank,
    model_type: str,
    seed: int,
    experiment_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    payload = load_json(experiment_dir / "main_metrics.json")
    return {
        "bank": bank.name,
        "model_type": model_type,
        "bank_name": bank.name,
        "seed": seed,
        "setting": f"{model_type}_{bank.name}",
        "concept_count_before_dedup": bank.total_before_dedup,
        "concept_count_after_dedup": bank.total_after_dedup,
        "num_views": int(args.num_views),
        "schedule": str(args.schedule),
        "val_accuracy": float(payload["val"]["accuracy"]),
        "val_macro_f1": float(payload["val"]["macro_f1"]),
        "val_macro_auroc": float(payload["val"]["macro_auroc"]) if payload["val"]["macro_auroc"] is not None else float("nan"),
        "test_accuracy": float(payload["test"]["accuracy"]),
        "test_macro_f1": float(payload["test"]["macro_f1"]),
        "test_macro_auroc": float(payload["test"]["macro_auroc"]) if payload["test"]["macro_auroc"] is not None else float("nan"),
        "pair_positive_pair_count": float(payload.get("pair_diagnostics", {}).get("positive_pair_count", float("nan"))),
        "pair_negative_pair_count": float(payload.get("pair_diagnostics", {}).get("negative_pair_count", float("nan"))),
        "pair_positive_weight_mean": float(payload.get("pair_diagnostics", {}).get("positive_weight_mean", float("nan"))),
        "pair_negative_weight_mean": float(payload.get("pair_diagnostics", {}).get("negative_weight_mean", float("nan"))),
        "image_text_positive_similarity_mean": float(payload.get("image_text_diagnostics", {}).get("image_text_positive_similarity_mean", float("nan"))),
        "image_text_negative_similarity_mean": float(payload.get("image_text_diagnostics", {}).get("image_text_negative_similarity_mean", float("nan"))),
        "image_text_similarity_gap": float(payload.get("image_text_diagnostics", {}).get("image_text_similarity_gap", float("nan"))),
        "ranking_method": str(args.ranking_method),
        "output_dir": str(experiment_dir),
    }


def read_existing_seed_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return read_csv_rows(path)


def merge_seed_rows(existing_rows: list[dict[str, Any]], new_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in existing_rows + new_rows:
        key = (str(row["bank"]), str(row["model_type"]), int(row["seed"]))
        normalized = dict(row)
        normalized["seed"] = int(normalized["seed"])
        merged[key] = normalized
    rows = list(merged.values())
    rows.sort(key=lambda row: (str(row["bank"]), str(row["model_type"]), int(row["seed"])))
    return rows


def run_formal_compare(args: argparse.Namespace, fixed_banks: list[FixedBank]) -> list[dict[str, Any]]:
    ensure_official_split_embedding_cache(args.embeddings_dir, args.backup_embeddings_dir)
    split_payloads = load_cached_split_payloads(args.embeddings_dir)
    image_encoder, _, preprocess_val, resolved_device = load_biomedclip(args.device)
    for parameter in image_encoder.parameters():
        parameter.requires_grad = False
    image_encoder.eval()

    rows: list[dict[str, Any]] = []
    seeds = parse_seed_list(args.seeds)
    requested_models = parse_csv_list(args.run_models)
    experiment_root = args.output_dir / "experiments"
    ensure_dir(experiment_root)

    for fixed in fixed_banks:
        for model_type, use_cycl in [("cbm_only", False), ("cycl", True)]:
            if model_type not in requested_models:
                continue
            for seed in seeds:
                experiment_dir = experiment_root / fixed.alias / model_type / f"views_{args.num_views}_{args.schedule}" / seed_dir_name(seed)
                if args.skip_existing and (experiment_dir / "main_metrics.json").exists():
                    rows.append(row_from_existing(fixed.bank, model_type, seed, experiment_dir, args))
                    continue
                run_row = run_cbm_seed(
                    bank=fixed.bank,
                    split_payloads=split_payloads,
                    image_encoder=image_encoder,
                    preprocess_val=preprocess_val,
                    resolved_device=resolved_device,
                    args=args,
                    experiment_dir=experiment_dir,
                    seed=seed,
                    model_type=model_type,
                    use_cycl=use_cycl,
                )
                run_row["bank"] = fixed.alias
                rows.append(run_row)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    return rows


def summarize_seed_results(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = ["val_accuracy", "val_macro_f1", "val_macro_auroc", "test_accuracy", "test_macro_f1", "test_macro_auroc"]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in seed_rows:
        grouped.setdefault((str(row["bank"]), str(row["model_type"])), []).append(row)
    summary_rows: list[dict[str, Any]] = []
    for (bank, model_type), rows in grouped.items():
        item: dict[str, Any] = {
            "bank": bank,
            "model_type": model_type,
            "setting": f"{model_type}_{bank}",
            "n_seeds": len(rows),
            "seed_list": ",".join(str(row["seed"]) for row in sorted(rows, key=lambda x: int(x["seed"]))),
            "concept_count_after_dedup": rows[0]["concept_count_after_dedup"],
        }
        for metric in metrics:
            values = np.array([float(row[metric]) for row in rows], dtype=np.float64)
            item[f"{metric}_mean"] = float(values.mean())
            item[f"{metric}_std"] = float(values.std(ddof=0))
        summary_rows.append(item)
    summary_rows.sort(key=lambda row: (str(row["bank"]), str(row["model_type"])))
    return summary_rows


def per_class_from_confusion(matrix: np.ndarray, split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, class_name in enumerate(LABEL_CODES):
        tp = float(matrix[idx, idx])
        fp = float(matrix[:, idx].sum() - tp)
        fn = float(matrix[idx, :].sum() - tp)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        rows.append(
            {
                "split": split,
                "class_name": class_name,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": int(matrix[idx, :].sum()),
            }
        )
    return rows


def collect_per_class_seed_rows(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    per_rows: list[dict[str, Any]] = []
    for row in seed_rows:
        payload = load_json(Path(str(row["output_dir"])) / "main_metrics.json")
        for split in ["val", "test"]:
            matrix = np.array(payload[split]["confusion_matrix"], dtype=np.int64)
            for class_row in per_class_from_confusion(matrix, split):
                per_rows.append(
                    {
                        "bank": row["bank"],
                        "model_type": row["model_type"],
                        "setting": f"{row['model_type']}_{row['bank']}",
                        "seed": row["seed"],
                        **class_row,
                    }
                )
    return per_rows


def summarize_per_class_metrics(per_seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in per_seed_rows:
        grouped.setdefault((str(row["bank"]), str(row["model_type"]), str(row["split"]), str(row["class_name"])), []).append(row)
    out: list[dict[str, Any]] = []
    for (bank, model_type, split, class_name), rows in grouped.items():
        item: dict[str, Any] = {
            "bank": bank,
            "model_type": model_type,
            "setting": f"{model_type}_{bank}",
            "split": split,
            "class_name": class_name,
            "n_seeds": len(rows),
            "support_mean": float(np.mean([float(row["support"]) for row in rows])),
        }
        for metric in ["precision", "recall", "f1"]:
            values = np.array([float(row[metric]) for row in rows], dtype=np.float64)
            item[f"{metric}_mean"] = float(values.mean())
            item[f"{metric}_std"] = float(values.std(ddof=0))
        out.append(item)
    out.sort(key=lambda row: (str(row["bank"]), str(row["model_type"]), str(row["split"]), str(row["class_name"])))
    return out


def write_report(
    output_dir: Path,
    bank_rows: list[dict[str, Any]],
    seed_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    per_class_summary: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    best = max(summary_rows, key=lambda row: float(row["test_macro_f1_mean"]))
    lines = [
        "# Formal Revised Bank Comparison",
        "",
        "## Fixed Banks",
        "",
    ]
    for row in bank_rows:
        lines.append(
            f"- `{row['bank']}`: path=`{row['path']}`, total={row['total_after_dedup']}, "
            f"HGC={row['HGC_concept_count']}, LGC={row['LGC_concept_count']}, "
            f"NTL={row['NTL_concept_count']}, NST={row['NST_concept_count']}"
        )
    lines.extend(
        [
            "",
            "## Training Setup",
            "",
            f"- Revised trainer: `/home/kunet.ae/100069491/newcode/ebtc_cycl_retrieval_stage_revised.py` plus `/home/kunet.ae/100069491/experiment/etbc_wli_train_cbm_cycl.py`.",
            f"- Seeds: `{args.seeds}`.",
            f"- Schedule: `{args.schedule}`, warmup={args.warmup_epochs}, joint={args.joint_epochs}.",
            f"- Views: `{args.num_views}`, color_aug={args.use_color_aug}, geom_aug={args.use_geom_aug}.",
            f"- Image-concept pairing: `{args.use_image_text_pairing}`.",
            "",
            "## Mean +/- Std Results",
            "",
            "| Bank | Model | Val Macro-F1 | Test ACC | Test Macro-F1 | Test Macro-AUROC |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(summary_rows, key=lambda item: float(item["test_macro_f1_mean"]), reverse=True):
        lines.append(
            f"| {row['bank']} | {row['model_type']} | "
            f"{row['val_macro_f1_mean']:.4f} +/- {row['val_macro_f1_std']:.4f} | "
            f"{row['test_accuracy_mean']:.4f} +/- {row['test_accuracy_std']:.4f} | "
            f"{row['test_macro_f1_mean']:.4f} +/- {row['test_macro_f1_std']:.4f} | "
            f"{row['test_macro_auroc_mean']:.4f} +/- {row['test_macro_auroc_std']:.4f} |"
        )

    def get_summary(bank: str, model_type: str) -> dict[str, Any] | None:
        return next((row for row in summary_rows if row["bank"] == bank and row["model_type"] == model_type), None)

    lines.extend(["", "## CyCL Gain By Bank", ""])
    stable_banks: list[str] = []
    seed_lookup = {
        (str(row["bank"]), str(row["model_type"]), int(row["seed"])): row
        for row in seed_rows
    }
    per_bank_positive_seed_counts: dict[str, tuple[int, int]] = {}
    for bank in [row["bank"] for row in bank_rows]:
        cbm = get_summary(str(bank), "cbm_only")
        cycl = get_summary(str(bank), "cycl")
        if cbm is None or cycl is None:
            lines.append(f"- `{bank}`: incomplete, both CBM-no-CyCL and CyCL are required for delta.")
            continue
        delta = float(cycl["test_macro_f1_mean"]) - float(cbm["test_macro_f1_mean"])
        seed_deltas: list[float] = []
        for seed in parse_seed_list(args.seeds):
            cbm_seed = seed_lookup.get((str(bank), "cbm_only", seed))
            cycl_seed = seed_lookup.get((str(bank), "cycl", seed))
            if cbm_seed is not None and cycl_seed is not None:
                seed_deltas.append(float(cycl_seed["test_macro_f1"]) - float(cbm_seed["test_macro_f1"]))
        positive_count = sum(1 for item in seed_deltas if item > 0.0)
        per_bank_positive_seed_counts[str(bank)] = (positive_count, len(seed_deltas))
        seed_delta_text = ", ".join(f"{item:+.4f}" for item in seed_deltas)
        lines.append(
            f"- `{bank}`: mean CyCL - CBM-no-CyCL test Macro-F1 = `{delta:+.4f}`; "
            f"positive seeds={positive_count}/{len(seed_deltas)}; seed deltas=[{seed_delta_text}]."
        )
        if delta > 0:
            stable_banks.append(str(bank))

    lines.extend(["", "## LGC / NTL Test Per-Class F1", ""])
    for bank in [row["bank"] for row in bank_rows]:
        for model_type in ["cbm_only", "cycl"]:
            vals = []
            for class_name in ["LGC", "NTL"]:
                item = next(
                    (row
                    for row in per_class_summary
                    if row["bank"] == bank and row["model_type"] == model_type and row["split"] == "test" and row["class_name"] == class_name
                    ),
                    None,
                )
                if item is not None:
                    vals.append(f"{class_name}={item['f1_mean']:.4f}+/-{item['f1_std']:.4f}")
            if vals:
                lines.append(f"- `{model_type}_{bank}`: " + ", ".join(vals))

    weighted = get_summary("weighted_sym10", "cycl")
    old = get_summary("old_disc_sym10", "cycl")
    filtered = get_summary("filtered_top300", "cycl")
    available_cycl = [row for row in summary_rows if row["model_type"] == "cycl"]
    weighted_best = (
        weighted is not None
        and old is not None
        and filtered is not None
        and float(weighted["test_macro_f1_mean"]) >= float(old["test_macro_f1_mean"])
        and float(weighted["test_macro_f1_mean"]) >= float(filtered["test_macro_f1_mean"])
    )
    weighted_best_available = (
        weighted is not None
        and all(float(weighted["test_macro_f1_mean"]) >= float(row["test_macro_f1_mean"]) for row in available_cycl)
    )
    complete_delta_count = sum(1 for row in bank_rows if get_summary(str(row["bank"]), "cbm_only") is not None and get_summary(str(row["bank"]), "cycl") is not None)
    cycl_consistent = complete_delta_count > 0 and len(stable_banks) == complete_delta_count
    all_seed_consistent = all(count == total and total > 0 for count, total in per_bank_positive_seed_counts.values())
    best_cycl = max(
        (row for row in summary_rows if row["model_type"] == "cycl"),
        key=lambda row: float(row["test_macro_f1_mean"]),
        default=None,
    )
    best_lgc = max(
        (
            row
            for row in per_class_summary
            if row["split"] == "test" and row["class_name"] == "LGC"
        ),
        key=lambda row: float(row["f1_mean"]),
        default=None,
    )
    best_ntl = max(
        (
            row
            for row in per_class_summary
            if row["split"] == "test" and row["class_name"] == "NTL"
        ),
        key=lambda row: float(row["f1_mean"]),
        default=None,
    )

    lines.extend(
        [
            "",
            "## Answers",
            "",
            f"1. Best downstream bank by test Macro-F1: `{best['bank']}` with `{best['model_type']}`.",
            f"2. Best CyCL-only bank: `{best_cycl['bank'] if best_cycl is not None else 'NA'}`.",
            f"3. `weighted_sym10` is best among available CyCL settings in this output: `{weighted_best_available}`.",
            f"4. `weighted_sym10` beats both original formal baselines `old_disc_sym10` and `filtered_top300` under CyCL when both are present: `{weighted_best}`.",
            f"5. CyCL is better than CBM-no-CyCL on every completed bank pair by mean Macro-F1: `{cycl_consistent}`.",
            f"6. CyCL is better than CBM-no-CyCL on every seed within every bank: `{all_seed_consistent}`.",
            f"7. Best LGC test F1 setting: `{best_lgc['setting'] if best_lgc is not None else 'NA'}`.",
            f"8. Best NTL test F1 setting: `{best_ntl['setting'] if best_ntl is not None else 'NA'}`.",
            "",
            "## Conclusion",
            "",
            f"- Current recommended bank from this formal compare: `{best['bank']}`. If forcing CyCL-only, use `{best_cycl['bank'] if best_cycl is not None else 'NA'}`.",
            "- Revised CyCL should not be treated as the default winner in this run because its gain is not stable across banks/seeds; keep it as a targeted ablation, especially for NTL behavior.",
        ]
    )
    (output_dir / "formal_compare_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    requested_banks = parse_csv_list(args.run_banks)
    fixed_banks = resolve_fixed_banks(requested_banks)
    bank_rows = bank_manifest_rows(fixed_banks)
    write_csv(
        args.output_dir / "bank_manifest.csv",
        [
            "bank",
            "path",
            "total_before_dedup",
            "total_after_dedup",
            "dedup_removed",
            "HGC_concept_count",
            "LGC_concept_count",
            "NTL_concept_count",
            "NST_concept_count",
        ],
        bank_rows,
    )
    new_seed_rows = run_formal_compare(args, fixed_banks)
    seed_rows = merge_seed_rows(read_existing_seed_rows(args.output_dir / "seed_results.csv"), new_seed_rows)
    seed_fields = [
        "bank",
        "model_type",
        "bank_name",
        "seed",
        "setting",
        "concept_count_before_dedup",
        "concept_count_after_dedup",
        "num_views",
        "schedule",
        "ranking_method",
        "val_accuracy",
        "val_macro_f1",
        "val_macro_auroc",
        "test_accuracy",
        "test_macro_f1",
        "test_macro_auroc",
        "pair_positive_pair_count",
        "pair_negative_pair_count",
        "pair_positive_weight_mean",
        "pair_negative_weight_mean",
        "image_text_positive_similarity_mean",
        "image_text_negative_similarity_mean",
        "image_text_similarity_gap",
        "output_dir",
    ]
    write_csv(args.output_dir / "seed_results.csv", seed_fields, seed_rows)
    summary_rows = summarize_seed_results(seed_rows)
    summary_fields = [
        "bank",
        "model_type",
        "setting",
        "n_seeds",
        "seed_list",
        "concept_count_after_dedup",
        "val_accuracy_mean",
        "val_accuracy_std",
        "val_macro_f1_mean",
        "val_macro_f1_std",
        "val_macro_auroc_mean",
        "val_macro_auroc_std",
        "test_accuracy_mean",
        "test_accuracy_std",
        "test_macro_f1_mean",
        "test_macro_f1_std",
        "test_macro_auroc_mean",
        "test_macro_auroc_std",
    ]
    write_csv(args.output_dir / "results_summary.csv", summary_fields, summary_rows)
    per_seed_rows = collect_per_class_seed_rows(seed_rows)
    write_csv(
        args.output_dir / "per_class_seed_results.csv",
        ["bank", "model_type", "setting", "seed", "split", "class_name", "precision", "recall", "f1", "support"],
        per_seed_rows,
    )
    per_summary = summarize_per_class_metrics(per_seed_rows)
    write_csv(
        args.output_dir / "per_class_summary.csv",
        [
            "bank",
            "model_type",
            "setting",
            "split",
            "class_name",
            "n_seeds",
            "support_mean",
            "precision_mean",
            "precision_std",
            "recall_mean",
            "recall_std",
            "f1_mean",
            "f1_std",
        ],
        per_summary,
    )
    write_json(
        args.output_dir / "experiment_config.json",
        {
            "fixed_banks": bank_rows,
            "seeds": parse_seed_list(args.seeds),
            "run_models": parse_csv_list(args.run_models),
            "schedule": args.schedule,
            "baseline_epochs": args.baseline_epochs,
            "warmup_epochs": args.warmup_epochs,
            "joint_epochs": args.joint_epochs,
            "baseline_lr": args.baseline_lr,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "main_lr": args.main_lr,
            "weight_decay": args.weight_decay,
            "lambda_concept": args.lambda_concept,
            "lambda_align": args.lambda_align,
            "lambda_cycl": args.lambda_cycl,
            "lambda_img_txt": args.lambda_img_txt,
            "alpha": args.alpha,
            "tau": args.tau,
            "tau_img_txt": args.tau_img_txt,
            "dropout": args.dropout,
            "proj_dim": args.proj_dim,
            "num_views": args.num_views,
            "ranking_method": args.ranking_method,
            "use_color_aug": args.use_color_aug,
            "use_geom_aug": args.use_geom_aug,
            "use_image_text_pairing": args.use_image_text_pairing,
        },
    )
    write_report(args.output_dir, bank_rows, seed_rows, summary_rows, per_summary, args)
    best = max(summary_rows, key=lambda row: float(row["test_macro_f1_mean"]))
    print(
        f"[done] output_dir={args.output_dir} new_runs={len(new_seed_rows)} total_runs={len(seed_rows)} "
        f"best={best['model_type']}_{best['bank']} test_macro_f1={best['test_macro_f1_mean']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
