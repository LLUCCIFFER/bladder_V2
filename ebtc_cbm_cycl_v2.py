#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

try:
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
except ImportError:  # pragma: no cover - fallback keeps the script usable without sklearn.
    accuracy_score = None
    f1_score = None
    roc_auc_score = None

from ebtc_cbm_cycl_v2_lib import (
    CBMCyCLV2,
    LABEL_CODES,
    MultiViewDatasetV2,
    build_image_concept_pairs_v2,
    build_image_image_pairs_v2,
    build_soft_targets_v2,
    build_v2_view_transforms,
    compute_image_text_class_matrix,
    ensure_dir,
    load_fixed_bank_v2,
    load_split_embeddings,
    matrix_summary,
    matrix_to_rows,
    plot_matrix_heatmap,
    save_augmentation_preview,
    save_prepared_bank_cache,
    select_balanced_rows,
    write_concept_metadata,
    write_augmentation_config_summary,
    write_csv,
    write_json,
    write_matrix_summary_markdown,
    write_pair_debug_outputs,
    write_soft_target_summary,
    weighted_image_image_contrastive_loss_v2,
    weighted_image_text_contrastive_loss_v2,
)


DEFAULT_BANK_DIR = Path(
    "/home/kunet.ae/100069491/newcode/ebtc_cycl_retrieval_stage_revised_outputs/banks/retrieval_top10_per_class_disc"
)
DEFAULT_RETRIEVAL_TOP10_BANK_DIR = Path(
    "/home/kunet.ae/100069491/newcode/ebtc_cycl_retrieval_stage_revised_outputs/banks/retrieval_top10_per_class"
)
DEFAULT_FILTERED_TOP300_BANK_DIR = Path(
    "/home/kunet.ae/100069491/newcode/ebtc_cycl_retrieval_stage_revised_outputs/banks/filtered_top300"
)
DEFAULT_EMBEDDINGS_DIR = Path(
    "/home/kunet.ae/100069491/newcode/ebtc_official_split_embedding_cache/embeddings"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/kunet.ae/100069491/newcode/ebtc_cbm_cycl_v2_outputs/outputs_v2"
)
DEFAULT_FORMAL_4GROUPS_DIR = DEFAULT_OUTPUT_DIR / "formal_4groups_3seeds"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EBTC CBM/CyCL V2 utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    matrix_parser = subparsers.add_parser("matrix", help="Phase 1: compute 4x4 image-text class similarity matrices.")
    matrix_parser.add_argument("--bank-dir", type=Path, default=DEFAULT_BANK_DIR)
    matrix_parser.add_argument(
        "--compare-bank-dir",
        type=Path,
        action="append",
        default=[],
        help="Optional additional fixed bank directory. Can be supplied multiple times.",
    )
    matrix_parser.add_argument("--embeddings-dir", type=Path, default=DEFAULT_EMBEDDINGS_DIR)
    matrix_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    matrix_parser.add_argument(
        "--also-write-root-default",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also write matrix_train/val/test.csv and heatmaps at output root for --bank-dir.",
    )

    prepare_parser = subparsers.add_parser("prepare", help="Phase 2: build V2 soft targets and prototype caches.")
    prepare_parser.add_argument("--bank-dir", type=Path, default=DEFAULT_BANK_DIR)
    prepare_parser.add_argument("--compare-bank-dir", type=Path, action="append", default=[])
    prepare_parser.add_argument("--embeddings-dir", type=Path, default=DEFAULT_EMBEDDINGS_DIR)
    prepare_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    prepare_parser.add_argument("--also-write-root-default", action=argparse.BooleanOptionalAction, default=True)

    aug_parser = subparsers.add_parser("preview-augmentations", help="Phase 3: save grouped multi-view augmentation previews.")
    aug_parser.add_argument("--embeddings-dir", type=Path, default=DEFAULT_EMBEDDINGS_DIR)
    aug_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    aug_parser.add_argument("--n-views", type=int, default=4, choices=[2, 4, 5])
    aug_parser.add_argument("--max-images", type=int, default=4)

    pair_parser = subparsers.add_parser("debug-pairs", help="Phase 4: build explicit pair masks/weights and save diagnostics.")
    pair_parser.add_argument("--bank-dir", type=Path, default=DEFAULT_BANK_DIR)
    pair_parser.add_argument("--compare-bank-dir", type=Path, action="append", default=[])
    pair_parser.add_argument("--embeddings-dir", type=Path, default=DEFAULT_EMBEDDINGS_DIR)
    pair_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    pair_parser.add_argument("--n-views", type=int, default=4, choices=[2, 4, 5])
    pair_parser.add_argument("--debug-samples", type=int, default=64)

    smoke_parser = subparsers.add_parser("smoke-test", help="Phase 5.5: minimal V2 forward/loss smoke test, no full training.")
    smoke_parser.add_argument("--bank-dir", type=Path, default=DEFAULT_BANK_DIR)
    smoke_parser.add_argument("--embeddings-dir", type=Path, default=DEFAULT_EMBEDDINGS_DIR)
    smoke_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    smoke_parser.add_argument("--device", default="auto")
    smoke_parser.add_argument("--model-type", choices=["cbm_v2", "cycl_v2", "both"], default="both")
    smoke_parser.add_argument("--n-views", type=int, default=4, choices=[2, 4, 5])
    smoke_parser.add_argument("--batch-size", type=int, default=4)
    smoke_parser.add_argument("--max-batches", type=int, default=1)
    smoke_parser.add_argument("--dropout", type=float, default=0.2)
    smoke_parser.add_argument("--proj-dim", type=int, default=128)
    smoke_parser.add_argument("--lambda-concept", type=float, default=1.0)
    smoke_parser.add_argument("--lambda-align", type=float, default=0.3)
    smoke_parser.add_argument("--lambda-ii", type=float, default=0.1)
    smoke_parser.add_argument("--lambda-it", type=float, default=0.1)
    smoke_parser.add_argument("--tau", type=float, default=0.07)

    train_parser = subparsers.add_parser(
        "train-debug",
        help="Phase 6/6.5: short staged V2 debug training, not full formal training.",
    )
    train_parser.add_argument("--bank-dir", type=Path, default=DEFAULT_BANK_DIR)
    train_parser.add_argument("--embeddings-dir", type=Path, default=DEFAULT_EMBEDDINGS_DIR)
    train_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    train_parser.add_argument("--device", default="auto")
    train_parser.add_argument("--model-type", choices=["cbm_v2", "cycl_v2", "both"], default="both")
    train_parser.add_argument("--n-views", type=int, default=4, choices=[2, 4, 5])
    train_parser.add_argument("--batch-size", type=int, default=8)
    train_parser.add_argument("--num-workers", type=int, default=2)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--warmup-epochs", type=int, default=2)
    train_parser.add_argument("--joint-epochs", type=int, default=3)
    train_parser.add_argument(
        "--max-train-batches",
        type=int,
        default=12,
        help="Limit batches per epoch for debug. Use 0 for the full train split.",
    )
    train_parser.add_argument(
        "--max-val-batches",
        type=int,
        default=0,
        help="Limit validation batches for debug. Use 0 for the full val split.",
    )
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--dropout", type=float, default=0.2)
    train_parser.add_argument("--proj-dim", type=int, default=128)
    train_parser.add_argument("--lambda-concept", type=float, default=1.0)
    train_parser.add_argument("--lambda-align", type=float, default=0.3)
    train_parser.add_argument("--lambda-ii", type=float, default=0.1)
    train_parser.add_argument("--lambda-it", type=float, default=0.1)
    train_parser.add_argument("--tau", type=float, default=0.07)

    diagnostic_parser = subparsers.add_parser(
        "diagnostic-ablation",
        help="Diagnostic V2 ablation: CBM, image-image only, image-text only, and both contrastive branches.",
    )
    diagnostic_parser.add_argument("--bank-dir", type=Path, default=DEFAULT_BANK_DIR)
    diagnostic_parser.add_argument("--embeddings-dir", type=Path, default=DEFAULT_EMBEDDINGS_DIR)
    diagnostic_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    diagnostic_parser.add_argument("--device", default="auto")
    diagnostic_parser.add_argument("--n-views", type=int, default=4, choices=[4])
    diagnostic_parser.add_argument("--batch-size", type=int, default=8)
    diagnostic_parser.add_argument("--num-workers", type=int, default=2)
    diagnostic_parser.add_argument("--seed", type=int, default=42)
    diagnostic_parser.add_argument("--warmup-epochs", type=int, default=4)
    diagnostic_parser.add_argument("--joint-epochs", type=int, default=4)
    diagnostic_parser.add_argument("--max-train-batches", type=int, default=12)
    diagnostic_parser.add_argument("--max-val-batches", type=int, default=0)
    diagnostic_parser.add_argument("--lr", type=float, default=1e-3)
    diagnostic_parser.add_argument("--weight-decay", type=float, default=1e-4)
    diagnostic_parser.add_argument("--dropout", type=float, default=0.2)
    diagnostic_parser.add_argument("--proj-dim", type=int, default=128)
    diagnostic_parser.add_argument("--lambda-concept", type=float, default=1.0)
    diagnostic_parser.add_argument("--lambda-align", type=float, default=0.3)
    diagnostic_parser.add_argument("--lambda-ii", type=float, default=0.02)
    diagnostic_parser.add_argument("--lambda-it", type=float, default=0.02)
    diagnostic_parser.add_argument("--tau", type=float, default=0.07)

    formal_parser = subparsers.add_parser(
        "formal-it-compare",
        help="First formal V2 comparison: CBM baseline vs image-text-only CyCL on two top10 banks.",
    )
    formal_parser.add_argument("--primary-bank-dir", type=Path, default=DEFAULT_BANK_DIR)
    formal_parser.add_argument("--secondary-bank-dir", type=Path, default=DEFAULT_RETRIEVAL_TOP10_BANK_DIR)
    formal_parser.add_argument("--embeddings-dir", type=Path, default=DEFAULT_EMBEDDINGS_DIR)
    formal_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    formal_parser.add_argument("--device", default="auto")
    formal_parser.add_argument("--n-views", type=int, default=4, choices=[4])
    formal_parser.add_argument("--batch-size", type=int, default=16)
    formal_parser.add_argument("--num-workers", type=int, default=2)
    formal_parser.add_argument("--seed", type=int, action="append", default=None)
    formal_parser.add_argument("--warmup-epochs", type=int, default=5)
    formal_parser.add_argument("--joint-epochs", type=int, default=10)
    formal_parser.add_argument("--max-train-batches", type=int, default=0)
    formal_parser.add_argument("--max-val-batches", type=int, default=0)
    formal_parser.add_argument("--lr", type=float, default=1e-3)
    formal_parser.add_argument("--weight-decay", type=float, default=1e-4)
    formal_parser.add_argument("--dropout", type=float, default=0.2)
    formal_parser.add_argument("--proj-dim", type=int, default=128)
    formal_parser.add_argument("--lambda-concept", type=float, default=1.0)
    formal_parser.add_argument("--lambda-align", type=float, default=0.3)
    formal_parser.add_argument("--tau", type=float, default=0.07)
    formal_parser.add_argument(
        "--run-secondary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run secondary retrieval_top10_per_class comparison after primary bank.",
    )
    formal_parser.add_argument(
        "--main-four-groups-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run only CBM and image-text-only lambda=0.02 on primary and secondary banks.",
    )

    supplement_parser = subparsers.add_parser(
        "supplement-main-ablation",
        help="Supplementary main-model ablations on retrieval_top10_per_class only.",
    )
    supplement_parser.add_argument("--bank-dir", type=Path, default=DEFAULT_RETRIEVAL_TOP10_BANK_DIR)
    supplement_parser.add_argument("--embeddings-dir", type=Path, default=DEFAULT_EMBEDDINGS_DIR)
    supplement_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    supplement_parser.add_argument("--device", default="auto")
    supplement_parser.add_argument("--experiment", choices=["lambda_sweep", "view_sweep"], required=True)
    supplement_parser.add_argument("--lambda-it-value", type=float, action="append", default=None)
    supplement_parser.add_argument("--view-count", type=int, action="append", default=None, choices=[2, 4, 5])
    supplement_parser.add_argument("--seed", type=int, action="append", default=None)
    supplement_parser.add_argument("--n-views", type=int, default=4, choices=[2, 4, 5])
    supplement_parser.add_argument("--batch-size", type=int, default=16)
    supplement_parser.add_argument("--num-workers", type=int, default=2)
    supplement_parser.add_argument("--warmup-epochs", type=int, default=5)
    supplement_parser.add_argument("--joint-epochs", type=int, default=10)
    supplement_parser.add_argument("--max-train-batches", type=int, default=0)
    supplement_parser.add_argument("--max-val-batches", type=int, default=0)
    supplement_parser.add_argument("--lr", type=float, default=1e-3)
    supplement_parser.add_argument("--weight-decay", type=float, default=1e-4)
    supplement_parser.add_argument("--dropout", type=float, default=0.2)
    supplement_parser.add_argument("--proj-dim", type=int, default=128)
    supplement_parser.add_argument("--lambda-concept", type=float, default=1.0)
    supplement_parser.add_argument("--lambda-align", type=float, default=0.3)
    supplement_parser.add_argument("--tau", type=float, default=0.07)

    analysis_parser = subparsers.add_parser(
        "final-analysis",
        help="Final test-set per-class metrics, confusion matrices, and CBM concept explanations from trained checkpoints.",
    )
    analysis_parser.add_argument("--bank-dir", type=Path, default=DEFAULT_RETRIEVAL_TOP10_BANK_DIR)
    analysis_parser.add_argument("--embeddings-dir", type=Path, default=DEFAULT_EMBEDDINGS_DIR)
    analysis_parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_FORMAL_4GROUPS_DIR / "formal_curves" / "checkpoints")
    analysis_parser.add_argument("--output-dir", type=Path, default=DEFAULT_FORMAL_4GROUPS_DIR / "final_analysis")
    analysis_parser.add_argument("--device", default="auto")
    analysis_parser.add_argument("--seed", type=int, action="append", default=None)
    analysis_parser.add_argument("--cbm-setting-name", default="secondary_cbm_no_cycl_v2")
    analysis_parser.add_argument("--cycl-setting-name", default="secondary_cycl_it_lambda0_02")
    analysis_parser.add_argument("--explain-setting-name", default="secondary_cycl_it_lambda0_02")
    analysis_parser.add_argument("--explain-seed", type=int, default=42)
    analysis_parser.add_argument("--num-samples", type=int, default=8)
    analysis_parser.add_argument("--top-k-concepts", type=int, default=8)
    analysis_parser.add_argument("--dropout", type=float, default=0.2)
    analysis_parser.add_argument("--proj-dim", type=int, default=128)
    return parser.parse_args()


def unique_bank_dirs(primary: Path, extras: list[Path]) -> list[Path]:
    ordered: list[Path] = []
    seen: set[str] = set()
    for path in [primary, *extras]:
        resolved = str(path.resolve())
        if resolved not in seen:
            ordered.append(path)
            seen.add(resolved)
    return ordered


def write_bank_matrix_outputs(
    bank_dir: Path,
    embeddings_dir: Path,
    output_dir: Path,
    write_root_alias: bool,
) -> list[dict[str, Any]]:
    bank = load_fixed_bank_v2(bank_dir)
    split_payloads = load_split_embeddings(embeddings_dir)
    bank_output_dir = output_dir / bank.name
    ensure_dir(bank_output_dir)

    write_concept_metadata(bank, bank_output_dir / "concept_metadata_with_class.csv")
    if write_root_alias:
        write_concept_metadata(bank, output_dir / "concept_metadata_with_class.csv")

    summary_rows: list[dict[str, Any]] = []
    for split in ["train", "val", "test"]:
        matrix = compute_image_text_class_matrix(bank, split_payloads[split])
        rows = matrix_to_rows(matrix)
        matrix_fields = ["image_class", *LABEL_CODES]

        bank_matrix_path = bank_output_dir / f"matrix_{split}.csv"
        bank_heatmap_path = bank_output_dir / f"matrix_heatmap_{split}.png"
        write_csv(bank_matrix_path, matrix_fields, rows)
        plot_matrix_heatmap(matrix, bank_heatmap_path, f"{bank.name} image-text class similarity ({split})")

        if write_root_alias:
            write_csv(output_dir / f"matrix_{split}.csv", matrix_fields, rows)
            plot_matrix_heatmap(matrix, output_dir / f"matrix_heatmap_{split}.png", f"{bank.name} image-text class similarity ({split})")

        item = matrix_summary(bank.name, split, matrix)
        item["bank_dir"] = str(bank.bank_dir)
        item["matrix_csv"] = str(bank_matrix_path)
        item["heatmap_png"] = str(bank_heatmap_path)
        summary_rows.append(item)

    write_json(
        bank_output_dir / "phase1_bank_info.json",
        {
            "bank_name": bank.name,
            "bank_dir": str(bank.bank_dir),
            "n_concepts_total": len(bank.merged_concepts),
            "n_concepts_by_class": {class_name: len(bank.per_class_rows[class_name]) for class_name in LABEL_CODES},
            "class_label_rule": bank.class_label_rule,
        },
    )
    return summary_rows


def run_matrix(args: argparse.Namespace) -> None:
    ensure_dir(args.output_dir)
    bank_dirs = unique_bank_dirs(args.bank_dir, args.compare_bank_dir)
    all_summary_rows: list[dict[str, Any]] = []
    bank_paths: dict[str, Path] = {}
    for index, bank_dir in enumerate(bank_dirs):
        bank_name = bank_dir.resolve().name
        bank_paths[bank_name] = bank_dir.resolve()
        all_summary_rows.extend(
            write_bank_matrix_outputs(
                bank_dir=bank_dir,
                embeddings_dir=args.embeddings_dir,
                output_dir=args.output_dir,
                write_root_alias=args.also_write_root_default and index == 0,
            )
        )

    summary_fields = [
        "bank_name",
        "bank_dir",
        "split",
        "mean_diagonal",
        "mean_off_diagonal",
        "diagonal_minus_off_diagonal",
        "all_row_margins_positive",
        "row_margin_HGC",
        "row_margin_LGC",
        "row_margin_NTL",
        "row_margin_NST",
        "col_margin_HGC",
        "col_margin_LGC",
        "col_margin_NTL",
        "col_margin_NST",
        "matrix_csv",
        "heatmap_png",
    ]
    write_csv(args.output_dir / "matrix_summary.csv", summary_fields, all_summary_rows)
    write_matrix_summary_markdown(args.output_dir / "matrix_summary.md", all_summary_rows, bank_paths)
    print(f"[done] Phase 1 matrix outputs written to {args.output_dir} banks={len(bank_dirs)}", flush=True)


def build_caches_for_bank_dirs(bank_dirs: list[Path], embeddings_dir: Path) -> list[Any]:
    split_payloads = load_split_embeddings(embeddings_dir)
    caches = []
    for bank_dir in bank_dirs:
        bank = load_fixed_bank_v2(bank_dir)
        caches.append(build_soft_targets_v2(bank, split_payloads))
    return caches


def run_prepare(args: argparse.Namespace) -> None:
    ensure_dir(args.output_dir)
    bank_dirs = unique_bank_dirs(args.bank_dir, args.compare_bank_dir)
    caches = build_caches_for_bank_dirs(bank_dirs, args.embeddings_dir)
    for index, cache in enumerate(caches):
        save_prepared_bank_cache(cache, args.output_dir, write_root_alias=args.also_write_root_default and index == 0)
    write_soft_target_summary(args.output_dir / "soft_targets_cache_summary.md", caches)
    print(f"[done] Phase 2 prepared caches written to {args.output_dir} banks={len(caches)}", flush=True)


def run_preview_augmentations(args: argparse.Namespace) -> None:
    ensure_dir(args.output_dir)
    split_payloads = load_split_embeddings(args.embeddings_dir)
    preview_dir = args.output_dir / "augmentation_preview"
    preview_path, view_names = save_augmentation_preview(
        split_payloads=split_payloads,
        output_dir=preview_dir,
        n_views=args.n_views,
        max_images=args.max_images,
    )
    write_augmentation_config_summary(args.output_dir / "augmentation_config_summary.md", args.n_views, view_names, preview_path)
    print(f"[done] Phase 3 augmentation preview written to {preview_path}", flush=True)


def run_debug_pairs(args: argparse.Namespace) -> None:
    ensure_dir(args.output_dir)
    bank_dirs = unique_bank_dirs(args.bank_dir, args.compare_bank_dir)
    caches = build_caches_for_bank_dirs(bank_dirs, args.embeddings_dir)
    combined_rows: list[dict[str, Any]] = []
    for index, cache in enumerate(caches):
        target_dir = args.output_dir if index == 0 else args.output_dir / cache.bank.name
        rows = write_pair_debug_outputs(
            path_md=target_dir / "pair_debug_summary.md",
            path_csv=target_dir / "pair_debug_tables.csv",
            cache=cache,
            n_views=args.n_views,
            debug_samples=args.debug_samples,
        )
        combined_rows.extend(rows)
    write_csv(args.output_dir / "pair_debug_tables.csv", ["bank_name", "n_views", "debug_samples", "metric", "value"], combined_rows)
    # The root summary is already written for the default bank. Keep a short combined index for comparison banks.
    lines = [
        "# Phase 4 Pair Debug Combined Index",
        "",
        f"Default bank summary: `{args.output_dir / 'pair_debug_summary.md'}`",
        "",
        "Banks included:",
        "",
        *[f"- `{cache.bank.name}`" for cache in caches],
        "",
        f"Combined table: `{args.output_dir / 'pair_debug_tables.csv'}`",
    ]
    (args.output_dir / "pair_debug_index.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[done] Phase 4 pair debug written to {args.output_dir} banks={len(caches)}", flush=True)


def get_normalize_transform(eval_transform: Any) -> Any:
    for transform in reversed(eval_transform.transforms):
        if isinstance(transform, transforms.Normalize):
            return transform
    raise RuntimeError("Could not locate Normalize transform in BioMedCLIP preprocess.")


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def import_biomedclip_loader() -> Any:
    newcode_dir = Path("/home/kunet.ae/100069491/newcode")
    experiment_dir = Path("/home/kunet.ae/100069491/experiment")
    for path in [newcode_dir, experiment_dir]:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from ebtc_cycl_retrieval_stage_revised import load_biomedclip  # noqa: WPS433

    return load_biomedclip


def run_single_smoke_model(
    model_type: str,
    args: argparse.Namespace,
    cache: Any,
    image_encoder: Any,
    preprocess_val: Any,
    device: str,
) -> list[str]:
    normalize_transform = get_normalize_transform(preprocess_val)
    view_transforms, view_names = build_v2_view_transforms(normalize_transform=normalize_transform, n_views=args.n_views)
    rows = select_balanced_rows(cache.split_rows["train"], max(args.batch_size * args.max_batches, args.batch_size))
    dataset = MultiViewDatasetV2(rows=rows, prototype_matrix=cache.prototype_matrix, view_transforms=view_transforms)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    embedding_dim = int(cache.bank.merged_embeddings.shape[1])
    model = CBMCyCLV2(
        embedding_dim=embedding_dim,
        num_concepts=len(cache.bank.merged_concepts),
        num_classes=len(LABEL_CODES),
        dropout=args.dropout,
        proj_dim=args.proj_dim,
    ).to(device)
    model.train()
    text_embeddings = torch.tensor(cache.bank.merged_embeddings, dtype=torch.float32, device=device)
    concept_membership = torch.tensor(cache.bank.concept_class_membership, dtype=torch.float32, device=device)

    log_lines = [
        f"## {model_type}",
        f"bank={cache.bank.name}",
        f"n_views={args.n_views}",
        f"view_names={view_names}",
    ]
    for batch_index, (views, labels, soft_targets, prototypes, _) in enumerate(loader):
        if batch_index >= args.max_batches:
            break
        views = views.to(device)
        labels = labels.to(device)
        soft_targets = soft_targets.to(device)
        prototypes = prototypes.to(device)
        batch_size, n_views = views.shape[:2]
        flat_views = views.reshape(batch_size * n_views, *views.shape[2:])
        with torch.no_grad():
            features = image_encoder.encode_image(flat_views)
            features = F.normalize(features, dim=-1).float()

        outputs = model.forward_image_features(features)
        concept_scores = outputs["concept_scores"].reshape(batch_size, n_views, -1)
        logits = outputs["logits"].reshape(batch_size, n_views, len(LABEL_CODES))
        projections = outputs["image_projection"].reshape(batch_size, n_views, -1)
        text_projections = model.forward_text_embeddings(text_embeddings)

        expanded_soft = soft_targets.unsqueeze(1).expand(-1, n_views, -1)
        expanded_proto = prototypes.unsqueeze(1).expand(-1, n_views, -1)
        concept_loss = F.mse_loss(concept_scores, expanded_soft)
        align_loss = F.mse_loss(concept_scores, expanded_proto)
        cls_loss = F.cross_entropy(
            logits.reshape(batch_size * n_views, len(LABEL_CODES)),
            labels.unsqueeze(1).expand(-1, n_views).reshape(-1),
        )

        ii_loss = torch.tensor(0.0, device=device)
        it_loss = torch.tensor(0.0, device=device)
        ii_diag: dict[str, Any] = {}
        ic_diag: dict[str, Any] = {}
        if model_type == "cycl_v2":
            _, _, ii_weights, ii_diag = build_image_image_pairs_v2(labels, soft_targets, n_views=n_views)
            ii_loss = weighted_image_image_contrastive_loss_v2(projections, ii_weights.to(device), tau=args.tau)
            _, _, ic_weights, ic_diag = build_image_concept_pairs_v2(labels, soft_targets, concept_membership)
            ic_weights_flat = ic_weights.repeat_interleave(n_views, dim=0).to(device)
            it_loss = weighted_image_text_contrastive_loss_v2(
                image_projections=projections.reshape(batch_size * n_views, -1),
                text_projections=text_projections,
                positive_weights=ic_weights_flat,
                tau=args.tau,
            )

        total = (
            cls_loss
            + args.lambda_concept * concept_loss
            + args.lambda_align * align_loss
            + args.lambda_ii * ii_loss
            + args.lambda_it * it_loss
        )
        losses = {
            "cls_loss": cls_loss,
            "concept_loss": concept_loss,
            "align_loss": align_loss,
            "image_image_loss": ii_loss,
            "image_text_loss": it_loss,
            "total_loss": total,
        }
        for name, value in losses.items():
            if not torch.isfinite(value):
                raise RuntimeError(f"Non-finite smoke loss: model={model_type} batch={batch_index} loss={name} value={value}")
        total.backward()
        log_lines.extend(
            [
                f"batch={batch_index}",
                f"views_shape={list(views.shape)}",
                f"features_shape={list(features.shape)}",
                f"concept_scores_shape={list(concept_scores.shape)}",
                f"text_projection_shape={list(text_projections.shape)}",
                f"cls_loss={float(cls_loss.item()):.6f}",
                f"concept_loss={float(concept_loss.item()):.6f}",
                f"align_loss={float(align_loss.item()):.6f}",
                f"image_image_loss={float(ii_loss.item()):.6f}",
                f"image_text_loss={float(it_loss.item()):.6f}",
                f"total_loss={float(total.item()):.6f}",
                f"ii_diag={ii_diag}",
                f"ic_diag={ic_diag}",
            ]
        )
    return log_lines


def run_smoke_test(args: argparse.Namespace) -> None:
    ensure_dir(args.output_dir)
    load_biomedclip = import_biomedclip_loader()
    image_encoder, _, preprocess_val, resolved_device = load_biomedclip(args.device)
    device = resolve_device(resolved_device)
    image_encoder.to(device)
    image_encoder.eval()
    for parameter in image_encoder.parameters():
        parameter.requires_grad = False

    split_payloads = load_split_embeddings(args.embeddings_dir)
    bank = load_fixed_bank_v2(args.bank_dir)
    cache = build_soft_targets_v2(bank, split_payloads)
    model_types = ["cbm_v2", "cycl_v2"] if args.model_type == "both" else [args.model_type]
    all_lines = [
        "# Phase 5.5 Smoke Test Log",
        f"device={device}",
        f"bank={bank.name}",
        f"bank_dir={bank.bank_dir}",
        "",
    ]
    for model_type in model_types:
        all_lines.extend(run_single_smoke_model(model_type, args, cache, image_encoder, preprocess_val, device))
        all_lines.append("")
    (args.output_dir / "smoke_test_log.txt").write_text("\n".join(all_lines), encoding="utf-8")
    summary = [
        "# Phase 5 V2 Model Debug Summary",
        "",
        "Implemented V2 model components:",
        "",
        "- Frozen BioMedCLIP image encoder is external and remains frozen.",
        "- Image adapter maps frozen image embeddings to a refined hidden representation.",
        "- Concept head predicts concept activations.",
        "- Classifier consumes concept activations only.",
        "- Image projection head produces image-side contrastive representations.",
        "- Text adapter/projection branch refines frozen concept text embeddings for image-concept contrastive learning.",
        "",
        "Smoke test status: passed finite forward/loss/backward checks for requested model type(s).",
        "",
        f"Log file: `{args.output_dir / 'smoke_test_log.txt'}`",
    ]
    (args.output_dir / "model_debug_summary.md").write_text("\n".join(summary), encoding="utf-8")
    print(f"[done] Phase 5.5 smoke test written to {args.output_dir / 'smoke_test_log.txt'}", flush=True)


def set_reproducible_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_class_weights(cache: Any, device: str) -> torch.Tensor:
    labels = np.array([int(row["label_index"]) for row in cache.split_rows["train"]], dtype=np.int64)
    counts = np.bincount(labels, minlength=len(LABEL_CODES)).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def build_model_for_cache(args: argparse.Namespace, cache: Any, device: str) -> CBMCyCLV2:
    embedding_dim = int(cache.bank.merged_embeddings.shape[1])
    model = CBMCyCLV2(
        embedding_dim=embedding_dim,
        num_concepts=len(cache.bank.merged_concepts),
        num_classes=len(LABEL_CODES),
        dropout=args.dropout,
        proj_dim=args.proj_dim,
    )
    return model.to(device)


def manual_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    per_class_f1: list[float] = []
    for class_index in range(len(LABEL_CODES)):
        true_positive = float(((y_true == class_index) & (y_pred == class_index)).sum())
        false_positive = float(((y_true != class_index) & (y_pred == class_index)).sum())
        false_negative = float(((y_true == class_index) & (y_pred != class_index)).sum())
        precision = true_positive / max(true_positive + false_positive, 1.0)
        recall = true_positive / max(true_positive + false_negative, 1.0)
        per_class_f1.append(2.0 * precision * recall / max(precision + recall, 1e-8))
    return float(np.mean(per_class_f1))


def compute_prediction_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = probabilities.argmax(axis=1)
    if accuracy_score is not None:
        accuracy = float(accuracy_score(labels, predictions))
    else:
        accuracy = float((labels == predictions).mean())
    if f1_score is not None:
        macro_f1 = float(f1_score(labels, predictions, average="macro", zero_division=0))
    else:
        macro_f1 = manual_macro_f1(labels, predictions)
    macro_auroc = float("nan")
    if roc_auc_score is not None:
        try:
            one_hot = np.eye(len(LABEL_CODES), dtype=np.float32)[labels]
            macro_auroc = float(roc_auc_score(one_hot, probabilities, average="macro", multi_class="ovr"))
        except ValueError:
            macro_auroc = float("nan")
    metrics = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "macro_auroc": macro_auroc,
    }
    for class_index, class_name in enumerate(LABEL_CODES):
        positives = labels == class_index
        true_positive = (predictions[positives] == class_index).sum() if positives.any() else 0
        metrics[f"recall_{class_name}"] = float(true_positive / max(int(positives.sum()), 1))
    return metrics


def compute_supervised_v2_losses(
    model_type: str,
    phase: str,
    args: argparse.Namespace,
    model: CBMCyCLV2,
    features: torch.Tensor,
    labels: torch.Tensor,
    soft_targets: torch.Tensor,
    prototypes: torch.Tensor,
    text_embeddings: torch.Tensor,
    concept_membership: torch.Tensor,
    class_weights: torch.Tensor,
    n_views: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, torch.Tensor]]:
    outputs = model.forward_image_features(features)
    batch_size = labels.shape[0]
    concept_scores = outputs["concept_scores"].reshape(batch_size, n_views, -1)
    logits = outputs["logits"].reshape(batch_size, n_views, len(LABEL_CODES))
    projections = outputs["image_projection"].reshape(batch_size, n_views, -1)

    expanded_soft = soft_targets.unsqueeze(1).expand(-1, n_views, -1)
    expanded_proto = prototypes.unsqueeze(1).expand(-1, n_views, -1)
    concept_loss = F.mse_loss(concept_scores, expanded_soft)
    align_loss = F.mse_loss(concept_scores, expanded_proto)

    zero = torch.tensor(0.0, device=features.device)
    cls_loss = zero
    ii_loss = zero
    it_loss = zero
    diagnostics: dict[str, Any] = {}

    if phase == "joint":
        cls_loss = F.cross_entropy(
            logits.reshape(batch_size * n_views, len(LABEL_CODES)),
            labels.unsqueeze(1).expand(-1, n_views).reshape(-1),
            weight=class_weights,
        )
        if model_type == "cycl_v2" and args.lambda_ii > 0.0:
            _, _, ii_weights, ii_diag = build_image_image_pairs_v2(labels, soft_targets, n_views=n_views)
            ii_loss = weighted_image_image_contrastive_loss_v2(projections, ii_weights.to(features.device), tau=args.tau)
            diagnostics.update(ii_diag)
        if model_type == "cycl_v2" and args.lambda_it > 0.0:
            _, _, ic_weights, ic_diag = build_image_concept_pairs_v2(labels, soft_targets, concept_membership)
            text_projections = model.forward_text_embeddings(text_embeddings)
            it_loss = weighted_image_text_contrastive_loss_v2(
                image_projections=projections.reshape(batch_size * n_views, -1),
                text_projections=text_projections,
                positive_weights=ic_weights.repeat_interleave(n_views, dim=0).to(features.device),
                tau=args.tau,
            )
            diagnostics.update(ic_diag)

    total_loss = args.lambda_concept * concept_loss + args.lambda_align * align_loss
    if phase == "joint":
        total_loss = total_loss + cls_loss
        if model_type == "cycl_v2":
            total_loss = total_loss + args.lambda_ii * ii_loss + args.lambda_it * it_loss

    losses = {
        "total_loss": total_loss,
        "cls_loss": cls_loss,
        "concept_loss": concept_loss,
        "align_loss": align_loss,
        "image_image_loss": ii_loss,
        "image_text_loss": it_loss,
    }
    return losses, diagnostics, outputs


def evaluate_v2_model(
    model: CBMCyCLV2,
    cache: Any,
    split_payloads: dict[str, Any],
    split: str,
    device: str,
    batch_size: int,
    class_weights: torch.Tensor,
    lambda_align: float,
    max_batches: int = 0,
) -> dict[str, Any]:
    model.eval()
    rows = cache.split_rows[split]
    embeddings = split_payloads[split].embeddings.astype(np.float32)
    prototype_matrix = torch.tensor(cache.prototype_matrix, dtype=torch.float32, device=device)
    all_labels: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []
    loss_sums = {"cls_loss": 0.0, "concept_loss": 0.0, "align_loss": 0.0, "total_loss": 0.0}
    sample_count = 0

    with torch.no_grad():
        for batch_index, start in enumerate(range(0, len(rows), batch_size)):
            if max_batches and batch_index >= max_batches:
                break
            end = min(start + batch_size, len(rows))
            features = torch.tensor(embeddings[start:end], dtype=torch.float32, device=device)
            labels = torch.tensor([int(row["label_index"]) for row in rows[start:end]], dtype=torch.long, device=device)
            soft_targets = torch.tensor(
                np.stack([row["soft_concepts"] for row in rows[start:end]]),
                dtype=torch.float32,
                device=device,
            )
            prototypes = prototype_matrix[labels]

            outputs = model.forward_image_features(features)
            logits = outputs["logits"]
            concept_scores = outputs["concept_scores"]
            cls_loss = F.cross_entropy(logits, labels, weight=class_weights)
            concept_loss = F.mse_loss(concept_scores, soft_targets)
            align_loss = F.mse_loss(concept_scores, prototypes)
            total_loss = cls_loss + concept_loss + lambda_align * align_loss

            probabilities = torch.softmax(logits, dim=1)
            all_labels.append(labels.detach().cpu().numpy())
            all_probabilities.append(probabilities.detach().cpu().numpy())
            current_count = end - start
            sample_count += current_count
            loss_sums["cls_loss"] += float(cls_loss.item()) * current_count
            loss_sums["concept_loss"] += float(concept_loss.item()) * current_count
            loss_sums["align_loss"] += float(align_loss.item()) * current_count
            loss_sums["total_loss"] += float(total_loss.item()) * current_count

    labels_array = np.concatenate(all_labels, axis=0)
    probabilities_array = np.concatenate(all_probabilities, axis=0)
    metrics = compute_prediction_metrics(labels_array, probabilities_array)
    output = {
        f"{split}_accuracy": metrics["accuracy"],
        f"{split}_macro_f1": metrics["macro_f1"],
        f"{split}_macro_auroc": metrics["macro_auroc"],
        f"{split}_samples": int(sample_count),
    }
    for class_name in LABEL_CODES:
        output[f"{split}_recall_{class_name}"] = metrics[f"recall_{class_name}"]
    for name, value in loss_sums.items():
        output[f"{split}_{name}"] = value / max(sample_count, 1)
    return output


def train_one_debug_model(
    model_type: str,
    args: argparse.Namespace,
    cache: Any,
    split_payloads: dict[str, Any],
    image_encoder: Any,
    preprocess_val: Any,
    device: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    normalize_transform = get_normalize_transform(preprocess_val)
    view_transforms, view_names = build_v2_view_transforms(normalize_transform=normalize_transform, n_views=args.n_views)
    dataset = MultiViewDatasetV2(
        rows=cache.split_rows["train"],
        prototype_matrix=cache.prototype_matrix,
        view_transforms=view_transforms,
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"),
        generator=generator,
    )

    model = build_model_for_cache(args, cache, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    text_embeddings = torch.tensor(cache.bank.merged_embeddings, dtype=torch.float32, device=device)
    concept_membership = torch.tensor(cache.bank.concept_class_membership, dtype=torch.float32, device=device)
    class_weights = compute_class_weights(cache, device)
    total_epochs = args.warmup_epochs + args.joint_epochs
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []

    for epoch in range(1, total_epochs + 1):
        phase = "warmup" if epoch <= args.warmup_epochs else "joint"
        model.train()
        running = {
            "total_loss": 0.0,
            "cls_loss": 0.0,
            "concept_loss": 0.0,
            "align_loss": 0.0,
            "image_image_loss": 0.0,
            "image_text_loss": 0.0,
        }
        samples = 0
        batch_count = 0

        for batch_index, (views, labels, soft_targets, prototypes, _) in enumerate(loader):
            if args.max_train_batches and batch_index >= args.max_train_batches:
                break
            views = views.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            soft_targets = soft_targets.to(device, non_blocking=True)
            prototypes = prototypes.to(device, non_blocking=True)
            batch_size, n_views = views.shape[:2]
            flat_views = views.reshape(batch_size * n_views, *views.shape[2:])

            with torch.no_grad():
                features = image_encoder.encode_image(flat_views)
                features = F.normalize(features, dim=-1).float()

            losses, _, _ = compute_supervised_v2_losses(
                model_type=model_type,
                phase=phase,
                args=args,
                model=model,
                features=features,
                labels=labels,
                soft_targets=soft_targets,
                prototypes=prototypes,
                text_embeddings=text_embeddings,
                concept_membership=concept_membership,
                class_weights=class_weights,
                n_views=n_views,
            )
            for name, value in losses.items():
                if not torch.isfinite(value):
                    raise RuntimeError(f"Non-finite train loss: model={model_type} epoch={epoch} batch={batch_index} {name}={value}")
            optimizer.zero_grad(set_to_none=True)
            losses["total_loss"].backward()
            optimizer.step()

            samples += batch_size
            batch_count += 1
            for name, value in losses.items():
                running[name] += float(value.detach().item()) * batch_size

        row: dict[str, Any] = {
            "model_type": model_type,
            "bank_name": cache.bank.name,
            "seed": args.seed,
            "epoch": epoch,
            "phase": phase,
            "n_views": args.n_views,
            "view_names": " ".join(view_names),
            "train_batches": batch_count,
            "train_samples": samples,
            "lr": args.lr,
            "warmup_epochs": args.warmup_epochs,
            "joint_epochs": args.joint_epochs,
            "max_train_batches": args.max_train_batches,
        }
        for name, value in running.items():
            row[name] = value / max(samples, 1)
        train_rows.append(row)

        val_metrics = evaluate_v2_model(
            model=model,
            cache=cache,
            split_payloads=split_payloads,
            split="val",
            device=device,
            batch_size=args.batch_size * 4,
            class_weights=class_weights,
            lambda_align=args.lambda_align,
            max_batches=args.max_val_batches,
        )
        val_rows.append(
            {
                "model_type": model_type,
                "bank_name": cache.bank.name,
                "seed": args.seed,
                "epoch": epoch,
                "phase": phase,
                **val_metrics,
            }
        )
        print(
            f"[train-debug] {model_type} epoch={epoch}/{total_epochs} phase={phase} "
            f"train_total={row['total_loss']:.4f} val_macro_f1={val_metrics['val_macro_f1']:.4f}",
            flush=True,
        )
    return train_rows, val_rows


def plot_loss_curves(rows: list[dict[str, Any]], output_path: Path, title: str) -> None:
    ensure_dir(output_path.parent)
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    epochs = [int(row["epoch"]) for row in rows]
    for loss_name in ["total_loss", "cls_loss", "concept_loss", "align_loss", "image_image_loss", "image_text_loss"]:
        values = [float(row[loss_name]) for row in rows]
        ax.plot(epochs, values, marker="o", label=loss_name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train loss")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_debug_training_summary(
    output_path: Path,
    args: argparse.Namespace,
    cache: Any,
    command_line: str,
    train_by_model: dict[str, list[dict[str, Any]]],
    val_by_model: dict[str, list[dict[str, Any]]],
) -> None:
    lines = [
        "# Phase 6.5 Debug Training Summary",
        "",
        "This is a short debug run only. It verifies that V2 warmup/joint training is finite and internally consistent; it is not a formal result.",
        "",
        "## Configuration",
        "",
        f"- Bank: `{cache.bank.name}`",
        f"- Bank path: `{cache.bank.bank_dir}`",
        f"- Seed: `{args.seed}`",
        f"- Views: `{args.n_views}`",
        f"- Warmup epochs: `{args.warmup_epochs}`",
        f"- Joint epochs: `{args.joint_epochs}`",
        f"- Batch size: `{args.batch_size}`",
        f"- Max train batches per epoch: `{args.max_train_batches}`",
        f"- Max val batches: `{args.max_val_batches}`",
        f"- LR / weight decay: `{args.lr}` / `{args.weight_decay}`",
        f"- Loss weights: lambda_concept={args.lambda_concept}, lambda_align={args.lambda_align}, lambda_ii={args.lambda_ii}, lambda_it={args.lambda_it}",
        "",
        "Warmup uses concept loss + prototype/alignment loss only. Classification, image-image contrastive, and image-text contrastive are not included during warmup.",
        "",
        "Joint CBM-no-CyCL V2 uses cls + concept + align only. It strictly excludes image-image and image-text contrastive losses.",
        "",
        "Joint CyCL V2 uses cls + concept + align + image-image contrastive + image-text contrastive.",
        "",
        "Command:",
        "",
        f"```bash\n{command_line}\n```",
        "",
        "## Final Epoch Metrics",
        "",
        "| Model | Final Phase | Train Total | Train Cls | Train Concept | Train Align | Train II | Train IT | Val Acc | Val Macro-F1 | Val Macro-AUROC |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_type in train_by_model:
        train_last = train_by_model[model_type][-1]
        val_last = val_by_model[model_type][-1]
        lines.append(
            f"| `{model_type}` | `{train_last['phase']}` | "
            f"{float(train_last['total_loss']):.6f} | {float(train_last['cls_loss']):.6f} | "
            f"{float(train_last['concept_loss']):.6f} | {float(train_last['align_loss']):.6f} | "
            f"{float(train_last['image_image_loss']):.6f} | {float(train_last['image_text_loss']):.6f} | "
            f"{float(val_last['val_accuracy']):.6f} | {float(val_last['val_macro_f1']):.6f} | "
            f"{float(val_last['val_macro_auroc']):.6f} |"
        )

    lines.extend(["", "## Stability Check", ""])
    for model_type, rows in train_by_model.items():
        finite = all(np.isfinite(float(row[name])) for row in rows for name in ["total_loss", "cls_loss", "concept_loss", "align_loss", "image_image_loss", "image_text_loss"])
        first_total = float(rows[0]["total_loss"])
        final_total = float(rows[-1]["total_loss"])
        final_it = float(rows[-1]["image_text_loss"])
        final_ii = float(rows[-1]["image_image_loss"])
        lines.append(
            f"- `{model_type}`: finite_losses={finite}, first_total={first_total:.6f}, final_total={final_total:.6f}, "
            f"final_image_image={final_ii:.6f}, final_image_text={final_it:.6f}."
        )
    lines.extend(
        [
            "",
            "Loss dominance interpretation should be conservative because this debug run uses a limited number of train batches per epoch.",
            "",
            "## Output Files",
            "",
            f"- `train_debug_cbm_v2.csv`",
            f"- `train_debug_cycl_v2.csv`",
            f"- `val_debug_cbm_v2.csv`",
            f"- `val_debug_cycl_v2.csv`",
            f"- `loss_curves_cbm_v2.png`",
            f"- `loss_curves_cycl_v2.png`",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_train_debug(args: argparse.Namespace) -> None:
    ensure_dir(args.output_dir)
    set_reproducible_seed(args.seed)
    load_biomedclip = import_biomedclip_loader()
    image_encoder, _, preprocess_val, resolved_device = load_biomedclip(args.device)
    device = resolve_device(resolved_device)
    image_encoder.to(device)
    image_encoder.eval()
    for parameter in image_encoder.parameters():
        parameter.requires_grad = False

    split_payloads = load_split_embeddings(args.embeddings_dir)
    bank = load_fixed_bank_v2(args.bank_dir)
    cache = build_soft_targets_v2(bank, split_payloads)
    model_types = ["cbm_v2", "cycl_v2"] if args.model_type == "both" else [args.model_type]
    train_by_model: dict[str, list[dict[str, Any]]] = {}
    val_by_model: dict[str, list[dict[str, Any]]] = {}

    for index, model_type in enumerate(model_types):
        set_reproducible_seed(args.seed + index)
        train_rows, val_rows = train_one_debug_model(
            model_type=model_type,
            args=args,
            cache=cache,
            split_payloads=split_payloads,
            image_encoder=image_encoder,
            preprocess_val=preprocess_val,
            device=device,
        )
        train_by_model[model_type] = train_rows
        val_by_model[model_type] = val_rows
        suffix = "cbm_v2" if model_type == "cbm_v2" else "cycl_v2"
        train_fields = [
            "model_type",
            "bank_name",
            "seed",
            "epoch",
            "phase",
            "n_views",
            "view_names",
            "train_batches",
            "train_samples",
            "lr",
            "warmup_epochs",
            "joint_epochs",
            "max_train_batches",
            "total_loss",
            "cls_loss",
            "concept_loss",
            "align_loss",
            "image_image_loss",
            "image_text_loss",
        ]
        val_fields = [
            "model_type",
            "bank_name",
            "seed",
            "epoch",
            "phase",
            "val_accuracy",
            "val_macro_f1",
            "val_macro_auroc",
            "val_samples",
            "val_cls_loss",
            "val_concept_loss",
            "val_align_loss",
            "val_total_loss",
        ]
        write_csv(args.output_dir / f"train_debug_{suffix}.csv", train_fields, train_rows)
        write_csv(args.output_dir / f"val_debug_{suffix}.csv", val_fields, val_rows)
        plot_loss_curves(train_rows, args.output_dir / f"loss_curves_{suffix}.png", f"{model_type} V2 debug train losses")

    write_debug_training_summary(
        output_path=args.output_dir / "debug_training_summary.md",
        args=args,
        cache=cache,
        command_line="python " + " ".join(sys.argv),
        train_by_model=train_by_model,
        val_by_model=val_by_model,
    )
    print(f"[done] Phase 6.5 debug training written to {args.output_dir}", flush=True)


def update_scalar_stats(stats: dict[str, float], values: torch.Tensor) -> None:
    values = values.detach().float().reshape(-1)
    if values.numel() == 0:
        return
    stats["count"] += float(values.numel())
    stats["sum"] += float(values.sum().item())
    stats["sumsq"] += float((values * values).sum().item())
    stats["min"] = min(stats["min"], float(values.min().item()))
    stats["max"] = max(stats["max"], float(values.max().item()))


def finalize_scalar_stats(stats: dict[str, float], prefix: str) -> dict[str, Any]:
    count = max(stats["count"], 1.0)
    mean = stats["sum"] / count
    variance = max(stats["sumsq"] / count - mean * mean, 0.0)
    return {
        f"{prefix}_count": int(stats["count"]),
        f"{prefix}_mean": mean,
        f"{prefix}_std": float(np.sqrt(variance)),
        f"{prefix}_min": stats["min"] if stats["count"] else 0.0,
        f"{prefix}_max": stats["max"] if stats["count"] else 0.0,
    }


def init_scalar_stats() -> dict[str, float]:
    return {"count": 0.0, "sum": 0.0, "sumsq": 0.0, "min": float("inf"), "max": float("-inf")}


def update_pair_stats(
    pair_stats: dict[str, dict[str, float]],
    labels: torch.Tensor,
    soft_targets: torch.Tensor,
    concept_membership: torch.Tensor,
    n_views: int,
) -> None:
    with torch.no_grad():
        _, _, ii_weights, _ = build_image_image_pairs_v2(labels, soft_targets, n_views=n_views)
        update_scalar_stats(pair_stats["image_image_positive_weight"], ii_weights[ii_weights > 0.0])
        _, _, ic_weights, _ = build_image_concept_pairs_v2(labels, soft_targets, concept_membership)
        update_scalar_stats(pair_stats["image_text_positive_weight"], ic_weights[ic_weights > 0.0])


def update_concept_activation_stats(
    concept_stats: dict[str, float],
    concept_scores: torch.Tensor,
) -> None:
    values = concept_scores.detach().float().reshape(-1)
    if values.numel() == 0:
        return
    concept_stats["count"] += float(values.numel())
    concept_stats["sum"] += float(values.sum().item())
    concept_stats["sumsq"] += float((values * values).sum().item())
    concept_stats["min"] = min(concept_stats["min"], float(values.min().item()))
    concept_stats["max"] = max(concept_stats["max"], float(values.max().item()))
    concept_stats["lt_0_1"] += float((values < 0.1).sum().item())
    concept_stats["lt_0_2"] += float((values < 0.2).sum().item())
    concept_stats["gt_0_8"] += float((values > 0.8).sum().item())


def init_concept_activation_stats() -> dict[str, float]:
    return {
        "count": 0.0,
        "sum": 0.0,
        "sumsq": 0.0,
        "min": float("inf"),
        "max": float("-inf"),
        "lt_0_1": 0.0,
        "lt_0_2": 0.0,
        "gt_0_8": 0.0,
    }


def finalize_concept_activation_stats(stats: dict[str, float]) -> dict[str, Any]:
    count = max(stats["count"], 1.0)
    mean = stats["sum"] / count
    variance = max(stats["sumsq"] / count - mean * mean, 0.0)
    return {
        "activation_count": int(stats["count"]),
        "activation_mean": mean,
        "activation_std": float(np.sqrt(variance)),
        "activation_min": stats["min"] if stats["count"] else 0.0,
        "activation_max": stats["max"] if stats["count"] else 0.0,
        "activation_frac_lt_0_1": stats["lt_0_1"] / count,
        "activation_frac_lt_0_2": stats["lt_0_2"] / count,
        "activation_frac_gt_0_8": stats["gt_0_8"] / count,
    }


def diagnostic_variants(args: argparse.Namespace) -> list[dict[str, Any]]:
    return [
        {
            "variant_name": "cbm_no_cycl_v2",
            "model_type": "cbm_v2",
            "loss_mode": "cls+concept+align",
            "lambda_ii": 0.0,
            "lambda_it": 0.0,
        },
        {
            "variant_name": "cycl_v2_image_image_only",
            "model_type": "cycl_v2",
            "loss_mode": "cls+concept+align+image_image",
            "lambda_ii": args.lambda_ii,
            "lambda_it": 0.0,
        },
        {
            "variant_name": "cycl_v2_image_text_only",
            "model_type": "cycl_v2",
            "loss_mode": "cls+concept+align+image_text",
            "lambda_ii": 0.0,
            "lambda_it": args.lambda_it,
        },
        {
            "variant_name": "cycl_v2_both",
            "model_type": "cycl_v2",
            "loss_mode": "cls+concept+align+image_image+image_text",
            "lambda_ii": args.lambda_ii,
            "lambda_it": args.lambda_it,
        },
    ]


def train_one_diagnostic_variant(
    variant: dict[str, Any],
    args: argparse.Namespace,
    cache: Any,
    split_payloads: dict[str, Any],
    image_encoder: Any,
    preprocess_val: Any,
    device: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    set_reproducible_seed(args.seed)
    variant_args = argparse.Namespace(**vars(args))
    variant_args.lambda_ii = float(variant["lambda_ii"])
    variant_args.lambda_it = float(variant["lambda_it"])

    normalize_transform = get_normalize_transform(preprocess_val)
    view_transforms, view_names = build_v2_view_transforms(normalize_transform=normalize_transform, n_views=args.n_views)
    dataset = MultiViewDatasetV2(
        rows=cache.split_rows["train"],
        prototype_matrix=cache.prototype_matrix,
        view_transforms=view_transforms,
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"),
        generator=generator,
    )

    model = build_model_for_cache(variant_args, cache, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    text_embeddings = torch.tensor(cache.bank.merged_embeddings, dtype=torch.float32, device=device)
    concept_membership = torch.tensor(cache.bank.concept_class_membership, dtype=torch.float32, device=device)
    class_weights = compute_class_weights(cache, device)
    total_epochs = args.warmup_epochs + args.joint_epochs
    result_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    concept_rows: list[dict[str, Any]] = []

    for epoch in range(1, total_epochs + 1):
        phase = "warmup" if epoch <= args.warmup_epochs else "joint"
        model.train()
        running = {
            "total_loss": 0.0,
            "cls_loss": 0.0,
            "concept_loss": 0.0,
            "align_loss": 0.0,
            "image_image_loss": 0.0,
            "image_text_loss": 0.0,
        }
        pair_stats = {
            "image_image_positive_weight": init_scalar_stats(),
            "image_text_positive_weight": init_scalar_stats(),
        }
        concept_stats = init_concept_activation_stats()
        samples = 0
        batch_count = 0

        for batch_index, (views, labels, soft_targets, prototypes, _) in enumerate(loader):
            if args.max_train_batches and batch_index >= args.max_train_batches:
                break
            views = views.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            soft_targets = soft_targets.to(device, non_blocking=True)
            prototypes = prototypes.to(device, non_blocking=True)
            batch_size, n_views = views.shape[:2]
            flat_views = views.reshape(batch_size * n_views, *views.shape[2:])

            with torch.no_grad():
                features = image_encoder.encode_image(flat_views)
                features = F.normalize(features, dim=-1).float()

            losses, _, outputs = compute_supervised_v2_losses(
                model_type=str(variant["model_type"]),
                phase=phase,
                args=variant_args,
                model=model,
                features=features,
                labels=labels,
                soft_targets=soft_targets,
                prototypes=prototypes,
                text_embeddings=text_embeddings,
                concept_membership=concept_membership,
                class_weights=class_weights,
                n_views=n_views,
            )
            for name, value in losses.items():
                if not torch.isfinite(value):
                    raise RuntimeError(
                        f"Non-finite diagnostic loss: variant={variant['variant_name']} epoch={epoch} batch={batch_index} {name}={value}"
                    )
            optimizer.zero_grad(set_to_none=True)
            losses["total_loss"].backward()
            optimizer.step()

            update_pair_stats(pair_stats, labels, soft_targets, concept_membership, n_views=n_views)
            update_concept_activation_stats(concept_stats, outputs["concept_scores"])

            samples += batch_size
            batch_count += 1
            for name, value in losses.items():
                running[name] += float(value.detach().item()) * batch_size

        val_metrics = evaluate_v2_model(
            model=model,
            cache=cache,
            split_payloads=split_payloads,
            split="val",
            device=device,
            batch_size=args.batch_size * 4,
            class_weights=class_weights,
            lambda_align=args.lambda_align,
            max_batches=args.max_val_batches,
        )
        result_row: dict[str, Any] = {
            "variant_name": variant["variant_name"],
            "model_type": variant["model_type"],
            "loss_mode": variant["loss_mode"],
            "bank_name": cache.bank.name,
            "seed": args.seed,
            "epoch": epoch,
            "phase": phase,
            "n_views": args.n_views,
            "view_names": " ".join(view_names),
            "lambda_ii": variant_args.lambda_ii,
            "lambda_it": variant_args.lambda_it,
            "train_batches": batch_count,
            "train_samples": samples,
        }
        for name, value in running.items():
            result_row[name] = value / max(samples, 1)
        result_row.update(val_metrics)
        result_rows.append(result_row)

        pair_row: dict[str, Any] = {
            "variant_name": variant["variant_name"],
            "epoch": epoch,
            "phase": phase,
            "lambda_ii": variant_args.lambda_ii,
            "lambda_it": variant_args.lambda_it,
        }
        pair_row.update(finalize_scalar_stats(pair_stats["image_image_positive_weight"], "image_image_positive_weight"))
        pair_row.update(finalize_scalar_stats(pair_stats["image_text_positive_weight"], "image_text_positive_weight"))
        pair_rows.append(pair_row)

        concept_row = {
            "variant_name": variant["variant_name"],
            "epoch": epoch,
            "phase": phase,
            "lambda_ii": variant_args.lambda_ii,
            "lambda_it": variant_args.lambda_it,
            **finalize_concept_activation_stats(concept_stats),
        }
        concept_rows.append(concept_row)

        print(
            f"[diagnostic] {variant['variant_name']} epoch={epoch}/{total_epochs} phase={phase} "
            f"total={result_row['total_loss']:.4f} val_macro_f1={val_metrics['val_macro_f1']:.4f}",
            flush=True,
        )
    return result_rows, pair_rows, concept_rows


def plot_diagnostic_loss_curve(rows: list[dict[str, Any]], output_path: Path, title: str) -> None:
    ensure_dir(output_path.parent)
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    epochs = [int(row["epoch"]) for row in rows]
    for loss_name in ["total_loss", "cls_loss", "concept_loss", "align_loss", "image_image_loss", "image_text_loss"]:
        values = [float(row[loss_name]) for row in rows]
        ax.plot(epochs, values, marker="o", label=loss_name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train loss")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def answer_diagnostic_questions(final_rows: dict[str, dict[str, Any]]) -> list[str]:
    baseline = float(final_rows["cbm_no_cycl_v2"]["val_macro_f1"])
    ii = float(final_rows["cycl_v2_image_image_only"]["val_macro_f1"])
    it = float(final_rows["cycl_v2_image_text_only"]["val_macro_f1"])
    both = float(final_rows["cycl_v2_both"]["val_macro_f1"])

    def direction(value: float) -> str:
        delta = value - baseline
        if delta > 0.01:
            return f"helpful (+{delta:.4f} Macro-F1 vs CBM)"
        if delta < -0.01:
            return f"harmful ({delta:.4f} Macro-F1 vs CBM)"
        return f"roughly neutral ({delta:+.4f} Macro-F1 vs CBM)"

    if both < min(ii, it) - 0.01:
        combo = "yes, the combination is worse than either individual contrastive branch."
    elif both < max(ii, it) - 0.01:
        combo = "partly, the combination is worse than the better individual branch."
    else:
        combo = "no clear evidence that the combination is worse than the individual branches."

    if ii < baseline - 0.01 and it < baseline - 0.01:
        cause = "both contrastive branches are currently harmful under this short schedule; reduce weights further and delay contrastive start."
    elif ii < baseline - 0.01 and it >= baseline - 0.01:
        cause = "the image-image branch is the more likely source of degradation."
    elif it < baseline - 0.01 and ii >= baseline - 0.01:
        cause = "the image-text branch is the more likely source of degradation."
    elif both < baseline - 0.01:
        cause = "the branches may interfere when combined even if individual terms are not clearly harmful."
    else:
        cause = "the smaller weights appear stable; remaining differences need full-split training rather than debug-batch evidence."

    return [
        f"1. Image-image only is {direction(ii)}.",
        f"2. Image-text only is {direction(it)}.",
        f"3. Combination check: {combo}",
        f"4. Main likely issue from this diagnostic run: {cause}",
    ]


def write_diagnostic_summary(
    output_path: Path,
    args: argparse.Namespace,
    cache: Any,
    variants: list[dict[str, Any]],
    all_results: list[dict[str, Any]],
    command_line: str,
) -> None:
    final_rows = {str(row["variant_name"]): row for row in all_results if int(row["epoch"]) == args.warmup_epochs + args.joint_epochs}
    lines = [
        "# V2 Diagnostic Ablation Summary",
        "",
        "This stage is diagnostic only. It fixes the bank and augmentation, then isolates image-image and image-text contrastive branches.",
        "",
        "## Fixed Setup",
        "",
        f"- Bank: `{cache.bank.name}`",
        f"- Bank path: `{cache.bank.bank_dir}`",
        f"- Views: `{args.n_views}` using the existing V2 4-view default",
        f"- Warmup epochs: `{args.warmup_epochs}`",
        f"- Joint epochs: `{args.joint_epochs}`",
        f"- Max train batches per epoch: `{args.max_train_batches}`",
        f"- Seed: `{args.seed}`",
        f"- Contrastive weights for diagnostic terms: lambda_ii={args.lambda_ii}, lambda_it={args.lambda_it}",
        "",
        "Command:",
        "",
        f"```bash\n{command_line}\n```",
        "",
        "## Final Validation Metrics",
        "",
        "| Variant | Loss Mode | lambda_ii | lambda_it | Val Acc | Val Macro-F1 | Val Macro-AUROC | Train Total | Train II | Train IT |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in variants:
        row = final_rows[str(variant["variant_name"])]
        lines.append(
            f"| `{row['variant_name']}` | `{row['loss_mode']}` | "
            f"{float(row['lambda_ii']):.4f} | {float(row['lambda_it']):.4f} | "
            f"{float(row['val_accuracy']):.6f} | {float(row['val_macro_f1']):.6f} | "
            f"{float(row['val_macro_auroc']):.6f} | {float(row['total_loss']):.6f} | "
            f"{float(row['image_image_loss']):.6f} | {float(row['image_text_loss']):.6f} |"
        )
    lines.extend(["", "## Diagnostic Answers", ""])
    lines.extend([f"- {answer}" for answer in answer_diagnostic_questions(final_rows)])
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `diagnostic_ablation_results.csv`",
            "- `diagnostic_pair_stats.csv`",
            "- `diagnostic_concept_stats.csv`",
            "- `diagnostic_loss_curves/`",
            "",
            "Do not treat these values as formal conclusions because the training uses a limited number of batches per epoch.",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_diagnostic_ablation(args: argparse.Namespace) -> None:
    ensure_dir(args.output_dir)
    set_reproducible_seed(args.seed)
    load_biomedclip = import_biomedclip_loader()
    image_encoder, _, preprocess_val, resolved_device = load_biomedclip(args.device)
    device = resolve_device(resolved_device)
    image_encoder.to(device)
    image_encoder.eval()
    for parameter in image_encoder.parameters():
        parameter.requires_grad = False

    split_payloads = load_split_embeddings(args.embeddings_dir)
    bank = load_fixed_bank_v2(args.bank_dir)
    if bank.name != DEFAULT_BANK_DIR.name:
        print(f"[warning] diagnostic bank is {bank.name}; expected default {DEFAULT_BANK_DIR.name}", flush=True)
    cache = build_soft_targets_v2(bank, split_payloads)
    variants = diagnostic_variants(args)
    all_results: list[dict[str, Any]] = []
    all_pair_rows: list[dict[str, Any]] = []
    all_concept_rows: list[dict[str, Any]] = []
    curve_dir = args.output_dir / "diagnostic_loss_curves"
    ensure_dir(curve_dir)

    for variant in variants:
        result_rows, pair_rows, concept_rows = train_one_diagnostic_variant(
            variant=variant,
            args=args,
            cache=cache,
            split_payloads=split_payloads,
            image_encoder=image_encoder,
            preprocess_val=preprocess_val,
            device=device,
        )
        all_results.extend(result_rows)
        all_pair_rows.extend(pair_rows)
        all_concept_rows.extend(concept_rows)
        plot_diagnostic_loss_curve(
            result_rows,
            curve_dir / f"{variant['variant_name']}.png",
            f"{variant['variant_name']} diagnostic losses",
        )

    result_fields = [
        "variant_name",
        "model_type",
        "loss_mode",
        "bank_name",
        "seed",
        "epoch",
        "phase",
        "n_views",
        "view_names",
        "lambda_ii",
        "lambda_it",
        "train_batches",
        "train_samples",
        "total_loss",
        "cls_loss",
        "concept_loss",
        "align_loss",
        "image_image_loss",
        "image_text_loss",
        "val_accuracy",
        "val_macro_f1",
        "val_macro_auroc",
        "val_samples",
        "val_cls_loss",
        "val_concept_loss",
        "val_align_loss",
        "val_total_loss",
    ]
    pair_fields = [
        "variant_name",
        "epoch",
        "phase",
        "lambda_ii",
        "lambda_it",
        "image_image_positive_weight_count",
        "image_image_positive_weight_mean",
        "image_image_positive_weight_std",
        "image_image_positive_weight_min",
        "image_image_positive_weight_max",
        "image_text_positive_weight_count",
        "image_text_positive_weight_mean",
        "image_text_positive_weight_std",
        "image_text_positive_weight_min",
        "image_text_positive_weight_max",
    ]
    concept_fields = [
        "variant_name",
        "epoch",
        "phase",
        "lambda_ii",
        "lambda_it",
        "activation_count",
        "activation_mean",
        "activation_std",
        "activation_min",
        "activation_max",
        "activation_frac_lt_0_1",
        "activation_frac_lt_0_2",
        "activation_frac_gt_0_8",
    ]
    write_csv(args.output_dir / "diagnostic_ablation_results.csv", result_fields, all_results)
    write_csv(args.output_dir / "diagnostic_pair_stats.csv", pair_fields, all_pair_rows)
    write_csv(args.output_dir / "diagnostic_concept_stats.csv", concept_fields, all_concept_rows)
    write_diagnostic_summary(
        output_path=args.output_dir / "diagnostic_summary.md",
        args=args,
        cache=cache,
        variants=variants,
        all_results=all_results,
        command_line="python " + " ".join(sys.argv),
    )
    print(f"[done] V2 diagnostic ablation written to {args.output_dir}", flush=True)


def formal_comparison_settings(args: argparse.Namespace) -> list[dict[str, Any]]:
    settings = [
        {
            "stage": "primary",
            "bank_dir": args.primary_bank_dir,
            "setting_name": "primary_cbm_no_cycl_v2",
            "model_type": "cbm_v2",
            "loss_mode": "cls+concept+align",
            "lambda_ii": 0.0,
            "lambda_it": 0.0,
        },
        {
            "stage": "primary",
            "bank_dir": args.primary_bank_dir,
            "setting_name": "primary_cycl_it_lambda0_02",
            "model_type": "cycl_v2",
            "loss_mode": "cls+concept+align+image_text",
            "lambda_ii": 0.0,
            "lambda_it": 0.02,
        },
    ]
    if not getattr(args, "main_four_groups_only", False):
        settings.append(
            {
                "stage": "primary",
                "bank_dir": args.primary_bank_dir,
                "setting_name": "primary_cycl_it_lambda0_05",
                "model_type": "cycl_v2",
                "loss_mode": "cls+concept+align+image_text",
                "lambda_ii": 0.0,
                "lambda_it": 0.05,
            }
        )
    if args.run_secondary:
        settings.extend(
            [
                {
                    "stage": "secondary",
                    "bank_dir": args.secondary_bank_dir,
                    "setting_name": "secondary_cbm_no_cycl_v2",
                    "model_type": "cbm_v2",
                    "loss_mode": "cls+concept+align",
                    "lambda_ii": 0.0,
                    "lambda_it": 0.0,
                },
                {
                    "stage": "secondary",
                    "bank_dir": args.secondary_bank_dir,
                    "setting_name": "secondary_cycl_it_lambda0_02",
                    "model_type": "cycl_v2",
                    "loss_mode": "cls+concept+align+image_text",
                    "lambda_ii": 0.0,
                    "lambda_it": 0.02,
                },
            ]
        )
    return settings


def plot_formal_curves(train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]], output_path: Path, title: str) -> None:
    ensure_dir(output_path.parent)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    train_epochs = [int(row["epoch"]) for row in train_rows]
    for loss_name in ["total_loss", "cls_loss", "concept_loss", "align_loss", "image_image_loss", "image_text_loss"]:
        values = [float(row[loss_name]) for row in train_rows]
        axes[0].plot(train_epochs, values, marker="o", label=loss_name)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Train loss")
    axes[0].set_title("Train losses")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=7)

    val_epochs = [int(row["epoch"]) for row in val_rows]
    for metric_name in ["val_accuracy", "val_macro_f1", "val_macro_auroc"]:
        values = [float(row[metric_name]) for row in val_rows]
        axes[1].plot(val_epochs, values, marker="o", label=metric_name)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation metric")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Validation metrics")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def train_one_formal_setting(
    setting: dict[str, Any],
    args: argparse.Namespace,
    cache: Any,
    split_payloads: dict[str, Any],
    image_encoder: Any,
    preprocess_val: Any,
    device: str,
    seed: int,
    curve_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    set_reproducible_seed(seed)
    setting_args = argparse.Namespace(**vars(args))
    setting_args.lambda_ii = float(setting["lambda_ii"])
    setting_args.lambda_it = float(setting["lambda_it"])

    normalize_transform = get_normalize_transform(preprocess_val)
    view_transforms, view_names = build_v2_view_transforms(normalize_transform=normalize_transform, n_views=args.n_views)
    dataset = MultiViewDatasetV2(
        rows=cache.split_rows["train"],
        prototype_matrix=cache.prototype_matrix,
        view_transforms=view_transforms,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"),
        generator=generator,
    )

    model = build_model_for_cache(setting_args, cache, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    text_embeddings = torch.tensor(cache.bank.merged_embeddings, dtype=torch.float32, device=device)
    concept_membership = torch.tensor(cache.bank.concept_class_membership, dtype=torch.float32, device=device)
    class_weights = compute_class_weights(cache, device)
    total_epochs = args.warmup_epochs + args.joint_epochs
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    best_score = float("-inf")
    best_epoch = -1
    best_state: dict[str, Any] | None = None

    for epoch in range(1, total_epochs + 1):
        phase = "warmup" if epoch <= args.warmup_epochs else "joint"
        model.train()
        running = {
            "total_loss": 0.0,
            "cls_loss": 0.0,
            "concept_loss": 0.0,
            "align_loss": 0.0,
            "image_image_loss": 0.0,
            "image_text_loss": 0.0,
        }
        samples = 0
        batch_count = 0

        for batch_index, (views, labels, soft_targets, prototypes, _) in enumerate(loader):
            if args.max_train_batches and batch_index >= args.max_train_batches:
                break
            views = views.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            soft_targets = soft_targets.to(device, non_blocking=True)
            prototypes = prototypes.to(device, non_blocking=True)
            batch_size, n_views = views.shape[:2]
            flat_views = views.reshape(batch_size * n_views, *views.shape[2:])

            with torch.no_grad():
                features = image_encoder.encode_image(flat_views)
                features = F.normalize(features, dim=-1).float()

            losses, _, _ = compute_supervised_v2_losses(
                model_type=str(setting["model_type"]),
                phase=phase,
                args=setting_args,
                model=model,
                features=features,
                labels=labels,
                soft_targets=soft_targets,
                prototypes=prototypes,
                text_embeddings=text_embeddings,
                concept_membership=concept_membership,
                class_weights=class_weights,
                n_views=n_views,
            )
            for name, value in losses.items():
                if not torch.isfinite(value):
                    raise RuntimeError(
                        f"Non-finite formal loss: setting={setting['setting_name']} seed={seed} epoch={epoch} batch={batch_index} {name}={value}"
                    )
            optimizer.zero_grad(set_to_none=True)
            losses["total_loss"].backward()
            optimizer.step()

            samples += batch_size
            batch_count += 1
            for name, value in losses.items():
                running[name] += float(value.detach().item()) * batch_size

        train_row: dict[str, Any] = {
            "setting_name": setting["setting_name"],
            "stage": setting["stage"],
            "bank_name": cache.bank.name,
            "model_type": setting["model_type"],
            "loss_mode": setting["loss_mode"],
            "seed": seed,
            "epoch": epoch,
            "phase": phase,
            "n_views": args.n_views,
            "view_names": " ".join(view_names),
            "lambda_ii": setting_args.lambda_ii,
            "lambda_it": setting_args.lambda_it,
            "train_batches": batch_count,
            "train_samples": samples,
        }
        for name, value in running.items():
            train_row[name] = value / max(samples, 1)
        train_rows.append(train_row)

        val_metrics = evaluate_v2_model(
            model=model,
            cache=cache,
            split_payloads=split_payloads,
            split="val",
            device=device,
            batch_size=args.batch_size * 4,
            class_weights=class_weights,
            lambda_align=args.lambda_align,
            max_batches=args.max_val_batches,
        )
        val_row = {
            "setting_name": setting["setting_name"],
            "stage": setting["stage"],
            "bank_name": cache.bank.name,
            "model_type": setting["model_type"],
            "loss_mode": setting["loss_mode"],
            "seed": seed,
            "epoch": epoch,
            "phase": phase,
            "lambda_ii": setting_args.lambda_ii,
            "lambda_it": setting_args.lambda_it,
            **val_metrics,
        }
        val_rows.append(val_row)
        score = float(val_metrics["val_macro_f1"])
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        print(
            f"[formal] {setting['setting_name']} seed={seed} epoch={epoch}/{total_epochs} phase={phase} "
            f"train_total={train_row['total_loss']:.4f} val_macro_f1={score:.4f} best_epoch={best_epoch}",
            flush=True,
        )

    if best_state is None:
        raise RuntimeError(f"No checkpoint selected for setting={setting['setting_name']} seed={seed}")
    model.load_state_dict(best_state)
    ensure_dir(curve_dir / "checkpoints")
    run_id = f"{setting['setting_name']}_seed{seed}"
    checkpoint_path = curve_dir / "checkpoints" / f"{run_id}_best.pt"
    torch.save(
        {
            "model_state_dict": best_state,
            "setting": setting,
            "seed": seed,
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_score,
            "bank_name": cache.bank.name,
            "bank_dir": str(cache.bank.bank_dir),
        },
        checkpoint_path,
    )

    best_val = evaluate_v2_model(
        model=model,
        cache=cache,
        split_payloads=split_payloads,
        split="val",
        device=device,
        batch_size=args.batch_size * 4,
        class_weights=class_weights,
        lambda_align=args.lambda_align,
        max_batches=0,
    )
    test_metrics = evaluate_v2_model(
        model=model,
        cache=cache,
        split_payloads=split_payloads,
        split="test",
        device=device,
        batch_size=args.batch_size * 4,
        class_weights=class_weights,
        lambda_align=args.lambda_align,
        max_batches=0,
    )
    train_fields = list(train_rows[0].keys())
    val_fields = list(val_rows[0].keys())
    write_csv(curve_dir / f"{run_id}_train_curve.csv", train_fields, train_rows)
    write_csv(curve_dir / f"{run_id}_val_curve.csv", val_fields, val_rows)
    plot_formal_curves(train_rows, val_rows, curve_dir / f"{run_id}.png", f"{run_id} formal curves")

    final_row: dict[str, Any] = {
        "setting_name": setting["setting_name"],
        "stage": setting["stage"],
        "bank_name": cache.bank.name,
        "bank_dir": str(cache.bank.bank_dir),
        "model_type": setting["model_type"],
        "loss_mode": setting["loss_mode"],
        "seed": seed,
        "n_views": args.n_views,
        "warmup_epochs": args.warmup_epochs,
        "joint_epochs": args.joint_epochs,
        "max_train_batches": args.max_train_batches,
        "lambda_ii": setting_args.lambda_ii,
        "lambda_it": setting_args.lambda_it,
        "best_epoch": best_epoch,
        "best_val_macro_f1_selected": best_score,
        "checkpoint_path": str(checkpoint_path),
        "train_curve_csv": str(curve_dir / f"{run_id}_train_curve.csv"),
        "val_curve_csv": str(curve_dir / f"{run_id}_val_curve.csv"),
        "curve_png": str(curve_dir / f"{run_id}.png"),
    }
    final_row.update(best_val)
    final_row.update(test_metrics)
    return final_row, train_rows, val_rows


def formal_result_fields() -> list[str]:
    fields = [
        "setting_name",
        "stage",
        "bank_name",
        "bank_dir",
        "model_type",
        "loss_mode",
        "seed",
        "n_views",
        "warmup_epochs",
        "joint_epochs",
        "max_train_batches",
        "lambda_ii",
        "lambda_it",
        "best_epoch",
        "best_val_macro_f1_selected",
        "val_accuracy",
        "val_macro_f1",
        "val_macro_auroc",
        "test_accuracy",
        "test_macro_f1",
        "test_macro_auroc",
        "val_samples",
        "test_samples",
    ]
    for split in ["val", "test"]:
        fields.extend([f"{split}_recall_{class_name}" for class_name in LABEL_CODES])
    fields.extend(
        [
            "checkpoint_path",
            "train_curve_csv",
            "val_curve_csv",
            "curve_png",
        ]
    )
    return fields


def summarize_formal_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["setting_name"]), []).append(row)
    summary_rows: list[dict[str, Any]] = []
    metric_names = [
        "val_accuracy",
        "val_macro_f1",
        "val_macro_auroc",
        "test_accuracy",
        "test_macro_f1",
        "test_macro_auroc",
    ]
    for setting_name, group in grouped.items():
        item: dict[str, Any] = {
            "setting_name": setting_name,
            "stage": group[0]["stage"],
            "bank_name": group[0]["bank_name"],
            "model_type": group[0]["model_type"],
            "loss_mode": group[0]["loss_mode"],
            "lambda_it": group[0]["lambda_it"],
            "n_seeds": len(group),
        }
        for metric_name in metric_names:
            values = np.array([float(row[metric_name]) for row in group], dtype=np.float32)
            item[f"{metric_name}_mean"] = float(values.mean())
            item[f"{metric_name}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary_rows.append(item)
    return summary_rows


def write_formal_summary(
    output_path: Path,
    args: argparse.Namespace,
    primary_rows: list[dict[str, Any]],
    secondary_rows: list[dict[str, Any]],
    command_line: str,
) -> None:
    all_rows = [*primary_rows, *secondary_rows]
    summaries = summarize_formal_rows(all_rows)
    by_setting = {str(row["setting_name"]): row for row in summaries}
    primary_cbm = by_setting.get("primary_cbm_no_cycl_v2")
    primary_002 = by_setting.get("primary_cycl_it_lambda0_02")
    primary_005 = by_setting.get("primary_cycl_it_lambda0_05")
    secondary_cbm = by_setting.get("secondary_cbm_no_cycl_v2")
    secondary_002 = by_setting.get("secondary_cycl_it_lambda0_02")

    def metric(row: dict[str, Any] | None, name: str) -> float:
        if row is None:
            return float("nan")
        return float(row[f"{name}_mean"])

    best = max(summaries, key=lambda row: float(row["test_macro_f1_mean"])) if summaries else None
    lines = [
        "# First Formal V2 Image-Text-Only Comparison",
        "",
        "This stage compares the clean CBM-no-CyCL V2 baseline against image-text-only CyCL V2. It does not run filtered_top300, augmentation sweeps, image-image-only, or both-branch settings.",
        "",
        "## Configuration",
        "",
        f"- Primary bank: `{args.primary_bank_dir}`",
        f"- Secondary bank: `{args.secondary_bank_dir}`",
        f"- Seeds: `{args.seed}`",
        f"- Views: `{args.n_views}`",
        f"- Warmup epochs: `{args.warmup_epochs}`",
        f"- Joint epochs: `{args.joint_epochs}`",
        f"- Max train batches: `{args.max_train_batches}` (`0` means full train split)",
        f"- Batch size: `{args.batch_size}`",
        f"- LR / weight decay: `{args.lr}` / `{args.weight_decay}`",
        f"- Model selection: best checkpoint by validation Macro-F1; test evaluated once from that checkpoint.",
        "",
        "Command:",
        "",
        f"```bash\n{command_line}\n```",
        "",
        "## Result Summary",
        "",
        "| Setting | Bank | Model | lambda_it | Seeds | Val Macro-F1 | Test Macro-F1 | Test Acc | Test AUROC |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| `{row['setting_name']}` | `{row['bank_name']}` | `{row['model_type']}` | "
            f"{float(row['lambda_it']):.4f} | {int(row['n_seeds'])} | "
            f"{float(row['val_macro_f1_mean']):.6f} ± {float(row['val_macro_f1_std']):.6f} | "
            f"{float(row['test_macro_f1_mean']):.6f} ± {float(row['test_macro_f1_std']):.6f} | "
            f"{float(row['test_accuracy_mean']):.6f} ± {float(row['test_accuracy_std']):.6f} | "
            f"{float(row['test_macro_auroc_mean']):.6f} ± {float(row['test_macro_auroc_std']):.6f} |"
        )
    lines.extend(["", "## Answers", ""])
    if primary_cbm and primary_002 and primary_005:
        best_primary_cycl = primary_002 if metric(primary_002, "test_macro_f1") >= metric(primary_005, "test_macro_f1") else primary_005
        delta = metric(best_primary_cycl, "test_macro_f1") - metric(primary_cbm, "test_macro_f1")
        lines.append(
            f"1. Image-text-only CyCL V2 vs clean CBM on the primary bank: best CyCL setting is `{best_primary_cycl['setting_name']}`, "
            f"test Macro-F1 delta={delta:+.6f}."
        )
        lambda_choice = "0.02" if metric(primary_002, "test_macro_f1") >= metric(primary_005, "test_macro_f1") else "0.05"
        lines.append(f"3. Better lambda_img_txt on the primary bank by test Macro-F1: `{lambda_choice}`.")
    else:
        lines.append("1. Primary bank comparison is incomplete.")
        lines.append("3. Lambda comparison is incomplete.")

    if primary_cbm and secondary_cbm and primary_002 and secondary_002:
        primary_best = max(metric(primary_cbm, "test_macro_f1"), metric(primary_002, "test_macro_f1"), metric(primary_005, "test_macro_f1") if primary_005 else float("-inf"))
        secondary_best = max(metric(secondary_cbm, "test_macro_f1"), metric(secondary_002, "test_macro_f1"))
        better_bank = "retrieval_top10_per_class_disc" if primary_best >= secondary_best else "retrieval_top10_per_class"
        lines.append(f"2. Better bank under this V2 setup by best test Macro-F1: `{better_bank}`.")
    else:
        lines.append("2. Bank comparison is incomplete.")

    if best:
        lines.append(f"4. Current best run summary: `{best['setting_name']}` with test Macro-F1={float(best['test_macro_f1_mean']):.6f}.")
    if len(args.seed) < 3:
        lines.append("Evidence strength: this run used fewer than 3 seeds, so multi-seed confirmation remains pending before final claims.")
    else:
        lines.append("Evidence strength: this run includes at least 3 seeds; inspect std before making final claims.")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_formal_it_compare(args: argparse.Namespace) -> None:
    ensure_dir(args.output_dir)
    if args.seed is None:
        args.seed = [42]
    else:
        args.seed = list(dict.fromkeys(int(seed) for seed in args.seed))
    curve_dir = args.output_dir / "formal_curves"
    ensure_dir(curve_dir)
    load_biomedclip = import_biomedclip_loader()
    image_encoder, _, preprocess_val, resolved_device = load_biomedclip(args.device)
    device = resolve_device(resolved_device)
    image_encoder.to(device)
    image_encoder.eval()
    for parameter in image_encoder.parameters():
        parameter.requires_grad = False

    split_payloads = load_split_embeddings(args.embeddings_dir)
    settings = formal_comparison_settings(args)
    cache_by_bank: dict[str, Any] = {}
    primary_rows: list[dict[str, Any]] = []
    secondary_rows: list[dict[str, Any]] = []

    for setting in settings:
        bank_path = Path(setting["bank_dir"]).resolve()
        bank_key = str(bank_path)
        if bank_key not in cache_by_bank:
            bank = load_fixed_bank_v2(bank_path)
            cache_by_bank[bank_key] = build_soft_targets_v2(bank, split_payloads)
        cache = cache_by_bank[bank_key]
        for seed in args.seed:
            final_row, _, _ = train_one_formal_setting(
                setting=setting,
                args=args,
                cache=cache,
                split_payloads=split_payloads,
                image_encoder=image_encoder,
                preprocess_val=preprocess_val,
                device=device,
                seed=int(seed),
                curve_dir=curve_dir,
            )
            if setting["stage"] == "primary":
                primary_rows.append(final_row)
            else:
                secondary_rows.append(final_row)

    fields = formal_result_fields()
    write_csv(args.output_dir / "formal_results_primary_bank.csv", fields, primary_rows)
    write_csv(args.output_dir / "formal_results_secondary_bank.csv", fields, secondary_rows)
    summary_fields = [
        "setting_name",
        "stage",
        "bank_name",
        "model_type",
        "loss_mode",
        "lambda_it",
        "n_seeds",
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
    write_csv(args.output_dir / "formal_results_summary.csv", summary_fields, summarize_formal_rows([*primary_rows, *secondary_rows]))
    write_formal_summary(
        output_path=args.output_dir / "formal_summary.md",
        args=args,
        primary_rows=primary_rows,
        secondary_rows=secondary_rows,
        command_line="python " + " ".join(sys.argv),
    )
    print(f"[done] formal image-text-only comparison written to {args.output_dir}", flush=True)


def supplement_settings(args: argparse.Namespace) -> list[dict[str, Any]]:
    settings: list[dict[str, Any]] = []
    if args.experiment == "lambda_sweep":
        lambda_values = args.lambda_it_value or [0.01, 0.02, 0.03]
        for value in lambda_values:
            label = str(value).replace(".", "_")
            settings.append(
                {
                    "stage": "supplement_lambda_sweep",
                    "bank_dir": args.bank_dir,
                    "setting_name": f"main_cycl_it_lambda{label}",
                    "model_type": "cycl_v2",
                    "loss_mode": "cls+concept+align+image_text",
                    "lambda_ii": 0.0,
                    "lambda_it": float(value),
                    "n_views": int(args.n_views),
                }
            )
    else:
        view_counts = args.view_count or [2, 4, 5]
        for view_count in view_counts:
            settings.append(
                {
                    "stage": "supplement_view_sweep",
                    "bank_dir": args.bank_dir,
                    "setting_name": f"main_cycl_it_views{view_count}",
                    "model_type": "cycl_v2",
                    "loss_mode": "cls+concept+align+image_text",
                    "lambda_ii": 0.0,
                    "lambda_it": 0.02,
                    "n_views": int(view_count),
                }
            )
    return settings


def summarize_supplement_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["setting_name"]), []).append(row)
    metric_names = [
        "val_accuracy",
        "val_macro_f1",
        "val_macro_auroc",
        "test_accuracy",
        "test_macro_f1",
        "test_macro_auroc",
    ]
    summary_rows: list[dict[str, Any]] = []
    for setting_name, group in grouped.items():
        item: dict[str, Any] = {
            "setting_name": setting_name,
            "stage": group[0]["stage"],
            "bank_name": group[0]["bank_name"],
            "model_type": group[0]["model_type"],
            "loss_mode": group[0]["loss_mode"],
            "lambda_it": group[0]["lambda_it"],
            "n_views": group[0]["n_views"],
            "n_seeds": len(group),
        }
        for metric_name in metric_names:
            values = np.array([float(row[metric_name]) for row in group], dtype=np.float32)
            item[f"{metric_name}_mean"] = float(values.mean())
            item[f"{metric_name}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary_rows.append(item)
    return sorted(summary_rows, key=lambda row: float(row["test_macro_f1_mean"]), reverse=True)


def supplement_summary_fields() -> list[str]:
    return [
        "setting_name",
        "stage",
        "bank_name",
        "model_type",
        "loss_mode",
        "lambda_it",
        "n_views",
        "n_seeds",
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


def write_supplement_summary(
    output_path: Path,
    args: argparse.Namespace,
    summary_rows: list[dict[str, Any]],
    command_line: str,
) -> None:
    best = summary_rows[0] if summary_rows else None
    lines = [
        "# Supplementary Main-Model Ablation Summary",
        "",
        "This supplementary stage fixes the main bank and main model, then varies only one factor.",
        "",
        "## Fixed Setup",
        "",
        f"- Experiment: `{args.experiment}`",
        f"- Bank: `{args.bank_dir}`",
        "- Model: `CyCL V2 image-text only`",
        f"- Seeds: `{args.seed}`",
        f"- Warmup epochs: `{args.warmup_epochs}`",
        f"- Joint epochs: `{args.joint_epochs}`",
        f"- Max train batches: `{args.max_train_batches}` (`0` means full train split)",
        f"- Batch size: `{args.batch_size}`",
        "",
        "Command:",
        "",
        f"```bash\n{command_line}\n```",
        "",
        "## Results",
        "",
        "| Setting | lambda_it | Views | Seeds | Val Macro-F1 | Test Macro-F1 | Test Acc | Test AUROC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| `{row['setting_name']}` | {float(row['lambda_it']):.4f} | {int(row['n_views'])} | {int(row['n_seeds'])} | "
            f"{float(row['val_macro_f1_mean']):.6f} ± {float(row['val_macro_f1_std']):.6f} | "
            f"{float(row['test_macro_f1_mean']):.6f} ± {float(row['test_macro_f1_std']):.6f} | "
            f"{float(row['test_accuracy_mean']):.6f} ± {float(row['test_accuracy_std']):.6f} | "
            f"{float(row['test_macro_auroc_mean']):.6f} ± {float(row['test_macro_auroc_std']):.6f} |"
        )
    if best:
        lines.extend(
            [
                "",
                "## Best Setting",
                "",
                f"Best by mean test Macro-F1: `{best['setting_name']}` with `{float(best['test_macro_f1_mean']):.6f} ± {float(best['test_macro_f1_std']):.6f}`.",
            ]
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_supplement_main_ablation(args: argparse.Namespace) -> None:
    ensure_dir(args.output_dir)
    if args.seed is None:
        args.seed = [42, 43, 44]
    else:
        args.seed = list(dict.fromkeys(int(seed) for seed in args.seed))
    curve_dir = args.output_dir / "supplement_curves"
    ensure_dir(curve_dir)

    load_biomedclip = import_biomedclip_loader()
    image_encoder, _, preprocess_val, resolved_device = load_biomedclip(args.device)
    device = resolve_device(resolved_device)
    image_encoder.to(device)
    image_encoder.eval()
    for parameter in image_encoder.parameters():
        parameter.requires_grad = False

    split_payloads = load_split_embeddings(args.embeddings_dir)
    bank = load_fixed_bank_v2(args.bank_dir)
    cache = build_soft_targets_v2(bank, split_payloads)
    all_rows: list[dict[str, Any]] = []

    for setting in supplement_settings(args):
        for seed in args.seed:
            setting_args = argparse.Namespace(**vars(args))
            setting_args.n_views = int(setting["n_views"])
            final_row, _, _ = train_one_formal_setting(
                setting=setting,
                args=setting_args,
                cache=cache,
                split_payloads=split_payloads,
                image_encoder=image_encoder,
                preprocess_val=preprocess_val,
                device=device,
                seed=int(seed),
                curve_dir=curve_dir,
            )
            all_rows.append(final_row)

    fields = formal_result_fields()
    write_csv(args.output_dir / "supplement_results.csv", fields, all_rows)
    summary_rows = summarize_supplement_rows(all_rows)
    write_csv(args.output_dir / "supplement_results_summary.csv", supplement_summary_fields(), summary_rows)
    write_supplement_summary(
        output_path=args.output_dir / "supplement_summary.md",
        args=args,
        summary_rows=summary_rows,
        command_line="python " + " ".join(sys.argv),
    )
    print(f"[done] supplement {args.experiment} written to {args.output_dir}", flush=True)


def load_v2_checkpoint_model(
    checkpoint_path: Path,
    cache: Any,
    device: str,
    dropout: float,
    proj_dim: int,
) -> CBMCyCLV2:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = CBMCyCLV2(
        embedding_dim=int(cache.bank.merged_embeddings.shape[1]),
        num_concepts=len(cache.bank.merged_concepts),
        num_classes=len(LABEL_CODES),
        dropout=dropout,
        proj_dim=proj_dim,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def predict_split_with_model(
    model: CBMCyCLV2,
    cache: Any,
    split_payloads: dict[str, Any],
    split: str,
    device: str,
    batch_size: int = 256,
) -> dict[str, Any]:
    embeddings = split_payloads[split].embeddings.astype(np.float32)
    rows = cache.split_rows[split]
    labels = np.array([int(row["label_index"]) for row in rows], dtype=np.int64)
    probabilities: list[np.ndarray] = []
    concept_scores: list[np.ndarray] = []
    logits_all: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, embeddings.shape[0], batch_size):
            end = min(start + batch_size, embeddings.shape[0])
            features = torch.tensor(embeddings[start:end], dtype=torch.float32, device=device)
            outputs = model.forward_image_features(features)
            probabilities.append(torch.softmax(outputs["logits"], dim=1).cpu().numpy())
            concept_scores.append(outputs["concept_scores"].cpu().numpy())
            logits_all.append(outputs["logits"].cpu().numpy())
    probs = np.concatenate(probabilities, axis=0)
    concepts = np.concatenate(concept_scores, axis=0)
    logits = np.concatenate(logits_all, axis=0)
    return {
        "rows": rows,
        "labels": labels,
        "probabilities": probs,
        "predictions": probs.argmax(axis=1),
        "concept_scores": concepts,
        "logits": logits,
    }


def per_class_metric_rows(
    labels: np.ndarray,
    predictions: np.ndarray,
    setting_name: str,
    seed: int,
    split: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for class_index, class_name in enumerate(LABEL_CODES):
        true_positive = float(((labels == class_index) & (predictions == class_index)).sum())
        false_positive = float(((labels != class_index) & (predictions == class_index)).sum())
        false_negative = float(((labels == class_index) & (predictions != class_index)).sum())
        support = int((labels == class_index).sum())
        precision = true_positive / max(true_positive + false_positive, 1.0)
        recall = true_positive / max(true_positive + false_negative, 1.0)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
        rows.append(
            {
                "setting_name": setting_name,
                "seed": seed,
                "split": split,
                "class_name": class_name,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
                "tp": int(true_positive),
                "fp": int(false_positive),
                "fn": int(false_negative),
            }
        )
    return rows


def summarize_per_class_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["setting_name"]), str(row["split"]), str(row["class_name"]))
        grouped.setdefault(key, []).append(row)
    summary: list[dict[str, Any]] = []
    for (setting_name, split, class_name), group in sorted(grouped.items()):
        item: dict[str, Any] = {
            "setting_name": setting_name,
            "split": split,
            "class_name": class_name,
            "n_seeds": len(group),
            "support": group[0]["support"],
        }
        for metric_name in ["precision", "recall", "f1"]:
            values = np.array([float(row[metric_name]) for row in group], dtype=np.float32)
            item[f"{metric_name}_mean"] = float(values.mean())
            item[f"{metric_name}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary.append(item)
    return summary


def compute_confusion_matrix(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    matrix = np.zeros((len(LABEL_CODES), len(LABEL_CODES)), dtype=np.int64)
    for true_label, predicted_label in zip(labels, predictions):
        matrix[int(true_label), int(predicted_label)] += 1
    return matrix


def confusion_matrix_rows(matrix: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_index, class_name in enumerate(LABEL_CODES):
        item: dict[str, Any] = {"true_class": class_name}
        for col_index, pred_name in enumerate(LABEL_CODES):
            item[pred_name] = int(matrix[row_index, col_index])
        item["support"] = int(matrix[row_index].sum())
        rows.append(item)
    return rows


def plot_confusion_matrix_heatmap(matrix: np.ndarray, output_path: Path, title: str) -> None:
    ensure_dir(output_path.parent)
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(LABEL_CODES)))
    ax.set_yticks(range(len(LABEL_CODES)))
    ax.set_xticklabels(LABEL_CODES)
    ax.set_yticklabels(LABEL_CODES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    threshold = matrix.max() * 0.5 if matrix.size else 0.0
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            color = "white" if matrix[row_index, col_index] > threshold else "black"
            ax.text(col_index, row_index, str(int(matrix[row_index, col_index])), ha="center", va="center", color=color)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def concept_classes_by_index(cache: Any) -> list[str]:
    membership = cache.bank.concept_class_membership
    return [LABEL_CODES[int(np.argmax(membership[index]))] for index in range(membership.shape[0])]


def select_explanation_indices(labels: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray, num_samples: int) -> list[int]:
    confidence = probabilities.max(axis=1)
    correct = np.where(labels == predictions)[0]
    incorrect = np.where(labels != predictions)[0]
    selected: list[int] = []
    if incorrect.size:
        incorrect_sorted = incorrect[np.argsort(-confidence[incorrect])]
        selected.extend(int(index) for index in incorrect_sorted[: max(num_samples // 2, 1)])
    if correct.size and len(selected) < num_samples:
        correct_sorted = correct[np.argsort(-confidence[correct])]
        selected.extend(int(index) for index in correct_sorted[: num_samples - len(selected)])
    if len(selected) < num_samples:
        all_sorted = np.argsort(-confidence)
        for index in all_sorted:
            if int(index) not in selected:
                selected.append(int(index))
            if len(selected) >= num_samples:
                break
    return selected[:num_samples]


def draw_explanation_panel(
    image_path: str,
    lines: list[str],
    output_path: Path,
) -> None:
    ensure_dir(output_path.parent)
    image = Image.open(image_path).convert("RGB")
    image.thumbnail((420, 420))
    font = ImageFont.load_default()
    wrapped: list[str] = []
    for line in lines:
        wrapped.extend(textwrap.wrap(line, width=72) or [""])
    line_height = 14
    text_width = 620
    panel_height = max(image.height, 24 + line_height * len(wrapped))
    canvas = Image.new("RGB", (image.width + text_width + 32, panel_height + 24), "white")
    canvas.paste(image, (12, 12))
    draw = ImageDraw.Draw(canvas)
    x_text = image.width + 24
    y_text = 12
    for line in wrapped:
        draw.text((x_text, y_text), line, fill="black", font=font)
        y_text += line_height
    canvas.save(output_path)


def write_explanation_samples(
    model: CBMCyCLV2,
    cache: Any,
    prediction_payload: dict[str, Any],
    setting_name: str,
    seed: int,
    output_dir: Path,
    num_samples: int,
    top_k: int,
) -> list[dict[str, Any]]:
    ensure_dir(output_dir)
    rows = prediction_payload["rows"]
    labels = prediction_payload["labels"]
    predictions = prediction_payload["predictions"]
    probabilities = prediction_payload["probabilities"]
    concept_scores = prediction_payload["concept_scores"]
    class_names = concept_classes_by_index(cache)
    classifier_weight = model.classifier.weight.detach().cpu().numpy()
    classifier_bias = model.classifier.bias.detach().cpu().numpy()
    selected_indices = select_explanation_indices(labels, predictions, probabilities, num_samples)
    explanation_rows: list[dict[str, Any]] = []
    sample_index_rows: list[dict[str, Any]] = []

    for sample_rank, sample_index in enumerate(selected_indices, start=1):
        row = rows[sample_index]
        true_index = int(labels[sample_index])
        pred_index = int(predictions[sample_index])
        pred_prob = float(probabilities[sample_index, pred_index])
        activations = concept_scores[sample_index]
        top_indices = np.argsort(-activations)[:top_k]
        lines = [
            f"Sample {sample_rank} | setting={setting_name} | seed={seed}",
            f"True: {LABEL_CODES[true_index]} | Pred: {LABEL_CODES[pred_index]} | Prob: {pred_prob:.4f}",
            f"Pred logit bias: {classifier_bias[pred_index]:.4f}",
            "",
            "Top activated concepts and contribution to predicted logit:",
        ]
        for concept_rank, concept_index in enumerate(top_indices, start=1):
            activation = float(activations[concept_index])
            weight = float(classifier_weight[pred_index, concept_index])
            contribution = activation * weight
            concept = cache.bank.merged_concepts[int(concept_index)]
            concept_class = class_names[int(concept_index)]
            lines.append(
                f"{concept_rank}. [{concept_class}] act={activation:.4f} weight={weight:.4f} contrib={contribution:.4f} | {concept}"
            )
            explanation_rows.append(
                {
                    "setting_name": setting_name,
                    "seed": seed,
                    "sample_rank": sample_rank,
                    "sample_index": sample_index,
                    "image_path": row["image_path"],
                    "true_class": LABEL_CODES[true_index],
                    "predicted_class": LABEL_CODES[pred_index],
                    "predicted_probability": pred_prob,
                    "concept_rank": concept_rank,
                    "concept_index": int(concept_index),
                    "concept_class": concept_class,
                    "concept": concept,
                    "activation": activation,
                    "pred_class_weight": weight,
                    "contribution_to_pred_logit": contribution,
                }
            )
        panel_path = output_dir / f"sample_{sample_rank:02d}_{LABEL_CODES[true_index]}_pred_{LABEL_CODES[pred_index]}.png"
        draw_explanation_panel(str(row["image_path"]), lines, panel_path)
        sample_index_rows.append(
            {
                "setting_name": setting_name,
                "seed": seed,
                "sample_rank": sample_rank,
                "sample_index": sample_index,
                "image_path": row["image_path"],
                "panel_path": str(panel_path),
                "true_class": LABEL_CODES[true_index],
                "predicted_class": LABEL_CODES[pred_index],
                "predicted_probability": pred_prob,
                "is_correct": bool(true_index == pred_index),
            }
        )

    write_csv(
        output_dir / "explanation_sample_index.csv",
        [
            "setting_name",
            "seed",
            "sample_rank",
            "sample_index",
            "image_path",
            "panel_path",
            "true_class",
            "predicted_class",
            "predicted_probability",
            "is_correct",
        ],
        sample_index_rows,
    )
    write_csv(
        output_dir / "explanation_concepts.csv",
        [
            "setting_name",
            "seed",
            "sample_rank",
            "sample_index",
            "image_path",
            "true_class",
            "predicted_class",
            "predicted_probability",
            "concept_rank",
            "concept_index",
            "concept_class",
            "concept",
            "activation",
            "pred_class_weight",
            "contribution_to_pred_logit",
        ],
        explanation_rows,
    )
    return sample_index_rows


def write_final_analysis_summary(
    output_path: Path,
    args: argparse.Namespace,
    per_class_summary: list[dict[str, Any]],
    command_line: str,
) -> None:
    lines = [
        "# Final V2 Per-Class And Explainability Analysis",
        "",
        "This analysis reads trained checkpoints only. It does not retrain models.",
        "",
        "## Inputs",
        "",
        f"- Bank: `{args.bank_dir}`",
        f"- Checkpoint root: `{args.checkpoint_root}`",
        f"- Seeds: `{args.seed}`",
        f"- CBM setting: `{args.cbm_setting_name}`",
        f"- CyCL setting: `{args.cycl_setting_name}`",
        "",
        "Command:",
        "",
        f"```bash\n{command_line}\n```",
        "",
        "## Per-Class Test Metrics",
        "",
        "| Setting | Class | Precision | Recall | F1 | Support |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in per_class_summary:
        lines.append(
            f"| `{row['setting_name']}` | `{row['class_name']}` | "
            f"{float(row['precision_mean']):.4f} ± {float(row['precision_std']):.4f} | "
            f"{float(row['recall_mean']):.4f} ± {float(row['recall_std']):.4f} | "
            f"{float(row['f1_mean']):.4f} ± {float(row['f1_std']):.4f} | {int(row['support'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Files",
            "",
            "- `per_class_metrics_test.csv`: per-seed precision/recall/F1.",
            "- `per_class_metrics_summary.csv`: mean/std across seeds.",
            "- `confusion_matrix_*_aggregate.csv/png`: 3-seed aggregate confusion matrices.",
            "- `explainability_samples/`: test images with top concepts and classifier contributions.",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_final_analysis(args: argparse.Namespace) -> None:
    ensure_dir(args.output_dir)
    if args.seed is None:
        args.seed = [42, 43, 44]
    else:
        args.seed = list(dict.fromkeys(int(seed) for seed in args.seed))
    device = resolve_device(args.device)
    split_payloads = load_split_embeddings(args.embeddings_dir)
    bank = load_fixed_bank_v2(args.bank_dir)
    cache = build_soft_targets_v2(bank, split_payloads)
    settings = [
        {"setting_name": args.cbm_setting_name, "model_type": "cbm_v2"},
        {"setting_name": args.cycl_setting_name, "model_type": "cycl_v2"},
    ]
    per_class_rows: list[dict[str, Any]] = []
    prediction_payloads: dict[tuple[str, int], dict[str, Any]] = {}
    models: dict[tuple[str, int], CBMCyCLV2] = {}

    for setting in settings:
        aggregate_matrix = np.zeros((len(LABEL_CODES), len(LABEL_CODES)), dtype=np.int64)
        for seed in args.seed:
            checkpoint_path = args.checkpoint_root / f"{setting['setting_name']}_seed{seed}_best.pt"
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
            model = load_v2_checkpoint_model(checkpoint_path, cache, device, args.dropout, args.proj_dim)
            payload = predict_split_with_model(model, cache, split_payloads, "test", device)
            prediction_payloads[(setting["setting_name"], seed)] = payload
            models[(setting["setting_name"], seed)] = model
            per_class_rows.extend(
                per_class_metric_rows(
                    labels=payload["labels"],
                    predictions=payload["predictions"],
                    setting_name=setting["setting_name"],
                    seed=seed,
                    split="test",
                )
            )
            matrix = compute_confusion_matrix(payload["labels"], payload["predictions"])
            aggregate_matrix += matrix
            seed_prefix = f"confusion_matrix_{setting['setting_name']}_seed{seed}"
            write_csv(args.output_dir / f"{seed_prefix}.csv", ["true_class", *LABEL_CODES, "support"], confusion_matrix_rows(matrix))
            plot_confusion_matrix_heatmap(matrix, args.output_dir / f"{seed_prefix}.png", f"{setting['setting_name']} seed={seed} test")
        prefix = f"confusion_matrix_{setting['setting_name']}_aggregate"
        write_csv(args.output_dir / f"{prefix}.csv", ["true_class", *LABEL_CODES, "support"], confusion_matrix_rows(aggregate_matrix))
        plot_confusion_matrix_heatmap(aggregate_matrix, args.output_dir / f"{prefix}.png", f"{setting['setting_name']} aggregate test")

    per_class_fields = [
        "setting_name",
        "seed",
        "split",
        "class_name",
        "precision",
        "recall",
        "f1",
        "support",
        "tp",
        "fp",
        "fn",
    ]
    summary_fields = [
        "setting_name",
        "split",
        "class_name",
        "n_seeds",
        "support",
        "precision_mean",
        "precision_std",
        "recall_mean",
        "recall_std",
        "f1_mean",
        "f1_std",
    ]
    per_class_summary = summarize_per_class_rows(per_class_rows)
    write_csv(args.output_dir / "per_class_metrics_test.csv", per_class_fields, per_class_rows)
    write_csv(args.output_dir / "per_class_metrics_summary.csv", summary_fields, per_class_summary)

    explain_key = (args.explain_setting_name, int(args.explain_seed))
    if explain_key not in prediction_payloads:
        checkpoint_path = args.checkpoint_root / f"{args.explain_setting_name}_seed{args.explain_seed}_best.pt"
        model = load_v2_checkpoint_model(checkpoint_path, cache, device, args.dropout, args.proj_dim)
        payload = predict_split_with_model(model, cache, split_payloads, "test", device)
    else:
        model = models[explain_key]
        payload = prediction_payloads[explain_key]
    write_explanation_samples(
        model=model,
        cache=cache,
        prediction_payload=payload,
        setting_name=args.explain_setting_name,
        seed=int(args.explain_seed),
        output_dir=args.output_dir / "explainability_samples",
        num_samples=args.num_samples,
        top_k=args.top_k_concepts,
    )
    write_final_analysis_summary(
        output_path=args.output_dir / "final_analysis_summary.md",
        args=args,
        per_class_summary=per_class_summary,
        command_line="python " + " ".join(sys.argv),
    )
    print(f"[done] final analysis written to {args.output_dir}", flush=True)


def main() -> None:
    args = parse_args()
    if args.command == "matrix":
        run_matrix(args)
    elif args.command == "prepare":
        run_prepare(args)
    elif args.command == "preview-augmentations":
        run_preview_augmentations(args)
    elif args.command == "debug-pairs":
        run_debug_pairs(args)
    elif args.command == "smoke-test":
        run_smoke_test(args)
    elif args.command == "train-debug":
        run_train_debug(args)
    elif args.command == "diagnostic-ablation":
        run_diagnostic_ablation(args)
    elif args.command == "formal-it-compare":
        run_formal_it_compare(args)
    elif args.command == "supplement-main-ablation":
        run_supplement_main_ablation(args)
    elif args.command == "final-analysis":
        run_final_analysis(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
