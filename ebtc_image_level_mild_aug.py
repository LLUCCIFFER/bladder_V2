#!/usr/bin/env python3
"""Image-level mild augmentation experiment for refined-vector CBM.

The current CBM pipeline mostly trains on cached BioMedCLIP embeddings. This
script adds a lightweight image-level augmentation stage before feature caching:

1. read official train manifest image paths,
2. generate deterministic mild augmented views from raw images,
3. encode each view with frozen BioMedCLIP,
4. pass augmented embeddings through the existing refinement adapter,
5. train the same concept-bottleneck classifier on expanded train features.

Validation and test remain single-view refined embeddings.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset

from ebtc_class_weight_ablation import (
    describe_weights,
    evaluate_ensemble,
    train_one_weight_mode,
)
from ebtc_concept_experiment import load_local_biomedclip
from ebtc_discriminative_whitelist_cycl import SplitPayload, load_split_payloads, write_csv, write_json
from ebtc_embedding_refinement_stage import ImageTextRefinementModel
from ebtc_project_paths import OFFICIAL_EMBEDDINGS_DIR, OUTPUT_ROOT, PROJECT_ROOT


DEFAULT_REFINED_STAGE_DIR = OUTPUT_ROOT / "ebtc_embedding_refinement_stage_conservative_outputs"
DEFAULT_OUTPUT_DIR = OUTPUT_ROOT / "ebtc_image_level_mild_aug_outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run image-level mild augmentation for refined-vector CBM.")
    parser.add_argument("--refined-stage-dir", type=Path, default=DEFAULT_REFINED_STAGE_DIR)
    parser.add_argument("--embeddings-dir", type=Path, default=OFFICIAL_EMBEDDINGS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--views", default="orig,color_bright,color_dark,rotate,crop")
    parser.add_argument(
        "--train-aggregation",
        choices=["expand", "mean"],
        default="expand",
        help="Use every augmented view as a train row, or average views back to one embedding per image.",
    )
    parser.add_argument("--weight-modes", default="ntl_boost2.5,inverse")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-cycl", type=float, default=0.0)
    parser.add_argument("--lambda-align", type=float, default=0.10)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--force-reencode", action="store_true")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_csv_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mild_aug_pil(image: Image.Image, view_name: str) -> Image.Image:
    """Apply deterministic mild PIL-level augmentation before BioMedCLIP preprocessing."""

    if view_name == "orig":
        return image
    if view_name == "color_bright":
        out = ImageEnhance.Brightness(image).enhance(1.06)
        out = ImageEnhance.Contrast(out).enhance(1.04)
        out = ImageEnhance.Color(out).enhance(1.03)
        return out
    if view_name == "color_dark":
        out = ImageEnhance.Brightness(image).enhance(0.94)
        out = ImageEnhance.Contrast(out).enhance(1.05)
        out = ImageEnhance.Color(out).enhance(0.97)
        return out
    if view_name == "rotate":
        return image.rotate(4.0, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=(0, 0, 0))
    if view_name == "crop":
        width, height = image.size
        crop_ratio = 0.94
        crop_w = int(width * crop_ratio)
        crop_h = int(height * crop_ratio)
        left = (width - crop_w) // 2
        upper = (height - crop_h) // 2
        return image.crop((left, upper, left + crop_w, upper + crop_h)).resize((width, height), Image.Resampling.BICUBIC)
    raise ValueError(f"Unsupported view name: {view_name}")


class MildAugmentedImageDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], view_names: list[str], preprocess: Any) -> None:
        self.rows = rows
        self.view_names = view_names
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.rows) * len(self.view_names)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        row_index = index // len(self.view_names)
        view_index = index % len(self.view_names)
        row = self.rows[row_index]
        view_name = self.view_names[view_index]
        with Image.open(row["image_path"]) as image:
            image_rgb = image.convert("RGB")
            aug_image = mild_aug_pil(image_rgb, view_name)
            tensor = self.preprocess(aug_image)
        return tensor, row_index, view_index


def encode_augmented_train_embeddings(
    rows: list[dict[str, str]],
    view_names: list[str],
    preprocess: Any,
    image_encoder: Any,
    device: str,
    batch_size: int,
    num_workers: int,
) -> np.ndarray:
    dataset = MildAugmentedImageDataset(rows=rows, view_names=view_names, preprocess=preprocess)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    parts: list[np.ndarray] = []
    with torch.no_grad():
        for images, _, _ in loader:
            images = images.to(device)
            features = image_encoder.encode_image(images)
            features = F.normalize(features, dim=-1)
            parts.append(features.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(parts, axis=0).astype(np.float32)


def load_refinement_model(refined_stage_dir: Path, dim: int, device: str) -> ImageTextRefinementModel:
    checkpoint_path = refined_stage_dir / "adapter_refinement" / "best_refinement_adapters.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_args = checkpoint.get("args", {})
    hidden_dim = int(checkpoint_args.get("adapter_hidden_dim", 128))
    dropout = float(checkpoint_args.get("adapter_dropout", 0.1))
    model = ImageTextRefinementModel(dim=dim, hidden_dim=hidden_dim, dropout=dropout).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def refine_embeddings(raw_embeddings: np.ndarray, refined_stage_dir: Path, device: str, batch_size: int) -> np.ndarray:
    model = load_refinement_model(refined_stage_dir, dim=raw_embeddings.shape[1], device=device)
    parts: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, raw_embeddings.shape[0], batch_size):
            batch = torch.tensor(raw_embeddings[start : start + batch_size], dtype=torch.float32, device=device)
            refined = model.encode_images(batch)
            parts.append(refined.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(parts, axis=0).astype(np.float32)


def build_augmented_manifest(rows: list[dict[str, str]], view_names: list[str]) -> list[dict[str, str]]:
    augmented: list[dict[str, str]] = []
    for row_index, row in enumerate(rows):
        for view_index, view_name in enumerate(view_names):
            out = dict(row)
            out["source_row_index"] = str(row_index)
            out["aug_view_index"] = str(view_index)
            out["aug_view_name"] = view_name
            out["aug_image_id"] = f"{row.get('image_id', row_index)}__{view_name}"
            augmented.append(out)
    return augmented


def save_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    ensure_dir(path.parent)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def l2_normalize(array: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.clip(denom, 1e-8, None)


def load_or_create_augmented_train_payload(args: argparse.Namespace, device: str) -> tuple[SplitPayload, list[str]]:
    cache_dir = args.output_dir / "augmented_embeddings"
    ensure_dir(cache_dir)
    view_names = parse_csv_list(args.views)
    raw_path = cache_dir / "image_embeddings_train_mildaug_raw.npy"
    refined_path = cache_dir / "image_embeddings_train_mildaug_refined.npy"
    manifest_path = cache_dir / "image_embeddings_train_mildaug_manifest.csv"

    train_rows = read_manifest(args.embeddings_dir / "image_embeddings_train_manifest.csv")
    augmented_rows = build_augmented_manifest(train_rows, view_names)

    if not args.force_reencode and refined_path.exists() and manifest_path.exists():
        refined = np.load(refined_path).astype(np.float32)
        rows = read_manifest(manifest_path)
        if args.train_aggregation == "mean":
            n_views = len(view_names)
            refined_mean = l2_normalize(refined.reshape(len(train_rows), n_views, -1).mean(axis=1).astype(np.float32))
            labels = np.array([["HGC", "LGC", "NTL", "NST"].index(row["class_name"]) for row in train_rows], dtype=np.int64)
            return SplitPayload(embeddings=refined_mean, labels=labels, rows=train_rows), view_names
        labels = np.array([["HGC", "LGC", "NTL", "NST"].index(row["class_name"]) for row in rows], dtype=np.int64)
        return SplitPayload(embeddings=refined, labels=labels, rows=rows), view_names

    image_encoder, _, preprocess = load_local_biomedclip(device, PROJECT_ROOT / "biomedbert_text_config")
    raw_embeddings = encode_augmented_train_embeddings(
        rows=train_rows,
        view_names=view_names,
        preprocess=preprocess,
        image_encoder=image_encoder,
        device=device,
        batch_size=args.encode_batch_size,
        num_workers=args.num_workers,
    )
    np.save(raw_path, raw_embeddings)
    refined_embeddings = refine_embeddings(raw_embeddings, args.refined_stage_dir, device, args.encode_batch_size)
    np.save(refined_path, refined_embeddings)
    save_manifest(manifest_path, augmented_rows)
    if args.train_aggregation == "mean":
        n_views = len(view_names)
        refined_mean = l2_normalize(refined_embeddings.reshape(len(train_rows), n_views, -1).mean(axis=1).astype(np.float32))
        labels = np.array([["HGC", "LGC", "NTL", "NST"].index(row["class_name"]) for row in train_rows], dtype=np.int64)
        return SplitPayload(embeddings=refined_mean, labels=labels, rows=train_rows), view_names
    labels = np.array([["HGC", "LGC", "NTL", "NST"].index(row["class_name"]) for row in augmented_rows], dtype=np.int64)
    return SplitPayload(embeddings=refined_embeddings, labels=labels, rows=augmented_rows), view_names


def load_clean_val_test_payloads(refined_stage_dir: Path, embeddings_dir: Path) -> dict[str, SplitPayload]:
    payloads = load_split_payloads(embeddings_dir)
    adapter_dir = refined_stage_dir / "adapter_refinement"
    for split in ["val", "test"]:
        payloads[split].embeddings = np.load(adapter_dir / f"refined_image_embeddings_{split}.npy").astype(np.float32)
    return {"val": payloads["val"], "test": payloads["test"]}


def load_concept_bank(refined_stage_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = np.load(refined_stage_dir / "refined_vectors_original_whitelist" / "whitelist_top10.npz", allow_pickle=True)
    return payload["concept_embeddings"].astype(np.float32), payload["M"].astype(np.float32)


def write_mild_aug_report(
    output_dir: Path,
    view_names: list[str],
    weight_modes: list[str],
    baseline_paths: list[Path],
) -> None:
    ensemble_path = output_dir / "mild_aug_ensemble_results.csv"
    lines = [
        "# Image-Level Mild Augmentation Results",
        "",
        "## Setup",
        "",
        "- Train images are expanded into deterministic mild image-level views before BioMedCLIP encoding.",
        f"- Views: `{', '.join(view_names)}`.",
        "- Val/test stay single-view clean refined embeddings.",
        "- Concept bank: refined vectors + original top10 whitelist.",
        "- Classifier path remains concept-bottleneck only.",
        "",
        "## Output Files",
        "",
        f"- Ensemble results: `{ensemble_path}`",
        f"- Augmented embeddings: `{output_dir / 'augmented_embeddings'}`",
        "",
        "## Baseline Reference Files",
        "",
    ]
    for path in baseline_paths:
        lines.append(f"- `{path}`")
    lines.extend(["", "## Weight Modes Run", ""])
    for mode in weight_modes:
        lines.append(f"- `{mode}`")
    if ensemble_path.exists():
        lines.extend(["", "## Results", ""])
        lines.append(ensemble_path.read_text(encoding="utf-8"))
    (output_dir / "mild_aug_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)
        try:
            torch.set_num_interop_threads(args.num_threads)
        except RuntimeError:
            pass

    device = resolve_device(args.device)
    ensure_dir(args.output_dir)
    write_json(
        args.output_dir / "experiment_config.json",
        {
            **vars(args),
            "refined_stage_dir": str(args.refined_stage_dir),
            "embeddings_dir": str(args.embeddings_dir),
            "output_dir": str(args.output_dir),
            "resolved_device": device,
            "train_aggregation": args.train_aggregation,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
        },
    )

    train_payload, view_names = load_or_create_augmented_train_payload(args, device)
    clean_payloads = load_clean_val_test_payloads(args.refined_stage_dir, args.embeddings_dir)
    split_payloads = {"train": train_payload, **clean_payloads}
    concept_embeddings, m_matrix = load_concept_bank(args.refined_stage_dir)

    weight_modes = parse_csv_list(args.weight_modes)
    seeds = parse_int_list(args.seeds)
    weight_rows = [describe_weights(split_payloads["train"].labels, mode) for mode in weight_modes]
    write_csv(args.output_dir / "mild_aug_class_weight_values.csv", weight_rows)

    train_args = argparse.Namespace(
        device=device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        lambda_cycl=args.lambda_cycl,
        lambda_align=args.lambda_align,
        tau=args.tau,
        batch_size=args.train_batch_size,
        epochs=args.epochs,
        patience=args.patience,
    )

    seed_rows: list[dict[str, Any]] = []
    for mode in weight_modes:
        for seed in seeds:
            print(f"[train] mild_aug mode={mode} seed={seed} device={device}", flush=True)
            row = train_one_weight_mode(split_payloads, concept_embeddings, m_matrix, seed, mode, train_args, args.output_dir)
            row["augmentation"] = "image_level_mild"
            row["train_aggregation"] = args.train_aggregation
            row["n_train_rows"] = int(train_payload.embeddings.shape[0])
            row["views"] = ",".join(view_names)
            seed_rows.append(row)
    write_csv(args.output_dir / "mild_aug_seed_results.csv", seed_rows)

    ensemble_rows: list[dict[str, Any]] = []
    for mode in weight_modes:
        rows = evaluate_ensemble(args.output_dir, mode, split_payloads, concept_embeddings, train_args)
        for row in rows:
            row["augmentation"] = "image_level_mild"
            row["train_aggregation"] = args.train_aggregation
            row["views"] = ",".join(view_names)
            row["n_train_rows"] = int(train_payload.embeddings.shape[0])
        ensemble_rows.extend(rows)
    write_csv(args.output_dir / "mild_aug_ensemble_results.csv", ensemble_rows)

    baseline_paths = [
        OUTPUT_ROOT / "ebtc_class_weight_ablation_search2_outputs" / "class_weight_ensemble_results.csv",
        OUTPUT_ROOT / "ebtc_class_weight_ablation_outputs" / "class_weight_ensemble_results.csv",
    ]
    write_mild_aug_report(args.output_dir, view_names, weight_modes, baseline_paths)
    print(json.dumps({"seed_rows": seed_rows, "ensemble_rows": ensemble_rows}, indent=2))


if __name__ == "__main__":
    main()
