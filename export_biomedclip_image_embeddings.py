#!/usr/bin/env python3
"""Export cached BioMedCLIP image embeddings with image names and labels.

The EBTC pipeline stores image embeddings as split-specific .npy arrays and
metadata as split-specific manifest CSV files. This script merges them into
flat CSV and pickle artifacts that are easier to inspect or share.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Iterable

import numpy as np

from ebtc_project_paths import IMAGE_EMBEDDING_EXPORT_DIR, OFFICIAL_EMBEDDINGS_DIR


DEFAULT_EMBEDDINGS_DIR = OFFICIAL_EMBEDDINGS_DIR
DEFAULT_OUTPUT_DIR = IMAGE_EMBEDDING_EXPORT_DIR
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export BioMedCLIP image embeddings with image names and labels."
    )
    parser.add_argument("--embeddings-dir", type=Path, default=DEFAULT_EMBEDDINGS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--csv-name", default="image_embeddings_all.csv")
    parser.add_argument("--pickle-name", default="image_embeddings_all.pkl")
    parser.add_argument(
        "--float-format",
        default=".8g",
        help="Format specifier for CSV embedding values, e.g. .8g or .10f.",
    )
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_split(embeddings_dir: Path, split: str) -> tuple[list[dict[str, str]], np.ndarray]:
    embedding_path = embeddings_dir / f"image_embeddings_{split}.npy"
    manifest_path = embeddings_dir / f"image_embeddings_{split}_manifest.csv"
    if not embedding_path.exists():
        raise FileNotFoundError(f"Missing embedding file: {embedding_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest file: {manifest_path}")

    embeddings = np.load(embedding_path).astype(np.float32, copy=False)
    metadata = read_manifest(manifest_path)
    if len(metadata) != embeddings.shape[0]:
        raise ValueError(
            f"Row mismatch for {split}: manifest has {len(metadata)} rows, "
            f"embeddings have {embeddings.shape[0]} rows."
        )
    for row in metadata:
        row["split"] = split
        row["label"] = row.get("class_name", "")
        row["image_name"] = row.get("annotation_image_name") or Path(row.get("image_path", "")).name
    return metadata, embeddings


def write_csv(
    output_path: Path,
    metadata: list[dict[str, str]],
    embeddings: np.ndarray,
    float_format: str,
) -> None:
    metadata_fields = [
        "split",
        "image_id",
        "image_name",
        "image_path",
        "label",
        "class_name",
        "case_id",
        "pt_id",
        "frame_id",
        "group_id",
        "official_split",
    ]
    embedding_fields = [f"emb_{idx:03d}" for idx in range(embeddings.shape[1])]
    fieldnames = metadata_fields + embedding_fields

    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row, vector in zip(metadata, embeddings, strict=True):
            out = {field: row.get(field, "") for field in metadata_fields}
            out.update(
                {
                    field: format(float(value), float_format)
                    for field, value in zip(embedding_fields, vector, strict=True)
                }
            )
            writer.writerow(out)


def write_pickle(
    output_path: Path,
    metadata: list[dict[str, str]],
    embeddings: np.ndarray,
    source_files: Iterable[str],
) -> None:
    payload = {
        "metadata": metadata,
        "embeddings": embeddings,
        "embedding_dim": int(embeddings.shape[1]),
        "embedding_columns": [f"emb_{idx:03d}" for idx in range(embeddings.shape[1])],
        "source_files": list(source_files),
        "notes": "BioMedCLIP image embeddings. Embeddings are aligned row-by-row with metadata.",
    }
    with output_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_metadata: list[dict[str, str]] = []
    split_embeddings: list[np.ndarray] = []
    source_files: list[str] = []
    split_counts: dict[str, int] = {}

    for split in SPLITS:
        metadata, embeddings = load_split(args.embeddings_dir, split)
        all_metadata.extend(metadata)
        split_embeddings.append(embeddings)
        split_counts[split] = int(embeddings.shape[0])
        source_files.append(str(args.embeddings_dir / f"image_embeddings_{split}.npy"))
        source_files.append(str(args.embeddings_dir / f"image_embeddings_{split}_manifest.csv"))

    all_embeddings = np.concatenate(split_embeddings, axis=0).astype(np.float32, copy=False)

    csv_path = args.output_dir / args.csv_name
    pickle_path = args.output_dir / args.pickle_name
    summary_path = args.output_dir / "image_embeddings_export_summary.json"

    write_csv(csv_path, all_metadata, all_embeddings, args.float_format)
    write_pickle(pickle_path, all_metadata, all_embeddings, source_files)

    labels: dict[str, int] = {}
    for row in all_metadata:
        label = row.get("label", "")
        labels[label] = labels.get(label, 0) + 1

    summary = {
        "output_csv": str(csv_path),
        "output_pickle": str(pickle_path),
        "n_images": int(all_embeddings.shape[0]),
        "embedding_dim": int(all_embeddings.shape[1]),
        "split_counts": split_counts,
        "label_counts": labels,
        "embeddings_dir": str(args.embeddings_dir),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
