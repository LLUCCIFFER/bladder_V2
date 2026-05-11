#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.linalg import LinAlgWarning
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoConfig

from ebtc_project_paths import (
    BIOMEDBERT_TEXT_CONFIG_DIR,
    CONCEPT_DIR,
    CONCEPT_EXPERIMENT_OUTPUT_DIR,
    DATASET_ROOT,
    MODEL_DIR,
)

ANNOTATIONS_PATH = DATASET_ROOT / "annotations.csv"
DEFAULT_OUTPUT_DIR = CONCEPT_EXPERIMENT_OUTPUT_DIR
CLASSES = ["HGC", "LGC", "NTL", "NST"]
SEED = 20260411
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
TEMPLATES = [
    "this is a white-light cystoscopy image showing {concept}",
    "white-light cystoscopy image with {concept}",
    "bladder endoscopy showing {concept}",
]
BINARY_CLASSIFIERS = ["logreg", "linearsvm"]
MULTICLASS_CLASSIFIERS = ["logreg", "linearsvm"]
BINARY_FEATURE_SETS = ["raw", "elbow", "top300", "top500", "top700", "drop10", "drop15", "drop20"]
MULTICLASS_FILTER_SETTINGS = ["elbow", "top300", "top500", "top700", "drop10", "drop15", "drop20"]
CV_FOLDS = 5
EXPECTED_BINARY_SUMMARY_ROWS = len(CLASSES) * len(BINARY_FEATURE_SETS) * len(BINARY_CLASSIFIERS)
ADAPTIVE_BINARY_SELECTOR_CLASSIFIERS = ["logreg", "linearsvm"]
ADAPTIVE_MULTICLASS_SETTING_NAMES = [f"binarybest_overall_{name}" for name in ADAPTIVE_BINARY_SELECTOR_CLASSIFIERS]
EXPECTED_MULTICLASS_ROWS = (1 + len(MULTICLASS_FILTER_SETTINGS) + len(ADAPTIVE_MULTICLASS_SETTING_NAMES)) * len(
    MULTICLASS_CLASSIFIERS
)

GROUP_PARSER_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("std", re.compile(r"^case_(?P<case>\d+)_pt_(?P<pt>\d+)_frame_(?P<frame>\d+)$")),
    ("hlt", re.compile(r"^case_(?P<case>\d+)_pt_(?P<pt>\d+)_(?:HLT|NST)_?_frame_(?P<frame>\d+)$")),
    ("cys_a", re.compile(r"^case_(?P<case>\d+)_cys_pt(?P<pt>\d+)_(?P<frame>\d+)$")),
    (
        "cys_b",
        re.compile(r"^cys_case_(?P<case>\d+)_pt0?(?P<pt>\d+)(?:_frame)?_(?P<frame>\d+)(?: \(copy\))?$"),
    ),
    ("cys_c", re.compile(r"^cys_case_(?P<case>\d+)_(?P<frame>\d+)$")),
]

NEGATION_PATTERNS = [
    re.compile(r"^(no|without|absence of|lack of|free of|negative for)\b"),
    re.compile(r"\b(no obvious|without obvious|no clear)\b"),
]
UNCERTAINTY_PATTERNS = [
    re.compile(r"\b(possible|possibly|probable|probably|likely|unlikely|suggestive|suspicious|uncertain|indeterminate)\b"),
    re.compile(r"\b(may|might|could|perhaps|compatible with|consistent with)\b"),
]
DIAGNOSIS_PATTERNS = [
    re.compile(r"\bhgc\b"),
    re.compile(r"\blgc\b"),
    re.compile(r"\bntl\b"),
    re.compile(r"\bnst\b"),
    re.compile(r"\bhigh[- ]grade\b"),
    re.compile(r"\blow[- ]grade\b"),
    re.compile(r"\bnon[- ]tumou?r\b"),
    re.compile(r"\bnon[- ]suspicious\b"),
    re.compile(r"\bcarcinoma\b"),
    re.compile(r"\bcancer\b"),
    re.compile(r"\bmalignan\w*\b"),
    re.compile(r"\bneoplasm\w*\b"),
    re.compile(r"\btumou?r\b"),
    re.compile(r"\bdiagnos\w*\b"),
]
PATHOLOGY_TERMS = {
    "histology",
    "histologic",
    "histological",
    "histopathology",
    "pathology",
    "pathologic",
    "pathological",
    "cytology",
    "cytologic",
    "cytological",
    "cell",
    "cells",
    "cellular",
    "nuclear",
    "nuclei",
    "mitosis",
    "mitotic",
    "urothelial",
    "microscopic",
    "microvascular",
    "biopsy",
}
ARTIFACT_TERMS = {
    "blur",
    "blurry",
    "glare",
    "reflection",
    "reflections",
    "shadow",
    "shadows",
    "artifact",
    "artefact",
    "instrument",
    "scope",
    "camera",
    "specular",
    "overexposed",
    "underexposed",
    "halation",
}
SYNONYM_REPLACEMENTS = {
    "blood vessel": "vessel",
    "blood vessels": "vessels",
    "vascularity": "vascular pattern",
    "vessel pattern": "vascular pattern",
    "vessels pattern": "vascular pattern",
    "frond like": "frond-like",
    "finger like": "finger-like",
    "cauliflower like": "cauliflower-like",
    "broad based": "broad-based",
    "sheet like": "sheet-like",
    "plateau like": "plateau-like",
}


@dataclass
class ParsedImageName:
    parser_name: str
    case_id: str
    pt_id: str
    frame_id: str


class ImageManifestDataset(Dataset):
    def __init__(self, manifest_df: pd.DataFrame, preprocess: Any) -> None:
        self.records = manifest_df.reset_index(drop=True)
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.records.iloc[index]
        with Image.open(row["image_path"]) as image:
            image = image.convert("RGB")
            tensor = self.preprocess(image)
        return tensor, index


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def trailing_punct_strip(text: str) -> str:
    return re.sub(r"[\s\.,;:!?]+$", "", text)


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def clean_concept_text(text: str) -> str:
    text = text.lower().strip()
    text = normalize_spaces(text)
    text = trailing_punct_strip(text)
    return text


def singularize_token(token: str) -> str:
    if len(token) <= 3:
        return token
    if token.endswith(("ss", "us", "is", "ous", "ics")):
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ses") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and not token.endswith(("as", "es")):
        return token[:-1]
    return token


def canonicalize_concept(text: str) -> str:
    text = clean_concept_text(text)
    for source, target in SYNONYM_REPLACEMENTS.items():
        text = text.replace(source, target)
    tokens = [singularize_token(token) for token in text.split()]
    return " ".join(tokens)


def parse_concat_json_arrays(text: str) -> list[list[str]]:
    decoder = json.JSONDecoder()
    arrays: list[list[str]] = []
    index = 0
    total = len(text)
    while index < total:
        while index < total and text[index].isspace():
            index += 1
        if index >= total:
            break
        obj, end = decoder.raw_decode(text, index)
        if not isinstance(obj, list):
            raise ValueError("Expected a JSON array in concatenated concept file.")
        arrays.append([str(item) for item in obj])
        index = end
    return arrays


def load_raw_concept_records() -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for class_name in CLASSES:
        path = CONCEPT_DIR / f"{class_name}.txt"
        arrays = parse_concat_json_arrays(path.read_text(encoding="utf-8"))
        rows: list[dict[str, Any]] = []
        flat_index = 0
        for block_index, concepts in enumerate(arrays, start=1):
            for item in concepts:
                rows.append(
                    {
                        "class_name": class_name,
                        "concept_raw": str(item),
                        "source_prompt_family": f"baseconcept_block_{block_index:02d}",
                        "source_llm": "provided_baseconcept",
                        "raw_index": flat_index,
                        "block_index": block_index,
                    }
                )
                flat_index += 1
        output[class_name] = rows
    return output


def concept_drop_reasons(text: str) -> list[str]:
    reasons: list[str] = []
    for pattern in NEGATION_PATTERNS:
        if pattern.search(text):
            reasons.append("negation")
            break
    for pattern in UNCERTAINTY_PATTERNS:
        if pattern.search(text):
            reasons.append("uncertainty")
            break
    for pattern in DIAGNOSIS_PATTERNS:
        if pattern.search(text):
            reasons.append("diagnosis")
            break
    tokens = set(re.findall(r"[a-z]+", text))
    if tokens & PATHOLOGY_TERMS:
        reasons.append("pathology_term")
    if tokens & ARTIFACT_TERMS:
        reasons.append("artifact")
    return reasons


def clean_concepts(output_dir: Path) -> dict[str, list[str]]:
    raw_dir = output_dir / "raw_concepts"
    cleaned_dir = output_dir / "cleaned_concepts"
    reports_dir = output_dir / "reports"
    ensure_dir(raw_dir)
    ensure_dir(cleaned_dir)
    ensure_dir(reports_dir)

    raw_records = load_raw_concept_records()
    kept_concepts: dict[str, list[str]] = {}
    cross_class_rows: list[dict[str, Any]] = []
    retained_lookup: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))

    for class_name, records in raw_records.items():
        raw_path = raw_dir / f"{class_name}.txt"
        raw_path.write_text("\n".join(record["concept_raw"] for record in records) + "\n", encoding="utf-8")

        seen_clean: set[str] = set()
        kept_rows: list[str] = []
        cleaned_rows: list[dict[str, Any]] = []
        for record in records:
            concept_raw = record["concept_raw"]
            concept_clean = clean_concept_text(concept_raw)
            concept_canonical = canonicalize_concept(concept_raw)
            is_exact_duplicate = concept_clean in seen_clean
            if not is_exact_duplicate:
                seen_clean.add(concept_clean)
            rule_reasons = concept_drop_reasons(concept_clean)
            keep_after_rule_cleaning = (not is_exact_duplicate) and (len(rule_reasons) == 0)
            if keep_after_rule_cleaning:
                kept_rows.append(concept_clean)
                retained_lookup[concept_clean][class_name].append(record["raw_index"])
            cleaned_rows.append(
                {
                    **record,
                    "concept_clean": concept_clean,
                    "concept_canonical": concept_canonical,
                    "is_exact_duplicate": is_exact_duplicate,
                    "keep_after_rule_cleaning": keep_after_rule_cleaning,
                    "drop_reasons": "|".join(rule_reasons),
                }
            )

        kept_concepts[class_name] = kept_rows
        pd.DataFrame(cleaned_rows).to_csv(cleaned_dir / f"{class_name}.csv", index=False)
        (cleaned_dir / f"{class_name}_keep.txt").write_text("\n".join(kept_rows) + "\n", encoding="utf-8")

    for concept_clean, class_map in sorted(retained_lookup.items()):
        if len(class_map) < 2:
            continue
        cross_class_rows.append(
            {
                "concept_clean": concept_clean,
                "concept_canonical": canonicalize_concept(concept_clean),
                "classes": "|".join(sorted(class_map)),
                "n_classes": len(class_map),
                "raw_indices_by_class": json.dumps(class_map, ensure_ascii=False),
                "high_risk_cross_class": True,
            }
        )
    pd.DataFrame(cross_class_rows).to_csv(reports_dir / "cross_class_duplicate_report.csv", index=False)

    concept_summary_rows = []
    for class_name, records in raw_records.items():
        df = pd.read_csv(cleaned_dir / f"{class_name}.csv")
        concept_summary_rows.append(
            {
                "class_name": class_name,
                "raw_total": len(records),
                "unique_clean_total": int(df["concept_clean"].nunique()),
                "kept_after_rule_cleaning": int(df["keep_after_rule_cleaning"].sum()),
                "exact_duplicates_removed": int(df["is_exact_duplicate"].sum()),
                "rule_removed": int((~df["keep_after_rule_cleaning"] & ~df["is_exact_duplicate"]).sum()),
            }
        )
    pd.DataFrame(concept_summary_rows).to_csv(reports_dir / "concept_cleaning_summary.csv", index=False)
    return kept_concepts


def parse_image_name(path: Path) -> ParsedImageName:
    stem = path.stem
    for parser_name, pattern in GROUP_PARSER_PATTERNS:
        match = pattern.match(stem)
        if match:
            case_raw = int(match.group("case"))
            pt_raw = int(match.groupdict().get("pt") or 0)
            frame_raw = int(match.group("frame"))
            return ParsedImageName(
                parser_name=parser_name,
                case_id=f"case_{case_raw:03d}",
                pt_id=f"pt_{pt_raw:03d}",
                frame_id=f"frame_{frame_raw:04d}",
            )
    raise ValueError(f"Unrecognized EBTC filename format: {path.name}")


def build_record_key(class_name: str, case_id: str, pt_id: str, frame_id: str) -> str:
    return f"{class_name}__{case_id}__{pt_id}__{frame_id}"


def build_manifest(output_dir: Path) -> pd.DataFrame:
    split_dir = output_dir / "splits"
    reports_dir = output_dir / "reports"
    ensure_dir(split_dir)
    ensure_dir(reports_dir)

    rows: list[dict[str, Any]] = []
    for class_name in CLASSES:
        class_dir = DATASET_ROOT / class_name
        for path in sorted(class_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            parsed = parse_image_name(path)
            rows.append(
                {
                    "image_id": f"{class_name}__{path.stem}",
                    "image_path": str(path),
                    "class_name": class_name,
                    "case_id": parsed.case_id,
                    "pt_id": parsed.pt_id,
                    "frame_id": parsed.frame_id,
                    "group_id": f"{class_name}__{parsed.case_id}__{parsed.pt_id}",
                    "record_key": build_record_key(class_name, parsed.case_id, parsed.pt_id, parsed.frame_id),
                    "parser_name": parsed.parser_name,
                }
            )

    manifest_df = pd.DataFrame(rows).sort_values(["class_name", "group_id", "frame_id", "image_path"]).reset_index(drop=True)
    manifest_df["row_index"] = np.arange(len(manifest_df))
    manifest_df.to_csv(split_dir / "manifest_all.csv", index=False)

    parser_summary = manifest_df.groupby(["class_name", "parser_name"]).size().reset_index(name="image_count")
    parser_summary.to_csv(reports_dir / "filename_pattern_summary.csv", index=False)
    return manifest_df


def class_distribution_error(df: pd.DataFrame, target_distribution: dict[str, float]) -> float:
    current = df["class_name"].value_counts(normalize=True).to_dict()
    error = 0.0
    for class_name in CLASSES:
        error += abs(current.get(class_name, 0.0) - target_distribution.get(class_name, 0.0))
    return error


def choose_group_fold(df: pd.DataFrame, n_splits: int, seed: int, target_fraction: float) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    dummy_x = np.zeros(len(df))
    y = df["class_name"].to_numpy()
    groups = df["group_id"].to_numpy()
    target_distribution = df["class_name"].value_counts(normalize=True).to_dict()

    best: tuple[np.ndarray, np.ndarray, dict[str, Any]] | None = None
    for fold_index, (train_idx, eval_idx) in enumerate(splitter.split(dummy_x, y, groups)):
        eval_df = df.iloc[eval_idx]
        fraction = len(eval_df) / len(df)
        fraction_error = abs(fraction - target_fraction)
        distribution_error = class_distribution_error(eval_df, target_distribution)
        score = fraction_error * 10.0 + distribution_error
        metadata = {
            "fold_index": fold_index,
            "n_splits": n_splits,
            "target_fraction": target_fraction,
            "actual_fraction": fraction,
            "fraction_error": fraction_error,
            "distribution_error": distribution_error,
            "score": score,
        }
        if best is None or score < best[2]["score"]:
            best = (train_idx, eval_idx, metadata)
    assert best is not None
    return best


def write_split_summary(output_dir: Path, split_map: dict[str, pd.DataFrame]) -> None:
    reports_dir = output_dir / "reports"
    split_dir = output_dir / "splits"
    ensure_dir(reports_dir)
    ensure_dir(split_dir)
    rows: list[dict[str, Any]] = []
    for split_name, df in split_map.items():
        group_counts = df.groupby("class_name")["group_id"].nunique().to_dict()
        image_counts = df["class_name"].value_counts().to_dict()
        for class_name in CLASSES:
            rows.append(
                {
                    "split": split_name,
                    "class_name": class_name,
                    "image_count": int(image_counts.get(class_name, 0)),
                    "group_count": int(group_counts.get(class_name, 0)),
                }
            )
        df.to_csv(split_dir / f"{split_name}.csv", index=False)
    pd.DataFrame(rows).to_csv(reports_dir / "split_summary.csv", index=False)


def load_official_annotations() -> pd.DataFrame:
    if not ANNOTATIONS_PATH.exists():
        raise FileNotFoundError(f"Missing EBTC annotations file: {ANNOTATIONS_PATH}")
    annotations_df = pd.read_csv(ANNOTATIONS_PATH)
    required_columns = {"HLY", "imaging type", "tissue type", "sub_dataset"}
    missing_columns = required_columns - set(annotations_df.columns)
    if missing_columns:
        raise ValueError(f"annotations.csv missing required columns: {sorted(missing_columns)}")

    rows: list[dict[str, Any]] = []
    for row in annotations_df.to_dict(orient="records"):
        image_name = str(row["HLY"]).strip()
        class_name = str(row["tissue type"]).strip()
        split_name = str(row["sub_dataset"]).strip().lower()
        if split_name not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split label in annotations.csv: {split_name}")
        parsed = parse_image_name(Path(image_name))
        rows.append(
            {
                "annotation_image_name": image_name,
                "class_name": class_name,
                "imaging_type": str(row["imaging type"]).strip(),
                "official_split": split_name,
                "case_id": parsed.case_id,
                "pt_id": parsed.pt_id,
                "frame_id": parsed.frame_id,
                "record_key": build_record_key(class_name, parsed.case_id, parsed.pt_id, parsed.frame_id),
            }
        )
    official_df = pd.DataFrame(rows)
    duplicates = official_df["record_key"].duplicated(keep=False)
    if duplicates.any():
        duplicate_keys = official_df.loc[duplicates, "record_key"].unique().tolist()
        raise ValueError(f"Duplicate annotation keys found: {duplicate_keys[:10]}")
    return official_df


def create_splits(manifest_df: pd.DataFrame, output_dir: Path, seed: int) -> dict[str, pd.DataFrame]:
    official_df = load_official_annotations()

    manifest_keys = set(manifest_df["record_key"])
    official_keys = set(official_df["record_key"])
    missing_in_manifest = sorted(official_keys - manifest_keys)
    missing_in_annotations = sorted(manifest_keys - official_keys)
    if missing_in_manifest or missing_in_annotations:
        raise ValueError(
            "Official annotations and image manifest do not match. "
            f"Missing in manifest: {missing_in_manifest[:10]}; missing in annotations: {missing_in_annotations[:10]}"
        )

    merged = manifest_df.merge(
        official_df[["record_key", "annotation_image_name", "imaging_type", "official_split"]],
        on="record_key",
        how="left",
        validate="one_to_one",
    )
    if merged["official_split"].isna().any():
        raise ValueError("Some manifest rows did not receive an official split assignment.")

    split_map: dict[str, pd.DataFrame] = {}
    for split_name in ["train", "val", "test"]:
        split_map[split_name] = (
            merged.loc[merged["official_split"] == split_name]
            .sort_values(["class_name", "group_id", "frame_id", "image_path"])
            .reset_index(drop=True)
        )

    group_cross_split = merged.groupby("group_id")["official_split"].nunique()
    write_split_summary(output_dir, split_map)
    save_json(
        output_dir / "reports" / "split_selection_metadata.json",
        {
            "split_source": "annotations.csv:sub_dataset",
            "annotations_path": str(ANNOTATIONS_PATH),
            "seed": seed,
            "annotation_image_count": int(len(official_df)),
            "manifest_image_count": int(len(manifest_df)),
            "cross_split_group_count": int((group_cross_split > 1).sum()),
            "split_notes": [
                "Official EBTC split is taken directly from annotations.csv sub_dataset.",
                "group_id is class_name + case_id + pt_id because numeric case ids are reused across classes.",
                "Variance scoring, standardization, and probing are fit on train only.",
                "Official split may place the same group_id across multiple splits; train-only coarse screening still keeps test frozen.",
            ],
        },
    )
    return split_map


def prepare_text_config_cache(local_text_config_dir: Path) -> None:
    ensure_dir(local_text_config_dir)
    config_path = local_text_config_dir / "config.json"
    if not config_path.exists():
        AutoConfig.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract").save_pretrained(local_text_config_dir)


def load_local_biomedclip(device: str, local_text_config_dir: Path) -> tuple[Any, Any, Any]:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import open_clip
    from open_clip.factory import _MODEL_CONFIGS

    prepare_text_config_cache(local_text_config_dir)
    with (MODEL_DIR / "open_clip_config.json").open("r", encoding="utf-8") as f:
        config = json.load(f)
    config["model_cfg"]["text_cfg"]["hf_model_name"] = str(local_text_config_dir)
    config["model_cfg"]["text_cfg"]["hf_tokenizer_name"] = str(MODEL_DIR)
    model_name = "biomedclip_local_ebtc_cpu"
    _MODEL_CONFIGS[model_name] = config["model_cfg"]
    preprocess_cfg = config["preprocess_cfg"]
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=model_name,
        pretrained=str(MODEL_DIR / "open_clip_pytorch_model.bin"),
        device=device,
        **{f"image_{key}": value for key, value in preprocess_cfg.items()},
    )
    tokenizer = open_clip.get_tokenizer(f"local-dir:{MODEL_DIR}")
    model.eval()
    return model, tokenizer, preprocess


def tokenize_texts(tokenizer: Any, texts: list[str], device: str) -> torch.Tensor:
    tokens = tokenizer(texts, context_length=256)
    if not isinstance(tokens, torch.Tensor):
        tokens = torch.tensor(tokens)
    return tokens.to(device)


def encode_text_embeddings(
    concepts_by_class: dict[str, list[str]],
    model: Any,
    tokenizer: Any,
    device: str,
    output_dir: Path,
    text_batch_size: int = 256,
) -> dict[str, np.ndarray]:
    embeddings_dir = output_dir / "embeddings"
    ensure_dir(embeddings_dir)
    text_embeddings: dict[str, np.ndarray] = {}
    for class_name in CLASSES:
        embedding_path = embeddings_dir / f"text_embeddings_{class_name}.npy"
        concept_path = embeddings_dir / f"text_embeddings_{class_name}_concepts.csv"
        if embedding_path.exists() and concept_path.exists():
            text_embeddings[class_name] = np.load(embedding_path)
            continue
        concepts = concepts_by_class[class_name]
        prompts = [template.format(concept=concept) for concept in concepts for template in TEMPLATES]
        batches: list[torch.Tensor] = []
        with torch.no_grad():
            for start in tqdm(range(0, len(prompts), text_batch_size), desc=f"[text] {class_name}", leave=False):
                batch_prompts = prompts[start : start + text_batch_size]
                tokens = tokenize_texts(tokenizer, batch_prompts, device)
                features = model.encode_text(tokens)
                features = F.normalize(features, dim=-1)
                batches.append(features.detach().cpu())
        stacked = torch.cat(batches, dim=0).view(len(concepts), len(TEMPLATES), -1)
        concept_embeddings = F.normalize(stacked.mean(dim=1), dim=-1).numpy()
        np.save(embedding_path, concept_embeddings)
        pd.DataFrame(
            {
                "concept": concepts,
                "concept_index": np.arange(len(concepts)),
            }
        ).to_csv(concept_path, index=False)
        text_embeddings[class_name] = concept_embeddings
    return text_embeddings


def encode_image_embeddings(
    split_map: dict[str, pd.DataFrame],
    model: Any,
    preprocess: Any,
    device: str,
    output_dir: Path,
    batch_size: int = 16,
) -> dict[str, np.ndarray]:
    embeddings_dir = output_dir / "embeddings"
    ensure_dir(embeddings_dir)
    image_embeddings: dict[str, np.ndarray] = {}
    for split_name, split_df in split_map.items():
        embedding_path = embeddings_dir / f"image_embeddings_{split_name}.npy"
        manifest_path = embeddings_dir / f"image_embeddings_{split_name}_manifest.csv"
        split_df_reset = split_df.reset_index(drop=True)
        if embedding_path.exists() and manifest_path.exists():
            cached_manifest = pd.read_csv(manifest_path)
            compare_columns = ["image_id", "image_path", "class_name", "group_id"]
            if (
                len(cached_manifest) == len(split_df_reset)
                and cached_manifest[compare_columns].equals(split_df_reset[compare_columns])
            ):
                image_embeddings[split_name] = np.load(embedding_path)
                continue
        dataset = ImageManifestDataset(split_df, preprocess)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        parts: list[np.ndarray] = []
        with torch.no_grad():
            for images, _ in tqdm(loader, desc=f"[image] {split_name}", leave=False):
                images = images.to(device)
                features = model.encode_image(images)
                features = F.normalize(features, dim=-1)
                parts.append(features.detach().cpu().numpy())
        embeddings = np.concatenate(parts, axis=0)
        np.save(embedding_path, embeddings)
        split_df_reset.to_csv(manifest_path, index=False)
        image_embeddings[split_name] = embeddings
    return image_embeddings


def plot_variance_histogram(values: np.ndarray, class_name: str, path: Path) -> None:
    ensure_dir(path.parent)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=40, color="#1f77b4", edgecolor="white")
    ax.set_title(f"{class_name} concept variance distribution")
    ax.set_xlabel("VarAll")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_cumulative_variance(sorted_values: np.ndarray, elbow_rank: int, class_name: str, path: Path) -> None:
    ensure_dir(path.parent)
    x = np.arange(1, len(sorted_values) + 1)
    cumulative = np.cumsum(sorted_values)
    total = float(cumulative[-1]) if len(cumulative) else 0.0
    y = cumulative / total if total > 0 else np.linspace(0, 1, len(x))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, color="#d62728", linewidth=2)
    ax.axvline(elbow_rank, color="#2ca02c", linestyle="--", linewidth=1.5, label=f"elbow={elbow_rank}")
    ax.set_title(f"{class_name} cumulative variance")
    ax.set_xlabel("Concept rank by variance")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_ylim(0, 1.02)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def elbow_rank_from_sorted(values_desc: np.ndarray) -> int:
    if len(values_desc) == 0:
        return 0
    total = float(values_desc.sum())
    if total <= 0:
        return len(values_desc)
    cumulative = np.cumsum(values_desc) / total
    x = np.linspace(0, 1, len(values_desc))
    distances = cumulative - x
    return int(np.argmax(distances) + 1)


def build_variance_score_df(
    concepts: list[str],
    similarity: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, int]:
    var_all = similarity.var(axis=1)
    mean_all = similarity.mean(axis=1)
    range_all = np.percentile(similarity, 95, axis=1) - np.percentile(similarity, 5, axis=1)

    order = np.argsort(-var_all)
    sorted_var = var_all[order]
    elbow_rank = elbow_rank_from_sorted(sorted_var)
    rank_var = np.empty_like(order)
    rank_var[order] = np.arange(1, len(order) + 1)

    n = len(concepts)
    keep_counts = {
        "keep_top300": min(300, n),
        "keep_top500": min(500, n),
        "keep_top700": min(700, n),
        "keep_drop10": max(1, int(math.ceil(n * 0.90))),
        "keep_drop15": max(1, int(math.ceil(n * 0.85))),
        "keep_drop20": max(1, int(math.ceil(n * 0.80))),
    }

    df = (
        pd.DataFrame(
            {
                "concept": concepts,
                "original_index": np.arange(n),
                "var_all": var_all,
                "mean_all": mean_all,
                "range_p95_p5": range_all,
                "rank_var": rank_var,
                "keep_elbow": rank_var <= elbow_rank,
                "keep_top300": rank_var <= keep_counts["keep_top300"],
                "keep_top500": rank_var <= keep_counts["keep_top500"],
                "keep_top700": rank_var <= keep_counts["keep_top700"],
                "keep_drop10": rank_var <= keep_counts["keep_drop10"],
                "keep_drop15": rank_var <= keep_counts["keep_drop15"],
                "keep_drop20": rank_var <= keep_counts["keep_drop20"],
            }
        )
        .sort_values("rank_var")
        .reset_index(drop=True)
    )
    return df, var_all, sorted_var, elbow_rank


def variance_scoring(
    concepts_by_class: dict[str, list[str]],
    text_embeddings: dict[str, np.ndarray],
    image_embeddings_train: np.ndarray,
    output_dir: Path,
) -> dict[str, pd.DataFrame]:
    score_dir = output_dir / "concept_scores"
    plots_dir = output_dir / "plots"
    ensure_dir(score_dir)
    ensure_dir(plots_dir)
    scores_by_class: dict[str, pd.DataFrame] = {}

    for class_name in CLASSES:
        score_path = score_dir / f"{class_name}_variance_scores.csv"
        hist_path = plots_dir / f"{class_name}_variance_hist.png"
        curve_path = plots_dir / f"{class_name}_cumulative_variance.png"
        if score_path.exists() and hist_path.exists() and curve_path.exists():
            existing_df = pd.read_csv(score_path)
            if "original_index" in existing_df.columns:
                scores_by_class[class_name] = existing_df
                continue
        concepts = concepts_by_class[class_name]
        text_matrix = text_embeddings[class_name]
        similarity = text_matrix @ image_embeddings_train.T
        df, var_all, sorted_var, elbow_rank = build_variance_score_df(concepts, similarity)
        df.to_csv(score_path, index=False)
        plot_variance_histogram(var_all, class_name, hist_path)
        plot_cumulative_variance(sorted_var, elbow_rank, class_name, curve_path)
        scores_by_class[class_name] = df

    return scores_by_class


def build_binary_feature_sets(
    score_df: pd.DataFrame,
) -> dict[str, np.ndarray]:
    original_indices = score_df["original_index"].to_numpy(dtype=int)
    feature_sets = {
        "raw": original_indices,
        "elbow": score_df.loc[score_df["keep_elbow"], "original_index"].to_numpy(dtype=int),
        "top300": score_df.loc[score_df["keep_top300"], "original_index"].to_numpy(dtype=int),
        "top500": score_df.loc[score_df["keep_top500"], "original_index"].to_numpy(dtype=int),
        "top700": score_df.loc[score_df["keep_top700"], "original_index"].to_numpy(dtype=int),
        "drop10": score_df.loc[score_df["keep_drop10"], "original_index"].to_numpy(dtype=int),
        "drop15": score_df.loc[score_df["keep_drop15"], "original_index"].to_numpy(dtype=int),
        "drop20": score_df.loc[score_df["keep_drop20"], "original_index"].to_numpy(dtype=int),
    }
    return {name: feature_sets[name] for name in BINARY_FEATURE_SETS}


def build_classifier(kind: str, binary: bool) -> Pipeline:
    if kind == "logreg":
        estimator = LogisticRegression(
            max_iter=2000,
            solver="liblinear" if binary else "lbfgs",
            class_weight="balanced",
        )
    elif kind == "linearsvm":
        estimator = LinearSVC(class_weight="balanced", dual="auto", max_iter=10000)
    elif kind == "ridge":
        estimator = RidgeClassifier(class_weight="balanced")
    else:
        raise ValueError(f"Unsupported classifier kind: {kind}")
    return Pipeline([("scaler", StandardScaler()), ("clf", estimator)])


def decision_scores(model: Pipeline, x: np.ndarray) -> np.ndarray:
    clf = model.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        proba = model.predict_proba(x)
        if proba.ndim == 2 and proba.shape[1] == 2:
            return proba[:, 1]
        return proba
    return model.decision_function(x)


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "f1": float(f1_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "specificity": float(specificity),
        "support_positive": int(y_true.sum()),
        "support_total": int(len(y_true)),
    }


def summarize_metric_dicts(metric_rows: list[dict[str, float]]) -> dict[str, float]:
    summary: dict[str, float] = {}
    keys = [key for key in metric_rows[0].keys() if key not in {"support_positive", "support_total"}]
    for key in keys:
        values = np.array([row[key] for row in metric_rows], dtype=float)
        summary[f"cv_{key}_mean"] = float(values.mean())
        summary[f"cv_{key}_std"] = float(values.std(ddof=0))
    summary["cv_support_positive_mean"] = float(np.mean([row["support_positive"] for row in metric_rows]))
    summary["cv_support_positive_std"] = float(np.std([row["support_positive"] for row in metric_rows], ddof=0))
    summary["cv_support_total_mean"] = float(np.mean([row["support_total"] for row in metric_rows]))
    summary["cv_support_total_std"] = float(np.std([row["support_total"] for row in metric_rows], ddof=0))
    return summary


def summarize_numeric_metric_dicts(metric_rows: list[dict[str, float]]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for key in metric_rows[0].keys():
        values = np.array([row[key] for row in metric_rows], dtype=float)
        summary[f"cv_{key}_mean"] = float(values.mean())
        summary[f"cv_{key}_std"] = float(values.std(ddof=0))
    return summary


def run_binary_probing(
    split_map: dict[str, pd.DataFrame],
    image_embeddings: dict[str, np.ndarray],
    text_embeddings: dict[str, np.ndarray],
    concepts_by_class: dict[str, list[str]],
    variance_scores: dict[str, pd.DataFrame],
    output_dir: Path,
    seed: int,
) -> None:
    reports_dir = output_dir / "reports"
    ensure_dir(reports_dir)
    train_df = split_map["train"].reset_index(drop=True)
    fold_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    fold_path = reports_dir / "binary_probe_fold_results.csv"
    summary_path = reports_dir / "binary_probe_results.csv"

    for class_name in CLASSES:
        concepts = concepts_by_class[class_name]
        raw_similarity = image_embeddings["train"] @ text_embeddings[class_name].T
        y = (train_df["class_name"] == class_name).astype(int).to_numpy()
        groups = train_df["group_id"].to_numpy()
        splitter = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
        fold_splits = list(splitter.split(np.zeros(len(y)), y, groups))
        fold_feature_sets: list[tuple[int, np.ndarray, np.ndarray, dict[str, np.ndarray]]] = []
        for fold_index, (fit_idx, eval_idx) in enumerate(fold_splits):
            fold_score_df, _, _, _ = build_variance_score_df(concepts, raw_similarity[fit_idx].T)
            fold_feature_sets.append((fold_index, fit_idx, eval_idx, build_binary_feature_sets(fold_score_df)))

        for feature_set_name in BINARY_FEATURE_SETS:
            for classifier_name in BINARY_CLASSIFIERS:
                fold_metrics: list[dict[str, float]] = []
                n_concepts_values: list[int] = []
                for fold_index, fit_idx, eval_idx, feature_sets in fold_feature_sets:
                    concept_indices = feature_sets[feature_set_name]
                    x_fit = raw_similarity[fit_idx][:, concept_indices]
                    x_eval = raw_similarity[eval_idx][:, concept_indices]
                    model = build_classifier(classifier_name, binary=True)
                    model.fit(x_fit, y[fit_idx])
                    y_pred = model.predict(x_eval)
                    y_score = decision_scores(model, x_eval)
                    metrics = compute_binary_metrics(y[eval_idx], y_pred, y_score)
                    fold_metrics.append(metrics)
                    n_concepts_values.append(int(len(concept_indices)))
                    fold_rows.append(
                        {
                            "target_class": class_name,
                            "feature_set": feature_set_name,
                            "n_concepts": int(len(concept_indices)),
                            "classifier": classifier_name,
                            "fold_index": fold_index,
                            **metrics,
                        }
                    )
                summary_rows.append(
                    {
                        "target_class": class_name,
                        "feature_set": feature_set_name,
                        "n_concepts": int(round(float(np.mean(n_concepts_values)))),
                        "n_concepts_std": float(np.std(n_concepts_values, ddof=0)),
                        "classifier": classifier_name,
                        **summarize_metric_dicts(fold_metrics),
                    }
                )
                pd.DataFrame(fold_rows).to_csv(fold_path, index=False)
                pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    pd.DataFrame(fold_rows).to_csv(fold_path, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)


def ordered_union_with_sources(
    concept_groups: dict[str, list[str]],
) -> tuple[list[str], dict[str, list[str]]]:
    ordered: list[str] = []
    sources: dict[str, list[str]] = defaultdict(list)
    seen: set[str] = set()
    for class_name in CLASSES:
        for concept in concept_groups[class_name]:
            sources[concept].append(class_name)
            if concept not in seen:
                seen.add(concept)
                ordered.append(concept)
    return ordered, sources


def build_multiclass_setting_concepts(
    concepts_by_class: dict[str, list[str]],
    variance_scores: dict[str, pd.DataFrame],
    additional_settings: dict[str, dict[str, list[str]]] | None = None,
) -> dict[str, dict[str, list[str]]]:
    settings: dict[str, dict[str, list[str]]] = {
        "raw_4000ish": {class_name: list(concepts_by_class[class_name]) for class_name in CLASSES}
    }
    for keep_name in MULTICLASS_FILTER_SETTINGS:
        settings[f"filtered_{keep_name}"] = {}
    for class_name in CLASSES:
        score_df = variance_scores[class_name]
        for keep_name in MULTICLASS_FILTER_SETTINGS:
            settings[f"filtered_{keep_name}"][class_name] = score_df.loc[
                score_df[f"keep_{keep_name}"], "concept"
            ].tolist()
    if additional_settings:
        settings.update(additional_settings)
    return settings


def select_binary_feature_sets(
    binary_results: pd.DataFrame,
    classifier_name: str,
    *,
    filtered_only: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    sort_keys = ["cv_f1_mean", "cv_balanced_accuracy_mean", "cv_pr_auc_mean", "cv_roc_auc_mean"]
    for class_name in CLASSES:
        class_rows = binary_results[
            (binary_results["target_class"] == class_name) & (binary_results["classifier"] == classifier_name)
        ].copy()
        if filtered_only:
            class_rows = class_rows[class_rows["feature_set"] != "raw"].copy()
        best_row = class_rows.sort_values(sort_keys, ascending=False).iloc[0]
        rows.append(
            {
                "target_class": class_name,
                "selector_classifier": classifier_name,
                "selected_feature_set": best_row["feature_set"],
                "selected_n_concepts": int(round(float(best_row["n_concepts"]))),
                "selected_n_concepts_std": float(best_row["n_concepts_std"]),
                "selected_cv_roc_auc_mean": float(best_row["cv_roc_auc_mean"]),
                "selected_cv_pr_auc_mean": float(best_row["cv_pr_auc_mean"]),
                "selected_cv_f1_mean": float(best_row["cv_f1_mean"]),
                "selected_cv_balanced_accuracy_mean": float(best_row["cv_balanced_accuracy_mean"]),
            }
        )
    return pd.DataFrame(rows)


def concepts_for_feature_set(score_df: pd.DataFrame, concepts: list[str], feature_set: str) -> list[str]:
    if feature_set == "raw":
        return list(concepts)
    mask_column = f"keep_{feature_set}"
    selected = score_df.loc[score_df[mask_column], ["original_index", "concept"]].sort_values("original_index")
    return selected["concept"].tolist()


def build_binary_selected_setting(
    selection_df: pd.DataFrame,
    variance_scores: dict[str, pd.DataFrame],
    concepts_by_class: dict[str, list[str]],
) -> dict[str, list[str]]:
    setting: dict[str, list[str]] = {}
    for row in selection_df.itertuples(index=False):
        setting[row.target_class] = concepts_for_feature_set(
            variance_scores[row.target_class],
            concepts_by_class[row.target_class],
            row.selected_feature_set,
        )
    return setting


def build_adaptive_multiclass_settings(
    binary_results: pd.DataFrame,
    variance_scores: dict[str, pd.DataFrame],
    concepts_by_class: dict[str, list[str]],
    reports_dir: Path,
) -> dict[str, dict[str, list[str]]]:
    settings: dict[str, dict[str, list[str]]] = {}
    selections_payload: dict[str, Any] = {}
    for classifier_name in ADAPTIVE_BINARY_SELECTOR_CLASSIFIERS:
        selection_df = select_binary_feature_sets(binary_results, classifier_name, filtered_only=False)
        selection_df.to_csv(reports_dir / f"binary_best_overall_{classifier_name}_selection.csv", index=False)
        setting_name = f"binarybest_overall_{classifier_name}"
        settings[setting_name] = build_binary_selected_setting(selection_df, variance_scores, concepts_by_class)
        selections_payload[setting_name] = settings[setting_name]
    save_json(reports_dir / "binary_best_overall_settings.json", selections_payload)
    return settings


def merged_text_embedding_matrix(
    setting_concepts: dict[str, list[str]],
    text_embeddings: dict[str, np.ndarray],
    concepts_by_class: dict[str, list[str]],
) -> tuple[np.ndarray, list[str], dict[str, list[str]]]:
    class_index_lookup = {
        class_name: {concept: idx for idx, concept in enumerate(concepts_by_class[class_name])}
        for class_name in CLASSES
    }
    ordered_concepts, sources = ordered_union_with_sources(setting_concepts)
    vectors: list[np.ndarray] = []
    for concept in ordered_concepts:
        source_classes = sources[concept]
        source_class = source_classes[0]
        concept_index = class_index_lookup[source_class][concept]
        vectors.append(text_embeddings[source_class][concept_index])
    return np.stack(vectors, axis=0), ordered_concepts, sources


def compute_multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    precision, recall, _, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
        zero_division=0,
    )
    if y_score.ndim != 2:
        raise ValueError("Multiclass y_score must be a 2D array.")
    row_sums = y_score.sum(axis=1)
    if np.allclose(row_sums, 1.0, atol=1e-5):
        macro_auroc = float(
            roc_auc_score(
                y_true,
                y_score,
                multi_class="ovr",
                average="macro",
                labels=np.arange(len(class_names)),
            )
        )
    else:
        y_true_bin = label_binarize(y_true, classes=np.arange(len(class_names)))
        macro_auroc = float(roc_auc_score(y_true_bin, y_score, average="macro"))
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "macro_auroc": macro_auroc,
    }
    for class_index, class_name in enumerate(class_names):
        metrics[f"precision_{class_name}"] = float(precision[class_index])
        metrics[f"recall_{class_name}"] = float(recall[class_index])
    return metrics


def run_multiclass_evaluation(
    split_map: dict[str, pd.DataFrame],
    image_embeddings: dict[str, np.ndarray],
    text_embeddings: dict[str, np.ndarray],
    concepts_by_class: dict[str, list[str]],
    variance_scores: dict[str, pd.DataFrame],
    output_dir: Path,
    seed: int,
) -> None:
    reports_dir = output_dir / "reports"
    ensure_dir(reports_dir)
    results_path = reports_dir / "multiclass_results.csv"
    fold_path = reports_dir / "multiclass_fold_results.csv"
    class_to_index = {class_name: idx for idx, class_name in enumerate(CLASSES)}
    train_df = split_map["train"].reset_index(drop=True)
    train_embeddings = image_embeddings["train"]
    y_train = train_df["class_name"].map(class_to_index).to_numpy()
    groups = train_df["group_id"].to_numpy()

    fold_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    binary_results = pd.read_csv(reports_dir / "binary_probe_results.csv")
    adaptive_settings = build_adaptive_multiclass_settings(binary_results, variance_scores, concepts_by_class, reports_dir)
    settings = build_multiclass_setting_concepts(concepts_by_class, variance_scores, adaptive_settings)
    save_json(reports_dir / "multiclass_feature_sets.json", settings)

    for setting_name, class_concepts in settings.items():
        merged_embeddings, merged_concepts, merged_sources = merged_text_embedding_matrix(
            class_concepts,
            text_embeddings,
            concepts_by_class,
        )
        save_json(
            reports_dir / f"{setting_name}_merged_concepts.json",
            [{"concept": concept, "source_classes": merged_sources[concept]} for concept in merged_concepts],
        )

    splitter = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
    fold_splits = list(splitter.split(np.zeros(len(y_train)), y_train, groups))
    metric_store: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    n_concepts_store: dict[tuple[str, str], list[int]] = defaultdict(list)
    oof_predictions: dict[tuple[str, str], dict[str, list[np.ndarray]]] = defaultdict(
        lambda: {"y_true": [], "y_pred": []}
    )

    for fold_index, (fit_idx, eval_idx) in enumerate(fold_splits):
        fold_variance_scores: dict[str, pd.DataFrame] = {}
        fit_embeddings = train_embeddings[fit_idx]
        for class_name in CLASSES:
            fold_score_df, _, _, _ = build_variance_score_df(
                concepts_by_class[class_name],
                text_embeddings[class_name] @ fit_embeddings.T,
            )
            fold_variance_scores[class_name] = fold_score_df
        # Adaptive settings are selected from train-only binary results and kept fixed during multiclass comparison.
        fold_settings = build_multiclass_setting_concepts(concepts_by_class, fold_variance_scores, adaptive_settings)

        y_fit = y_train[fit_idx]
        y_eval = y_train[eval_idx]
        for setting_name, class_concepts in fold_settings.items():
            merged_embeddings, merged_concepts, _ = merged_text_embedding_matrix(
                class_concepts,
                text_embeddings,
                concepts_by_class,
            )
            x_fit = train_embeddings[fit_idx] @ merged_embeddings.T
            x_eval = train_embeddings[eval_idx] @ merged_embeddings.T

            for classifier_name in MULTICLASS_CLASSIFIERS:
                model = build_classifier(classifier_name, binary=False)
                model.fit(x_fit, y_fit)
                y_pred = model.predict(x_eval)
                y_score = decision_scores(model, x_eval)
                metrics = compute_multiclass_metrics(y_eval, y_pred, y_score, CLASSES)
                key = (setting_name, classifier_name)
                metric_store[key].append(metrics)
                n_concepts_store[key].append(int(len(merged_concepts)))
                oof_predictions[key]["y_true"].append(y_eval)
                oof_predictions[key]["y_pred"].append(y_pred)
                fold_rows.append(
                    {
                        "setting": setting_name,
                        "classifier": classifier_name,
                        "split": "train_cv",
                        "fold_index": fold_index,
                        "n_concepts_total": int(len(merged_concepts)),
                        **metrics,
                    }
                )
                pd.DataFrame(fold_rows).to_csv(fold_path, index=False)

    for (setting_name, classifier_name), metric_rows in metric_store.items():
        n_values = n_concepts_store[(setting_name, classifier_name)]
        summary_rows.append(
            {
                "setting": setting_name,
                "classifier": classifier_name,
                "split": "train_cv",
                "n_concepts_total": int(round(float(np.mean(n_values)))),
                "n_concepts_total_std": float(np.std(n_values, ddof=0)),
                **summarize_numeric_metric_dicts(metric_rows),
            }
        )
        y_true = np.concatenate(oof_predictions[(setting_name, classifier_name)]["y_true"])
        y_pred = np.concatenate(oof_predictions[(setting_name, classifier_name)]["y_pred"])
        matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(CLASSES)))
        pd.DataFrame(matrix, index=CLASSES, columns=CLASSES).to_csv(
            reports_dir / f"confusion_matrix_{setting_name}_{classifier_name}_train_cv.csv"
        )
        pd.DataFrame(summary_rows).to_csv(results_path, index=False)

    pd.DataFrame(fold_rows).to_csv(fold_path, index=False)
    pd.DataFrame(summary_rows).to_csv(results_path, index=False)


def write_run_summary(
    output_dir: Path,
    concepts_by_class: dict[str, list[str]],
    split_map: dict[str, pd.DataFrame],
) -> None:
    payload = {
        "dataset_root": str(DATASET_ROOT),
        "annotations_path": str(ANNOTATIONS_PATH),
        "model_dir": str(MODEL_DIR),
        "concept_dir": str(CONCEPT_DIR),
        "classes": CLASSES,
        "n_cleaned_concepts_by_class": {class_name: len(concepts_by_class[class_name]) for class_name in CLASSES},
        "split_image_counts": {split_name: Counter(df["class_name"]) for split_name, df in split_map.items()},
        "split_group_counts": {
            split_name: df.groupby("class_name")["group_id"].nunique().to_dict()
            for split_name, df in split_map.items()
        },
        "fixed_prompt_templates": TEMPLATES,
        "notes": [
            "Official EBTC split is taken from annotations.csv sub_dataset.",
            "Static-cleaning baseline is the pre-variance concept bank used as the raw baseline.",
            "Merged multiclass settings deduplicate exact cross-class concept collisions by concept text.",
        ],
    }
    save_json(output_dir / "reports" / "run_summary.json", payload)


def run_pipeline(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    ensure_dir(args.output_dir)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=LinAlgWarning)

    concepts_by_class = clean_concepts(args.output_dir)
    manifest_df = build_manifest(args.output_dir)
    split_map = create_splits(manifest_df, args.output_dir, seed=args.seed)
    write_run_summary(args.output_dir, concepts_by_class, split_map)

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    model, tokenizer, preprocess = load_local_biomedclip(device=device, local_text_config_dir=args.local_text_config_dir)
    text_embeddings = encode_text_embeddings(
        concepts_by_class,
        model,
        tokenizer,
        device,
        args.output_dir,
        text_batch_size=args.text_batch_size,
    )
    image_embeddings = encode_image_embeddings(
        split_map,
        model,
        preprocess,
        device,
        args.output_dir,
        batch_size=args.image_batch_size,
    )
    variance_scores = variance_scoring(concepts_by_class, text_embeddings, image_embeddings["train"], args.output_dir)
    run_binary_probing(
        split_map,
        image_embeddings,
        text_embeddings,
        concepts_by_class,
        variance_scores,
        args.output_dir,
        seed=args.seed,
    )
    run_multiclass_evaluation(
        split_map,
        image_embeddings,
        text_embeddings,
        concepts_by_class,
        variance_scores,
        args.output_dir,
        seed=args.seed,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EBTC concept filtering experiment with BioMedCLIP.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--local-text-config-dir",
        type=Path,
        default=BIOMEDBERT_TEXT_CONFIG_DIR,
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--image-batch-size", type=int, default=16)
    parser.add_argument("--text-batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
