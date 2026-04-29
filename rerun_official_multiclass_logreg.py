#!/home/kunet.ae/100069491/.conda/envs/torch/bin/python
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/home/kunet.ae/100069491/newcode")
OUTPUT_DIR = ROOT / "ebtc_concept_experiment_outputs"
sys.path.insert(0, str(ROOT))

import ebtc_concept_experiment as exp


def load_cached_artifacts() -> tuple[dict[str, pd.DataFrame], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, list[str]], dict[str, pd.DataFrame]]:
    split_map = {
        split_name: pd.read_csv(OUTPUT_DIR / "splits" / f"{split_name}.csv")
        for split_name in ["train", "val", "test"]
    }
    image_embeddings = {
        split_name: np.load(OUTPUT_DIR / "embeddings" / f"image_embeddings_{split_name}.npy")
        for split_name in ["train", "val", "test"]
    }
    text_embeddings = {
        class_name: np.load(OUTPUT_DIR / "embeddings" / f"text_embeddings_{class_name}.npy")
        for class_name in exp.CLASSES
    }
    concepts_by_class = {
        class_name: pd.read_csv(OUTPUT_DIR / "embeddings" / f"text_embeddings_{class_name}_concepts.csv")["concept"].tolist()
        for class_name in exp.CLASSES
    }
    variance_scores = {
        class_name: pd.read_csv(OUTPUT_DIR / "concept_scores" / f"{class_name}_variance_scores.csv")
        for class_name in exp.CLASSES
    }
    return split_map, image_embeddings, text_embeddings, concepts_by_class, variance_scores


def main() -> None:
    split_map, image_embeddings, text_embeddings, concepts_by_class, variance_scores = load_cached_artifacts()
    exp.MULTICLASS_CLASSIFIERS = ["logreg", "linearsvm"]
    exp.run_multiclass_evaluation(
        split_map=split_map,
        image_embeddings=image_embeddings,
        text_embeddings=text_embeddings,
        concepts_by_class=concepts_by_class,
        variance_scores=variance_scores,
        output_dir=OUTPUT_DIR,
        seed=exp.SEED,
    )


if __name__ == "__main__":
    main()
