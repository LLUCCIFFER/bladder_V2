from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold

import ebtc_concept_experiment as exp
from ebtc_project_paths import CONCEPT_EXPERIMENT_OUTPUT_DIR


OUTPUT_DIR = CONCEPT_EXPERIMENT_OUTPUT_DIR
REPORTS_DIR = OUTPUT_DIR / "reports"
FILTER_ONLY_FEATURE_SETS = [name for name in exp.BINARY_FEATURE_SETS if name != "raw"]
ADAPTIVE_MULTICLASS_CLASSIFIERS = ["logreg"]


def load_cached_artifacts() -> tuple[dict[str, pd.DataFrame], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, list[str]], dict[str, pd.DataFrame], pd.DataFrame]:
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
    binary_results = pd.read_csv(REPORTS_DIR / "binary_probe_results.csv")
    return split_map, image_embeddings, text_embeddings, concepts_by_class, variance_scores, binary_results


def select_best_filtered_settings(binary_results: pd.DataFrame, classifier_name: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    sort_keys = ["cv_roc_auc_mean", "cv_pr_auc_mean", "cv_f1_mean", "cv_balanced_accuracy_mean"]
    for class_name in exp.CLASSES:
        all_rows = binary_results[
            (binary_results["target_class"] == class_name) & (binary_results["classifier"] == classifier_name)
        ].copy()
        filtered_rows = all_rows[all_rows["feature_set"].isin(FILTER_ONLY_FEATURE_SETS)].copy()
        best_filtered = filtered_rows.sort_values(sort_keys, ascending=False).iloc[0]
        best_overall = all_rows.sort_values(sort_keys, ascending=False).iloc[0]
        rows.append(
            {
                "target_class": class_name,
                "selector_classifier": classifier_name,
                "best_filtered_feature_set": best_filtered["feature_set"],
                "best_filtered_n_concepts": int(round(float(best_filtered["n_concepts"]))),
                "best_filtered_n_concepts_std": float(best_filtered["n_concepts_std"]),
                "best_filtered_cv_roc_auc_mean": float(best_filtered["cv_roc_auc_mean"]),
                "best_filtered_cv_pr_auc_mean": float(best_filtered["cv_pr_auc_mean"]),
                "best_filtered_cv_f1_mean": float(best_filtered["cv_f1_mean"]),
                "best_filtered_cv_balanced_accuracy_mean": float(best_filtered["cv_balanced_accuracy_mean"]),
                "best_overall_feature_set": best_overall["feature_set"],
                "best_overall_n_concepts": int(round(float(best_overall["n_concepts"]))),
                "best_overall_cv_roc_auc_mean": float(best_overall["cv_roc_auc_mean"]),
                "best_overall_cv_pr_auc_mean": float(best_overall["cv_pr_auc_mean"]),
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
            row.best_filtered_feature_set,
        )
    return setting


def evaluate_multiclass_settings(
    settings: dict[str, dict[str, list[str]]],
    split_map: dict[str, pd.DataFrame],
    image_embeddings: dict[str, np.ndarray],
    text_embeddings: dict[str, np.ndarray],
    concepts_by_class: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = split_map["train"].reset_index(drop=True)
    class_to_index = {class_name: idx for idx, class_name in enumerate(exp.CLASSES)}
    y_all = train_df["class_name"].map(class_to_index).to_numpy()
    groups = train_df["group_id"].to_numpy()
    x_image = image_embeddings["train"]
    splitter = StratifiedGroupKFold(n_splits=exp.CV_FOLDS, shuffle=True, random_state=exp.SEED)

    fold_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for setting_name, class_concepts in settings.items():
        merged_embeddings, merged_concepts, merged_sources = exp.merged_text_embedding_matrix(
            class_concepts,
            text_embeddings,
            concepts_by_class,
        )
        with (REPORTS_DIR / f"{setting_name}_merged_concepts.json").open("w", encoding="utf-8") as f:
            json.dump(
                [{"concept": concept, "source_classes": merged_sources[concept]} for concept in merged_concepts],
                f,
                indent=2,
                ensure_ascii=False,
            )
        x_all = x_image @ merged_embeddings.T
        metric_store: dict[str, list[dict[str, float]]] = {name: [] for name in ADAPTIVE_MULTICLASS_CLASSIFIERS}
        predictions: dict[str, dict[str, list[np.ndarray]]] = {
            name: {"y_true": [], "y_pred": []} for name in ADAPTIVE_MULTICLASS_CLASSIFIERS
        }

        for fold_index, (fit_idx, eval_idx) in enumerate(splitter.split(np.zeros(len(y_all)), y_all, groups)):
            x_fit = x_all[fit_idx]
            x_eval = x_all[eval_idx]
            y_fit = y_all[fit_idx]
            y_eval = y_all[eval_idx]

            for classifier_name in ADAPTIVE_MULTICLASS_CLASSIFIERS:
                model = exp.build_classifier(classifier_name, binary=False)
                model.fit(x_fit, y_fit)
                y_pred = model.predict(x_eval)
                y_score = exp.decision_scores(model, x_eval)
                metrics = exp.compute_multiclass_metrics(y_eval, y_pred, y_score, exp.CLASSES)
                metric_store[classifier_name].append(metrics)
                predictions[classifier_name]["y_true"].append(y_eval)
                predictions[classifier_name]["y_pred"].append(y_pred)
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

        for classifier_name in ADAPTIVE_MULTICLASS_CLASSIFIERS:
            summary_rows.append(
                {
                    "setting": setting_name,
                    "classifier": classifier_name,
                    "split": "train_cv",
                    "n_concepts_total": int(len(merged_concepts)),
                    "n_concepts_total_std": 0.0,
                    **exp.summarize_numeric_metric_dicts(metric_store[classifier_name]),
                }
            )
            y_true = np.concatenate(predictions[classifier_name]["y_true"])
            y_pred = np.concatenate(predictions[classifier_name]["y_pred"])
            matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(exp.CLASSES)))
            pd.DataFrame(matrix, index=exp.CLASSES, columns=exp.CLASSES).to_csv(
                REPORTS_DIR / f"confusion_matrix_{setting_name}_{classifier_name}_train_cv.csv"
            )

    fold_df = pd.DataFrame(fold_rows)
    summary_df = pd.DataFrame(summary_rows)
    return fold_df, summary_df


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    split_map, image_embeddings, text_embeddings, concepts_by_class, variance_scores, binary_results = load_cached_artifacts()

    logreg_selection = select_best_filtered_settings(binary_results, "logreg")
    linearsvm_selection = select_best_filtered_settings(binary_results, "linearsvm")
    logreg_selection.to_csv(REPORTS_DIR / "binary_best_filtered_logreg_selection.csv", index=False)
    linearsvm_selection.to_csv(REPORTS_DIR / "binary_best_filtered_linearsvm_selection.csv", index=False)

    settings = {
        "filtered_binarybest_logreg": build_binary_selected_setting(logreg_selection, variance_scores, concepts_by_class),
    }
    with (REPORTS_DIR / "binary_selected_settings.json").open("w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2, ensure_ascii=False)

    fold_df, summary_df = evaluate_multiclass_settings(
        settings,
        split_map,
        image_embeddings,
        text_embeddings,
        concepts_by_class,
    )
    fold_df.to_csv(REPORTS_DIR / "multiclass_binary_selected_fold_results.csv", index=False)
    summary_df.to_csv(REPORTS_DIR / "multiclass_binary_selected_results.csv", index=False)

    existing_multiclass = pd.read_csv(REPORTS_DIR / "multiclass_results.csv")
    merged = pd.concat([existing_multiclass, summary_df], ignore_index=True)
    merged.to_csv(REPORTS_DIR / "multiclass_results_with_binary_selected.csv", index=False)


if __name__ == "__main__":
    main()
