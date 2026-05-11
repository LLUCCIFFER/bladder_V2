#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ebtc_project_paths import CONCEPT_EXPERIMENT_OUTPUT_DIR

OUTPUT_ROOT = CONCEPT_EXPERIMENT_OUTPUT_DIR
REPORTS_DIR = OUTPUT_ROOT / "reports"
EXPORT_DIR = OUTPUT_ROOT / "final_concept_banks_officialsplit"
CLASSES = ["HGC", "LGC", "NTL", "NST"]
SETTINGS = ["filtered_top700", "filtered_top300", "binarybest_overall_logreg"]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: object) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_feature_sets() -> dict[str, dict[str, list[str]]]:
    data = json.loads((REPORTS_DIR / "multiclass_feature_sets.json").read_text(encoding="utf-8"))
    return {setting: data[setting] for setting in SETTINGS}


def export_concept_banks() -> list[dict[str, object]]:
    settings = load_feature_sets()
    summary_rows: list[dict[str, object]] = []
    ensure_dir(EXPORT_DIR)

    for setting_name, class_map in settings.items():
        setting_dir = EXPORT_DIR / setting_name
        ensure_dir(setting_dir)
        merged_rows: list[dict[str, object]] = []

        for class_name in CLASSES:
            concepts = class_map[class_name]
            (setting_dir / f"{class_name}.txt").write_text("\n".join(concepts) + "\n", encoding="utf-8")
            pd.DataFrame(
                {
                    "class_name": class_name,
                    "concept_rank": range(1, len(concepts) + 1),
                    "concept": concepts,
                }
            ).to_csv(setting_dir / f"{class_name}.csv", index=False)
            summary_rows.append(
                {
                    "setting": setting_name,
                    "class_name": class_name,
                    "n_concepts": len(concepts),
                }
            )
            merged_rows.extend(
                {
                    "class_name": class_name,
                    "concept_rank_within_class": idx,
                    "concept": concept,
                }
                for idx, concept in enumerate(concepts, start=1)
            )

        pd.DataFrame(merged_rows).to_csv(setting_dir / "merged_concepts.csv", index=False)
        write_json(
            setting_dir / "metadata.json",
            {
                "setting": setting_name,
                "classes": CLASSES,
                "n_concepts_by_class": {class_name: len(class_map[class_name]) for class_name in CLASSES},
                "n_concepts_total": sum(len(class_map[class_name]) for class_name in CLASSES),
                "source_reports": {
                    "multiclass_feature_sets": str(REPORTS_DIR / "multiclass_feature_sets.json"),
                    "binary_selection_logreg": str(REPORTS_DIR / "binary_best_overall_logreg_selection.csv"),
                },
            },
        )

    pd.DataFrame(summary_rows).to_csv(EXPORT_DIR / "concept_bank_summary.csv", index=False)
    return summary_rows


def build_paper_style_summary() -> str:
    split_df = pd.read_csv(REPORTS_DIR / "split_summary.csv")
    binary_df = pd.read_csv(REPORTS_DIR / "binary_probe_results.csv")
    multiclass_df = pd.read_csv(REPORTS_DIR / "multiclass_results.csv")
    binary_best_logreg = pd.read_csv(REPORTS_DIR / "binary_best_overall_logreg_selection.csv")
    binary_best_linearsvm = pd.read_csv(REPORTS_DIR / "binary_best_overall_linearsvm_selection.csv")

    lines: list[str] = []
    lines.append("# EBTC Official-Split Concept Filtering Summary")
    lines.append("")
    lines.append("## Methods")
    lines.append("")
    lines.append(
        "We used the official EBTC split defined by `annotations.csv` (`sub_dataset`) and kept the official "
        "`train/val/test` partition fixed throughout the experiment."
    )
    lines.append("")
    lines.append("Dataset composition:")
    for row in split_df.itertuples(index=False):
        lines.append(
            f"- {row.split}: {row.class_name} images = {row.image_count}, groups = {row.group_count}"
        )
    lines.append("")
    lines.append(
        "All concept cleaning, variance-based ranking, binary probing, and multiclass comparison were performed "
        "using train-only 5-fold StratifiedGroupKFold cross-validation on the official training split."
    )
    lines.append("")
    lines.append(
        "For each fold, variance scores were recomputed on the fold-specific training portion only, and the "
        "resulting concept subsets were then evaluated on the held-out fold. Binary tasks were defined as "
        "one-vs-rest for HGC, LGC, NTL, and NST. Multiclass tasks used merged concept banks and were evaluated "
        "with multinomial logistic regression and linear SVM."
    )
    lines.append("")
    lines.append(
        "Adaptive `binarybest` concept banks were derived from binary probing results by ranking feature sets with "
        "the following priority: cross-validated F1, balanced accuracy, PR-AUC, and ROC-AUC."
    )
    lines.append("")
    lines.append("## Binary Probing Results")
    lines.append("")
    lines.append("Best overall binary setting per class for logistic regression:")
    for row in binary_best_logreg.itertuples(index=False):
        lines.append(
            f"- {row.target_class}: `{row.selected_feature_set}` ({row.selected_n_concepts} concepts), "
            f"F1 = {row.selected_cv_f1_mean:.4f}, balanced accuracy = {row.selected_cv_balanced_accuracy_mean:.4f}, "
            f"PR-AUC = {row.selected_cv_pr_auc_mean:.4f}, ROC-AUC = {row.selected_cv_roc_auc_mean:.4f}"
        )
    lines.append("")
    lines.append("Best overall binary setting per class for linear SVM:")
    for row in binary_best_linearsvm.itertuples(index=False):
        lines.append(
            f"- {row.target_class}: `{row.selected_feature_set}` ({row.selected_n_concepts} concepts), "
            f"F1 = {row.selected_cv_f1_mean:.4f}, balanced accuracy = {row.selected_cv_balanced_accuracy_mean:.4f}, "
            f"PR-AUC = {row.selected_cv_pr_auc_mean:.4f}, ROC-AUC = {row.selected_cv_roc_auc_mean:.4f}"
        )
    lines.append("")
    lines.append(
        "Across both binary classifiers, HGC consistently favored `top300`, LGC favored `top700`, NTL favored "
        "`raw`, and NST preferred either `top300` (logistic regression) or `raw` (linear SVM). This indicates "
        "that the four class-specific concept banks do not share a single optimal compression ratio."
    )
    lines.append("")
    lines.append("## Multiclass Results")
    lines.append("")
    for classifier_name in ["logreg", "linearsvm"]:
        sub = multiclass_df[multiclass_df["classifier"] == classifier_name].sort_values(
            "cv_macro_f1_mean", ascending=False
        )
        lines.append(f"Top multiclass settings for `{classifier_name}`:")
        for row in sub.head(5).itertuples(index=False):
            lines.append(
                f"- {row.setting}: {row.n_concepts_total} concepts, accuracy = {row.cv_accuracy_mean:.4f}, "
                f"macro-F1 = {row.cv_macro_f1_mean:.4f}, macro-AUROC = {row.cv_macro_auroc_mean:.4f}"
            )
        lines.append("")

    lines.append(
        "For logistic regression, the best overall multiclass representation was `filtered_top700` "
        "(2800 concepts; accuracy 0.6921, macro-F1 0.6458), outperforming both the raw baseline "
        "(`raw_4000ish`, 5544 concepts; accuracy 0.6873, macro-F1 0.6373) and the binary-guided adaptive banks."
    )
    lines.append("")
    lines.append(
        "For linear SVM, the best overall multiclass representation was `filtered_top300` "
        "(1200 concepts; accuracy 0.7119, macro-F1 0.6733). The strongest adaptive bank under linear SVM was "
        "`binarybest_overall_logreg` (2640 concepts; accuracy 0.6969, macro-F1 0.6631), which improved over "
        "the raw baseline (`raw_4000ish`, accuracy 0.6897, macro-F1 0.6472) but still did not exceed "
        "`filtered_top300`."
    )
    lines.append("")
    lines.append(
        "These results show that concept filtering remained beneficial under the official EBTC split. However, "
        "the per-class concept subsets selected from binary probing did not yield the strongest joint multiclass "
        "representation after merging. In both classifier families, the best-performing multiclass banks were "
        "still obtained from a unified global filtering rule rather than from a class-wise adaptive binarybest rule."
    )
    lines.append("")
    lines.append("## Practical Recommendation")
    lines.append("")
    lines.append(
        "- `filtered_top700` should be retained as the main logistic-regression concept bank for the official split."
    )
    lines.append(
        "- `filtered_top300` should be retained as the main linear-SVM concept bank and as the compact explainability-oriented bank."
    )
    lines.append(
        "- `binarybest_overall_logreg` should be kept as an adaptive comparison bank because it is the strongest binary-guided setting in multiclass evaluation."
    )
    lines.append("")
    lines.append(
        "The three exported final concept banks are therefore `filtered_top700`, `filtered_top300`, and "
        "`binarybest_overall_logreg`."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    export_concept_banks()
    summary_text = build_paper_style_summary()
    summary_path = REPORTS_DIR / "officialsplit_paper_style_summary.md"
    summary_path.write_text(summary_text, encoding="utf-8")


if __name__ == "__main__":
    main()
