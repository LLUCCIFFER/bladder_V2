#!/usr/bin/env python3
"""Class-weighting ablation for the refined-vector original-top10 CBM.

This script keeps the current best lightweight pipeline fixed:

    refined BioMedCLIP image vectors -> adapter -> concept activations
    -> linear classifier

Only the classification-loss class weights are changed. It is intended to test
whether class weighting improves the minority NTL class without changing the
concept bank or filtering logic.
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
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score

from ebtc_discriminative_whitelist_cycl import (
    AdapterConceptCBM,
    CLASSES,
    EPS,
    SplitPayload,
    evaluate_model,
    load_split_payloads,
    make_loader,
    profile_weighted_contrastive_loss,
    resolve_device,
    row_minmax_tensor,
    save_loss_curve,
    set_seed,
    softmax_np,
    summarize_results,
    write_csv,
    write_json,
)
from ebtc_project_paths import OFFICIAL_EMBEDDINGS_DIR, OUTPUT_ROOT


DEFAULT_REFINED_STAGE_DIR = OUTPUT_ROOT / "ebtc_embedding_refinement_stage_conservative_outputs"
DEFAULT_OUTPUT_DIR = OUTPUT_ROOT / "ebtc_class_weight_ablation_outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run class-weighting ablation on refined vectors + original top10.")
    parser.add_argument("--refined-stage-dir", type=Path, default=DEFAULT_REFINED_STAGE_DIR)
    parser.add_argument("--embeddings-dir", type=Path, default=OFFICIAL_EMBEDDINGS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--weight-modes", default="none,sqrt_inverse,inverse")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lambda-cycl", type=float, default=0.0)
    parser.add_argument("--lambda-align", type=float, default=0.10)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--num-threads", type=int, default=2)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_csv_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def load_refined_pipeline_payloads(refined_stage_dir: Path, embeddings_dir: Path) -> tuple[dict[str, SplitPayload], np.ndarray, np.ndarray]:
    """Load refined image embeddings and the original top10 refined concept bank."""

    split_payloads = load_split_payloads(embeddings_dir)
    adapter_dir = refined_stage_dir / "adapter_refinement"
    for split in ["train", "val", "test"]:
        split_payloads[split].embeddings = np.load(adapter_dir / f"refined_image_embeddings_{split}.npy").astype(np.float32)

    whitelist_path = refined_stage_dir / "refined_vectors_original_whitelist" / "whitelist_top10.npz"
    payload = np.load(whitelist_path, allow_pickle=True)
    concept_embeddings = payload["concept_embeddings"].astype(np.float32)
    m_matrix = payload["M"].astype(np.float32)
    return split_payloads, concept_embeddings, m_matrix


def class_weight_vector(labels: np.ndarray, mode: str, device: torch.device) -> torch.Tensor | None:
    """Return class weights for cross entropy.

    Modes:
    - none: no CE weighting
    - inverse: N / (C * N_c)
    - sqrt_inverse: [N / (C * N_c)] ** 0.5
    - powX: [N / (C * N_c)] ** X, for example pow0.75
    - ntl_boostX: no global weighting, but multiply NTL by X
    """

    mode = mode.strip().lower()
    if mode == "none":
        return None

    counts = np.bincount(labels, minlength=len(CLASSES)).astype(np.float32)
    inverse = counts.sum() / np.clip(len(CLASSES) * counts, EPS, None)

    if mode == "inverse":
        weights = inverse
    elif mode == "sqrt_inverse":
        weights = inverse**0.5
    elif mode.startswith("pow"):
        alpha = float(mode.replace("pow", ""))
        weights = inverse**alpha
    elif mode.startswith("ntl_boost"):
        factor = float(mode.replace("ntl_boost", ""))
        weights = np.ones(len(CLASSES), dtype=np.float32)
        weights[CLASSES.index("NTL")] = factor
    else:
        raise ValueError(f"Unsupported class weight mode: {mode}")

    return torch.tensor(weights, dtype=torch.float32, device=device)


def describe_weights(labels: np.ndarray, mode: str) -> dict[str, Any]:
    weights = class_weight_vector(labels, mode, torch.device("cpu"))
    counts = np.bincount(labels, minlength=len(CLASSES)).astype(int)
    row: dict[str, Any] = {"weight_mode": mode}
    for idx, class_name in enumerate(CLASSES):
        row[f"{class_name}_train_count"] = int(counts[idx])
        row[f"{class_name}_weight"] = float(weights[idx]) if weights is not None else 1.0
    return row


def train_one_weight_mode(
    split_payloads: dict[str, SplitPayload],
    concept_embeddings: np.ndarray,
    m_matrix: np.ndarray,
    seed: int,
    weight_mode: str,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    set_seed(seed)
    device = resolve_device(args.device)
    model = AdapterConceptCBM(concept_embeddings, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    class_weights = class_weight_vector(split_payloads["train"].labels, weight_mode, device)
    class_profiles = torch.tensor(m_matrix.T, dtype=torch.float32, device=device)
    train_loader = make_loader(split_payloads["train"], batch_size=args.batch_size, shuffle=True)

    best_state: dict[str, torch.Tensor] | None = None
    best_val_f1 = -1.0
    best_epoch = 0
    patience_counter = 0
    curve_rows: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = {"total": 0.0, "cls": 0.0, "cycl": 0.0, "align": 0.0}
        seen = 0
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            outputs = model(features)
            loss_cls = F.cross_entropy(outputs["logits"], labels, weight=class_weights)
            activation_norm = row_minmax_tensor(outputs["concept_activations"])
            loss_align = F.mse_loss(activation_norm, class_profiles[labels])
            loss_cycl = profile_weighted_contrastive_loss(outputs["adapted_embeddings"], labels, class_profiles, tau=args.tau)
            loss = loss_cls + args.lambda_cycl * loss_cycl + args.lambda_align * loss_align
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_n = int(labels.numel())
            seen += batch_n
            running["total"] += float(loss.detach().cpu()) * batch_n
            running["cls"] += float(loss_cls.detach().cpu()) * batch_n
            running["cycl"] += float(loss_cycl.detach().cpu()) * batch_n
            running["align"] += float(loss_align.detach().cpu()) * batch_n

        train_loss = {key: value / max(1, seen) for key, value in running.items()}
        val_metrics = evaluate_model(model, split_payloads["val"], device=device, batch_size=args.batch_size)
        curve_rows.append(
            {
                "weight_mode": weight_mode,
                "seed": seed,
                "epoch": epoch,
                "train_total_loss": train_loss["total"],
                "train_cls_loss": train_loss["cls"],
                "train_cycl_loss": train_loss["cycl"],
                "train_align_loss": train_loss["align"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_macro_auroc": val_metrics["macro_auroc"],
            }
        )
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = float(val_metrics["macro_f1"])
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= args.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    seed_dir = output_dir / weight_mode / "training" / "top10" / f"seed_{seed}"
    ensure_dir(seed_dir)
    write_csv(seed_dir / "training_curve.csv", curve_rows)
    save_loss_curve(seed_dir / "loss_curve.png", curve_rows, title=f"{weight_mode} seed {seed}")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "weight_mode": weight_mode,
            "seed": seed,
            "concepts": int(concept_embeddings.shape[0]),
            "classes": CLASSES,
            "args": vars(args),
        },
        seed_dir / "best_checkpoint.pt",
    )

    val_metrics = evaluate_model(model, split_payloads["val"], device=device, batch_size=args.batch_size)
    test_metrics = evaluate_model(model, split_payloads["test"], device=device, batch_size=args.batch_size)
    result: dict[str, Any] = {
        "weight_mode": weight_mode,
        "top_k": 10,
        "seed": seed,
        "n_concepts": int(concept_embeddings.shape[0]),
        "best_epoch": best_epoch,
        "device": str(device),
        "lambda_cycl": args.lambda_cycl,
        "lambda_align": args.lambda_align,
        "val_accuracy": val_metrics["accuracy"],
        "val_macro_f1": val_metrics["macro_f1"],
        "val_macro_auroc": val_metrics["macro_auroc"],
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_macro_auroc": test_metrics["macro_auroc"],
    }
    for class_name in CLASSES:
        result[f"test_{class_name}_precision"] = test_metrics[f"{class_name}_precision"]
        result[f"test_{class_name}_recall"] = test_metrics[f"{class_name}_recall"]
        result[f"test_{class_name}_f1"] = test_metrics[f"{class_name}_f1"]
    write_json(seed_dir / "metrics.json", result)
    return result


def evaluate_ensemble(
    output_dir: Path,
    weight_mode: str,
    split_payloads: dict[str, SplitPayload],
    concept_embeddings: np.ndarray,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    checkpoint_paths = sorted((output_dir / weight_mode / "training" / "top10").glob("seed_*/best_checkpoint.pt"))
    models: list[AdapterConceptCBM] = []
    for checkpoint_path in checkpoint_paths:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = AdapterConceptCBM(concept_embeddings, hidden_dim=args.hidden_dim, dropout=args.dropout)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        models.append(model)

    rows: list[dict[str, Any]] = []
    confusion_payload: dict[str, Any] = {}
    for split in ["val", "test"]:
        features = torch.tensor(split_payloads[split].embeddings, dtype=torch.float32)
        labels = split_payloads[split].labels
        all_probs: list[np.ndarray] = []
        with torch.no_grad():
            for model in models:
                logits = model(features)["logits"].numpy()
                all_probs.append(softmax_np(logits))
        probs = np.mean(all_probs, axis=0)
        preds = probs.argmax(axis=1)
        row: dict[str, Any] = {
            "weight_mode": weight_mode,
            "split": split,
            "n_models": len(models),
            "n_concepts": int(concept_embeddings.shape[0]),
            "accuracy": float(accuracy_score(labels, preds)),
            "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
            "macro_auroc": float(roc_auc_score(labels, probs, multi_class="ovr", average="macro", labels=list(range(len(CLASSES))))),
        }
        precision, recall, f1, support = precision_recall_fscore_support(labels, preds, labels=list(range(len(CLASSES))), zero_division=0)
        for idx, class_name in enumerate(CLASSES):
            row[f"{class_name}_precision"] = float(precision[idx])
            row[f"{class_name}_recall"] = float(recall[idx])
            row[f"{class_name}_f1"] = float(f1[idx])
            row[f"{class_name}_support"] = int(support[idx])
        rows.append(row)
        confusion_payload[split] = confusion_matrix(labels, preds, labels=list(range(len(CLASSES)))).tolist()

    ensemble_dir = output_dir / weight_mode / "ensemble_3seed"
    ensure_dir(ensemble_dir)
    write_csv(ensemble_dir / "ensemble_metrics.csv", rows)
    write_json(
        ensemble_dir / "ensemble_manifest.json",
        {
            "weight_mode": weight_mode,
            "checkpoint_paths": [str(path) for path in checkpoint_paths],
            "n_models": len(models),
        },
    )
    write_json(ensemble_dir / "ensemble_confusion_matrices.json", confusion_payload)
    return rows


def write_report(output_dir: Path, seed_summary_rows: list[dict[str, Any]], ensemble_rows: list[dict[str, Any]], weight_rows: list[dict[str, Any]]) -> None:
    seed_df = {row["weight_mode"]: row for row in seed_summary_rows}
    test_ensemble = [row for row in ensemble_rows if row["split"] == "test"]
    best = max(test_ensemble, key=lambda row: float(row["macro_f1"])) if test_ensemble else None

    lines = [
        "# Class Weighting Ablation",
        "",
        "## Fixed Setting",
        "",
        "- Bank: refined vectors + original top10 whitelist.",
        "- Classifier: concept activations -> linear classifier.",
        "- Loss: `CE + lambda_cycl * CyCL + lambda_align * align`.",
        "- This ablation changes only the CE class-weighting strategy.",
        "",
        "## Class Weights",
        "",
        "| Mode | HGC | LGC | NTL | NST |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in weight_rows:
        lines.append(
            f"| {row['weight_mode']} | {float(row['HGC_weight']):.4f} | {float(row['LGC_weight']):.4f} | "
            f"{float(row['NTL_weight']):.4f} | {float(row['NST_weight']):.4f} |"
        )

    lines.extend(["", "## Seed-Mean Results", "", "| Mode | Val Macro-F1 | Test Acc | Test Macro-F1 | Test AUROC | NTL F1 |", "|---|---:|---:|---:|---:|---:|"])
    for mode, row in seed_df.items():
        lines.append(
            f"| {mode} | {float(row['val_macro_f1_mean']):.4f} ± {float(row['val_macro_f1_std']):.4f} | "
            f"{float(row['test_accuracy_mean']):.4f} ± {float(row['test_accuracy_std']):.4f} | "
            f"{float(row['test_macro_f1_mean']):.4f} ± {float(row['test_macro_f1_std']):.4f} | "
            f"{float(row['test_macro_auroc_mean']):.4f} ± {float(row['test_macro_auroc_std']):.4f} | "
            f"{float(row.get('test_NTL_f1_mean', 0.0)):.4f} |"
        )

    lines.extend(["", "## 3-Seed Ensemble Test Results", "", "| Mode | Test Acc | Test Macro-F1 | Test AUROC | HGC F1 | LGC F1 | NTL F1 | NST F1 |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for row in test_ensemble:
        lines.append(
            f"| {row['weight_mode']} | {float(row['accuracy']):.4f} | {float(row['macro_f1']):.4f} | "
            f"{float(row['macro_auroc']):.4f} | {float(row['HGC_f1']):.4f} | {float(row['LGC_f1']):.4f} | "
            f"{float(row['NTL_f1']):.4f} | {float(row['NST_f1']):.4f} |"
        )

    if best is not None:
        lines.extend(
            [
                "",
                "## Conclusion",
                "",
                f"- Best ensemble Macro-F1 in this ablation: `{best['weight_mode']}`.",
                f"- Test Macro-F1: `{float(best['macro_f1']):.4f}`.",
                f"- Test NTL F1: `{float(best['NTL_f1']):.4f}`.",
                "- Select the final weighting by validation Macro-F1 / NTL tradeoff, not by test-only behavior.",
            ]
        )

    (output_dir / "class_weight_ablation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
            "resolved_device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
        },
    )

    split_payloads, concept_embeddings, m_matrix = load_refined_pipeline_payloads(args.refined_stage_dir, args.embeddings_dir)
    weight_modes = parse_csv_list(args.weight_modes)
    seeds = parse_int_list(args.seeds)

    weight_rows = [describe_weights(split_payloads["train"].labels, mode) for mode in weight_modes]
    write_csv(args.output_dir / "class_weight_values.csv", weight_rows)

    seed_rows: list[dict[str, Any]] = []
    for mode in weight_modes:
        for seed in seeds:
            print(f"[run] mode={mode} seed={seed} device={device}", flush=True)
            result = train_one_weight_mode(split_payloads, concept_embeddings, m_matrix, seed, mode, args, args.output_dir)
            seed_rows.append(result)
    write_csv(args.output_dir / "class_weight_seed_results.csv", seed_rows)

    summary_rows: list[dict[str, Any]] = []
    for mode in weight_modes:
        mode_rows = [row for row in seed_rows if row["weight_mode"] == mode]
        summary = summarize_results(mode_rows)
        for row in summary:
            row["weight_mode"] = mode
            for class_name in CLASSES:
                values = [float(item[f"test_{class_name}_f1"]) for item in mode_rows]
                row[f"test_{class_name}_f1_mean"] = float(np.mean(values))
                row[f"test_{class_name}_f1_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            summary_rows.append(row)
    write_csv(args.output_dir / "class_weight_results_summary.csv", summary_rows)

    ensemble_rows: list[dict[str, Any]] = []
    for mode in weight_modes:
        ensemble_rows.extend(evaluate_ensemble(args.output_dir, mode, split_payloads, concept_embeddings, args))
    write_csv(args.output_dir / "class_weight_ensemble_results.csv", ensemble_rows)
    write_report(args.output_dir, summary_rows, ensemble_rows, weight_rows)
    print(json.dumps({"summary": summary_rows, "ensemble": ensemble_rows}, indent=2))


if __name__ == "__main__":
    main()
