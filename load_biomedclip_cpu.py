#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from open_clip import create_model_and_transforms, get_tokenizer
from open_clip.factory import _MODEL_CONFIGS
from transformers import AutoConfig

from ebtc_project_paths import BIOMEDBERT_TEXT_CONFIG_DIR, MODEL_DIR, PROJECT_ROOT


def prepare_text_config_cache(source_model_name: str, local_text_config_dir: Path):
    local_text_config_dir.mkdir(parents=True, exist_ok=True)
    config_path = local_text_config_dir / "config.json"
    if not config_path.exists():
        AutoConfig.from_pretrained(source_model_name).save_pretrained(local_text_config_dir)


def load_local_biomedclip(model_dir: Path, local_text_config_dir: Path):
    config_path = model_dir / "open_clip_config.json"
    weights_path = model_dir / "open_clip_pytorch_model.bin"

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    prepare_text_config_cache(
        config["model_cfg"]["text_cfg"]["hf_model_name"],
        local_text_config_dir,
    )
    config["model_cfg"]["text_cfg"]["hf_model_name"] = str(local_text_config_dir)
    config["model_cfg"]["text_cfg"]["hf_tokenizer_name"] = str(model_dir)

    model_name = "biomedclip_local_cpu"
    if model_name not in _MODEL_CONFIGS:
        _MODEL_CONFIGS[model_name] = config["model_cfg"]
    else:
        _MODEL_CONFIGS[model_name] = config["model_cfg"]

    preprocess_cfg = config["preprocess_cfg"]
    model, _, preprocess = create_model_and_transforms(
        model_name=model_name,
        pretrained=str(weights_path),
        device="cpu",
        **{f"image_{k}": v for k, v in preprocess_cfg.items()},
    )
    tokenizer = get_tokenizer(model_name)
    return model.eval(), preprocess, tokenizer


def run_demo(model, preprocess, tokenizer, image_path: Path, labels: list[str]):
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0)
    texts = tokenizer([f"this is a photo of {label}" for label in labels], context_length=256)

    with torch.no_grad():
        image_features, text_features, logit_scale = model(image, texts)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        scores = (logit_scale * image_features @ text_features.T).softmax(dim=-1)[0]

    ranked = sorted(
        (
            {
                "label": label,
                "probability": float(prob),
            }
            for label, prob in zip(labels, scores.tolist())
        ),
        key=lambda item: item["probability"],
        reverse=True,
    )
    return ranked


def main():
    parser = argparse.ArgumentParser(description="Load a local BiomedCLIP checkpoint on CPU and run a small validation.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=MODEL_DIR,
        help="Directory containing open_clip_config.json and open_clip_pytorch_model.bin",
    )
    parser.add_argument(
        "--image-path",
        type=Path,
        default=MODEL_DIR / "example_data/biomed_image_classification_example_data/adenocarcinoma_histopathology.jpg",
        help="Image used for a quick CPU inference check.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=[
            "adenocarcinoma histopathology",
            "brain MRI",
            "covid line chart",
            "squamous cell carcinoma histopathology",
        ],
        help="Candidate labels for the quick zero-shot validation.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_ROOT / "biomedclip_cpu_check.json",
        help="Where to save the validation result.",
    )
    parser.add_argument(
        "--local-text-config-dir",
        type=Path,
        default=BIOMEDBERT_TEXT_CONFIG_DIR,
        help="Local cache directory for the text encoder config.json.",
    )
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    model, preprocess, tokenizer = load_local_biomedclip(args.model_dir, args.local_text_config_dir)
    ranked = run_demo(model, preprocess, tokenizer, args.image_path, args.labels)

    result = {
        "device": "cpu",
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "model_dir": str(args.model_dir),
        "local_text_config_dir": str(args.local_text_config_dir),
        "image_path": str(args.image_path),
        "top_prediction": ranked[0],
        "all_predictions": ranked,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
