"""
================================================================================
predict.py — Phase 4: Standalone Inference Script (The Deliverable)
================================================================================

USAGE
-----
    python predict.py --image path/to/face.jpg
    python predict.py --image path/to/face.jpg --checkpoint checkpoints/best_model.pth
    python predict.py --image path/to/face.jpg --show

OUTPUT EXAMPLE
--------------
    ┌─────────────────────────────────────────────┐
    │  Deepfake Detection Result                  │
    │─────────────────────────────────────────────│
    │  File   : examples/face.jpg                 │
    │  Result : 🔴 FAKE                           │
    │  Score  : FAKE=98.2%  |  REAL=1.8%          │
    └─────────────────────────────────────────────┘

HOW INFERENCE WORKS
-------------------
  1. Load the image with PIL (handles JPEG, PNG, WebP, etc.)
  2. Apply the SAME deterministic transforms used during validation
     (resize → centrecrop → tensor → ImageNet normalise).
     ⚠️  Using training augmentation (random flips etc.) at inference time
         would produce different results on each run — never do this.
  3. Add a batch dimension: [C,H,W] → [1,C,H,W] (model expects batches)
  4. model(input) → logits of shape [1, 2]
  5. Softmax(logits) → probabilities that sum to 1.0
  6. argmax → predicted class index (0=FAKE, 1=REAL)
  7. Print confidence score and label

WHY SOFTMAX AT INFERENCE (not during training)?
  During training we use nn.CrossEntropyLoss which computes Softmax internally
  (more numerically stable).  At inference time we need the actual probabilities
  to report a confidence score, so we apply torch.softmax explicitly here.
================================================================================
"""

import os
import sys
import argparse

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# Ensure the project root is on the path for importing src.*
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.model   import build_model
from src.dataset import IMAGENET_MEAN, IMAGENET_STD, INPUT_SIZE


# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT PATHS
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CHECKPOINT = os.path.join("checkpoints", "best_model.pth")


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE TRANSFORM (DETERMINISTIC — identical to val transforms in dataset.py)
# ─────────────────────────────────────────────────────────────────────────────

def get_inference_transform() -> transforms.Compose:
    """
    Returns the exact same transform pipeline used during validation.

    CRITICAL: the transform must match what the model saw during training.
    Any difference (different mean/std, different crop strategy) will cause
    "distribution shift" — the model will see inputs it was never trained on,
    leading to poor predictions even from a well-trained model.
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        ),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_model(
    checkpoint_path: str,
    device: torch.device
) -> tuple[torch.nn.Module, dict]:
    """
    Load the model architecture and restore weights from a checkpoint file.

    The checkpoint dict saved during training contains:
      - model_state    : OrderedDict of layer weights
      - class_to_idx  : {'FAKE': 0, 'REAL': 1}
      - val_acc        : Best validation accuracy (useful to display)
      - epoch          : Which epoch produced the best model

    Returns
    -------
    model        : Restored model in evaluation mode
    class_to_idx : Label-to-integer mapping
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Please run 'python src/train.py' first to train and save a model."
        )

    print(f"[Predict] Loading checkpoint: {checkpoint_path}")

    # Load checkpoint dict (map_location ensures it loads on any device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Reconstruct the model architecture (must match what was trained)
    model = build_model(freeze_backbone=True)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()    # CRITICAL: disables Dropout for deterministic inference

    class_to_idx = checkpoint.get("class_to_idx", {"FAKE": 0, "REAL": 1})
    best_val_acc = checkpoint.get("val_acc", "N/A")
    best_epoch   = checkpoint.get("epoch", "N/A")

    print(f"[Predict] Loaded model from epoch {best_epoch} "
          f"(val accuracy = {best_val_acc:.1f}%)")

    return model, class_to_idx


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE-IMAGE PREDICTION
# ─────────────────────────────────────────────────────────────────────────────

def predict_image(
    image_path: str,
    model: torch.nn.Module,
    class_to_idx: dict,
    device: torch.device,
    show: bool = False,
) -> dict:
    """
    Run inference on a single image file and return a result dictionary.

    Args
    ----
    image_path   : Absolute or relative path to the image
    model        : Loaded model in eval mode
    class_to_idx : {'FAKE': 0, 'REAL': 1} mapping from training
    device       : Torch device
    show         : If True, open the image for visual inspection

    Returns
    -------
    result dict with keys: label, confidence, probabilities, image_path
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # ── Step 1: Load image ────────────────────────────────────────────────────
    # PIL is used because torchvision transforms expect PIL Images.
    # .convert("RGB") ensures we always have 3 channels even for grayscale/RGBA.
    image = Image.open(image_path).convert("RGB")

    if show:
        image.show(title=os.path.basename(image_path))

    # ── Step 2: Apply inference transforms ───────────────────────────────────
    transform = get_inference_transform()
    input_tensor = transform(image)     # shape: [3, 224, 224]

    # ── Step 3: Add batch dimension ───────────────────────────────────────────
    # PyTorch models always expect [Batch, Channels, Height, Width].
    # unsqueeze(0) inserts a size-1 batch dimension: [3,224,224] → [1,3,224,224]
    input_batch = input_tensor.unsqueeze(0).to(device)

    # ── Step 4: Forward pass (no gradients needed at inference) ───────────────
    with torch.no_grad():
        logits = model(input_batch)     # shape: [1, 2]

    # ── Step 5: Convert logits to probabilities via Softmax ───────────────────
    # Softmax ensures probabilities sum to 1.0 and are in range [0, 1].
    # dim=1 applies softmax across the class dimension (not batch dimension).
    probabilities = F.softmax(logits, dim=1).squeeze(0)  # shape: [2]

    # ── Step 6: Get predicted class ───────────────────────────────────────────
    predicted_idx = probabilities.argmax().item()

    # Invert the class_to_idx map: {0: 'FAKE', 1: 'REAL'}
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    predicted_label = idx_to_class[predicted_idx]
    confidence      = probabilities[predicted_idx].item() * 100

    # Build a human-readable probability dict
    prob_dict = {
        cls: probabilities[idx].item() * 100
        for cls, idx in class_to_idx.items()
    }

    return {
        "label"        : predicted_label,
        "confidence"   : confidence,
        "probabilities": prob_dict,
        "image_path"   : image_path,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PRETTY PRINTER
# ─────────────────────────────────────────────────────────────────────────────

def print_result(result: dict) -> None:
    """
    Print a formatted result box to stdout.

    Example output:
    ┌─────────────────────────────────────────────┐
    │  Deepfake Detection Result                  │
    │─────────────────────────────────────────────│
    │  File   : /path/to/face.jpg                 │
    │  Result : 🔴 FAKE                           │
    │  Score  : FAKE=98.2%  |  REAL=1.8%          │
    └─────────────────────────────────────────────┘
    """
    label      = result["label"]
    confidence = result["confidence"]
    probs      = result["probabilities"]
    path       = result["image_path"]

    icon   = "🔴" if label == "FAKE" else "🟢"
    width  = 47

    score_str = "  |  ".join(
        f"{cls}={prob:.1f}%" for cls, prob in probs.items()
    )

    print("\n" + "┌" + "─" * width + "┐")
    print(f"│  {'Deepfake Detection Result':<{width-2}}│")
    print("│" + "─" * width + "│")
    print(f"│  {'File':<8}: {os.path.basename(path):<{width-12}}│")
    print(f"│  {'Result':<8}: {icon} {label:<{width-13}}│")
    print(f"│  {'Score':<8}: {score_str:<{width-12}}│")
    print("└" + "─" * width + "┘\n")

    # Machine-readable single-line format (useful for piping to other scripts)
    print(f"Prediction: {label} ({confidence:.1f}% confidence)")


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Deepfake / Synthetic Image Detector — Inference Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py --image data/test/FAKE/0001.jpg
  python predict.py --image uploads/profile.png --show
  python predict.py --image face.jpg --checkpoint checkpoints/best_model.pth
        """
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the input image (JPEG, PNG, WebP, etc.)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help=f"Path to trained model checkpoint (default: {DEFAULT_CHECKPOINT})"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open the image for visual inspection before prediction"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Device selection ──────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # ── Load model ────────────────────────────────────────────────────────────
    model, class_to_idx = load_model(args.checkpoint, device)

    # ── Run prediction ────────────────────────────────────────────────────────
    result = predict_image(
        image_path   = args.image,
        model        = model,
        class_to_idx = class_to_idx,
        device       = device,
        show         = args.show,
    )

    # ── Display result ────────────────────────────────────────────────────────
    print_result(result)

    # Exit code: 1 = FAKE, 0 = REAL (allows scripting: if predict.py ...; then ...)
    sys.exit(1 if result["label"] == "FAKE" else 0)


if __name__ == "__main__":
    main()
