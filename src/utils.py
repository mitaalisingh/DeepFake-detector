"""
================================================================================
src/utils.py — Shared Utility Functions
================================================================================

PURPOSE
-------
Centralises helpers that are called from multiple scripts:
  - Reproducibility seed setting
  - Training metric plotting
  - Classification report generation

Keeping utilities here avoids code duplication and makes the individual
phase scripts (train.py, predict.py) cleaner and easier to read.
================================================================================
"""

import os
import random

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix


# ─────────────────────────────────────────────────────────────────────────────
# 1. REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """
    Fix random seeds across ALL sources of randomness so experiments are
    exactly reproducible when run with the same seed.

    Sources of randomness in a PyTorch training run:
      - Python's built-in random module (used by ImageFolder shuffle)
      - NumPy (used internally by some transforms)
      - PyTorch CPU operations
      - PyTorch CUDA operations (two separate seeds)
      - cuDNN backend (setting deterministic=True trades a tiny speed penalty
        for bit-exact reproducibility on GPU)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)                    # for multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False              # disable auto-tuner


# ─────────────────────────────────────────────────────────────────────────────
# 2. TRAINING CURVE PLOTTER
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(
    train_losses: list[float],
    val_losses: list[float],
    train_accs: list[float],
    val_accs: list[float],
    save_path: str = "logs/training_curves.png"
) -> None:
    """
    Save a 1×2 figure containing:
      Left  → Train vs. Validation Loss per epoch
      Right → Train vs. Validation Accuracy per epoch

    These curves are essential for the project report to demonstrate:
      - Whether the model converged
      - Whether early stopping triggered at the right time
      - Whether there is evidence of over/underfitting

    Args
    ----
    train_losses : list of average training losses per epoch
    val_losses   : list of average validation losses per epoch
    train_accs   : list of training accuracy values (0–100) per epoch
    val_accs     : list of validation accuracy values (0–100) per epoch
    save_path    : where to save the PNG (directory must exist)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    epochs = range(1, len(train_losses) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Loss subplot ──────────────────────────────────────────────────────────
    axes[0].plot(epochs, train_losses, "b-o", label="Train Loss", markersize=4)
    axes[0].plot(epochs, val_losses,   "r-o", label="Val Loss",   markersize=4)
    axes[0].set_title("Loss per Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # ── Accuracy subplot ──────────────────────────────────────────────────────
    axes[1].plot(epochs, train_accs, "b-o", label="Train Acc", markersize=4)
    axes[1].plot(epochs, val_accs,   "r-o", label="Val Acc",   markersize=4)
    axes[1].set_title("Accuracy per Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_ylim(0, 100)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.suptitle("MobileNetV2 — CIFAKE Training Curves", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Utils] Training curves saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. EVALUATION METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    all_labels: list[int],
    all_preds: list[int],
    class_names: list[str] = ["FAKE", "REAL"]
) -> None:
    """
    Print a detailed per-class evaluation report after training completes.

    Metrics explained for the project report:
      - Precision  : Of all images the model labelled FAKE, what fraction truly are?
                     High precision = few false alarms.
      - Recall     : Of all truly FAKE images, what fraction did the model catch?
                     High recall = few missed fakes.
      - F1-Score   : Harmonic mean of Precision & Recall. Best single-number summary.
      - Support    : Number of ground-truth samples per class in the test set.

    WHY F1 MATTERS HERE
    -------------------
    In a real-world scenario (e.g., a dating site flagging fake profiles), missing
    a fake (low recall) is more harmful than a false alarm (low precision).
    F1 balances both, making it the primary metric to report.

    Args
    ----
    all_labels  : Ground-truth integer class indices (0 or 1)
    all_preds   : Model-predicted class indices (0 or 1)
    class_names : Human-readable names matching the index order
    """
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(all_labels, all_preds, target_names=class_names))

    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix (rows=actual, cols=predicted):")
    print(f"          {class_names[0]:>6}  {class_names[1]:>6}")
    for i, row in enumerate(cm):
        print(f"{class_names[i]:>10}  {row[0]:>6}  {row[1]:>6}")
    print("=" * 60 + "\n")
