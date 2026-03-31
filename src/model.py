"""
================================================================================
src/model.py — Phase 2: Model Architecture (Transfer Learning)
================================================================================

WHY TRANSFER LEARNING?
----------------------
Training a deep CNN from scratch for binary image classification requires
millions of labelled examples and days of compute.  Transfer learning lets
us "steal" general visual features (edges, textures, shapes) already learned
by a model trained on ImageNet-1K (1.28 million images, 1000 classes) and
adapt only the final decision layer to our specific FAKE vs. REAL task.

This is analogous to hiring a painter who already knows how to mix colours
(ImageNet features) and just teaching them the specific style you want
(deepfake detection).

WHY MOBILENETV2?
----------------
MobileNetV2 (Howard et al., 2018) is designed for efficiency without
sacrificing accuracy.  Key design principles that make it ideal here:

  1. Inverted Residuals + Depthwise Separable Convolutions
       Standard convolution = O(k² × C_in × C_out) multiplications per pixel.
       Depthwise separable = O(k² × C_in + C_in × C_out) — up to 9× cheaper
       for a 3×3 kernel, with minimal accuracy loss.

  2. Linear Bottlenecks
       Activations in the bottleneck are kept LINEAR (no ReLU) to prevent
       information destruction in low-dimensional feature spaces.

  3. Only ~3.4M parameters (vs. ResNet-50's ~25M)
       Fits comfortably on a laptop GPU or even CPU for inference — critical
       for a real-world deployment on a web server.

  4. Strong accuracy on image classification benchmarks
       Top-1 accuracy of ~72% on ImageNet when trained from scratch, and much
       higher when fine-tuned on domain-specific data like CIFAKE.

ALTERNATIVE CONSIDERED: ResNet-18
  ResNet-18 (~11M params) is also a valid choice and widely used in research.
  MobileNetV2 was preferred here for its smaller footprint while retaining
  comparable accuracy on binary classification tasks.

CUSTOM CLASSIFICATION HEAD
--------------------------
The original MobileNetV2 classifier was designed for 1000 ImageNet classes.
We replace it with:

    Linear(1280 → 512)  →  ReLU  →  Dropout(0.4)  →  Linear(512 → 2)
              ↑                           ↑                   ↑
        large feature          regularisation          FAKE / REAL logits
           vector from         prevents co-
           backbone            adaptation of neurons

Dropout(0.4) randomly zeros 40% of neurons during training, forcing the
network to learn redundant representations — a key defence against overfitting.
================================================================================
"""

import torch
import torch.nn as nn
from torchvision import models


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

NUM_CLASSES   = 2          # FAKE (0) and REAL (1)
DROPOUT_RATE  = 0.4        # fraction of neurons dropped during training
HIDDEN_DIM    = 512        # intermediate layer width in the custom head
BACKBONE_OUT  = 1280       # MobileNetV2's final feature vector dimension


# ─────────────────────────────────────────────────────────────────────────────
# MODEL BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_model(
    freeze_backbone: bool = True,
    num_classes: int = NUM_CLASSES,
    dropout: float = DROPOUT_RATE
) -> nn.Module:
    """
    Build and return a transfer-learning model based on MobileNetV2.

    Args
    ----
    freeze_backbone : If True, all convolutional layers are frozen (only the
                      custom head is trained).  Set to False to fine-tune the
                      entire network — useful after initial training converges.
    num_classes     : Number of output classes (2 for binary classification).
    dropout         : Dropout probability in the custom classification head.

    Returns
    -------
    model : nn.Module ready to be moved to a device and trained.

    FREEZE vs. FINE-TUNE STRATEGY
    ─────────────────────────────
    Phase 1 (current): freeze_backbone=True
      → Only the ~600K custom head parameters are trained.
      → Faster, less risk of overfitting, converges in fewer epochs.
      → Ideal when your dataset is small relative to ImageNet.

    Phase 2 (optional follow-up): freeze_backbone=False
      → Unfreeze and retrain the whole network at a very small LR (1e-5).
      → Allows the backbone to specialise its features for deepfake detection.
      → Requires more data / careful LR scheduling to avoid "catastrophic forgetting".
    """

    # ── Load pre-trained MobileNetV2 from torchvision model zoo ───────────────
    # weights=IMAGENET1K_V1 downloads the official weights trained by Google.
    # These weights encode powerful, generalisable visual features.
    backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    # ── Optionally freeze all backbone (feature extractor) layers ─────────────
    # requires_grad=False tells PyTorch NOT to compute gradients for these params,
    # which speeds up training and reduces memory usage significantly.
    if freeze_backbone:
        for param in backbone.features.parameters():
            param.requires_grad = False
        print("[Model] Backbone FROZEN — only classification head will be trained.")
    else:
        print("[Model] Backbone UNFROZEN — full fine-tuning enabled.")

    # ── Replace the final classifier with our custom binary head ──────────────
    # backbone.classifier is currently: [Dropout(0.2), Linear(1280, 1000)]
    # We replace it entirely with our custom two-layer head.
    backbone.classifier = nn.Sequential(

        # Layer 1: Project 1280-dim feature vector down to 512-dim
        nn.Linear(BACKBONE_OUT, HIDDEN_DIM),

        # Non-linearity: ReLU introduces non-linearity so the head can learn
        # complex decision boundaries, not just a linear hyperplane.
        nn.ReLU(inplace=True),

        # Regularisation: randomly zero neurons at training time to prevent
        # the model from relying too heavily on any single feature.
        nn.Dropout(p=dropout),

        # Layer 2: Final projection to NUM_CLASSES logits (raw scores, not probs)
        # Softmax is NOT applied here — CrossEntropyLoss handles that internally.
        nn.Linear(HIDDEN_DIM, num_classes),
    )

    # ── Count and report trainable parameters ─────────────────────────────────
    total_params     = sum(p.numel() for p in backbone.parameters())
    trainable_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    print(f"[Model] Total params:     {total_params:,}")
    print(f"[Model] Trainable params: {trainable_params:,}  "
          f"({100 * trainable_params / total_params:.1f}%)")

    return backbone


# ─────────────────────────────────────────────────────────────────────────────
# QUICK ARCHITECTURE SANITY CHECK (run this file directly to verify shapes)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = build_model(freeze_backbone=True)

    # Simulate a batch of 4 images at 224×224 with 3 colour channels
    dummy_input = torch.randn(4, 3, 224, 224)
    output = model(dummy_input)

    print(f"\n[Sanity Check] Input shape:  {dummy_input.shape}")
    print(f"[Sanity Check] Output shape: {output.shape}")
    # Expected: torch.Size([4, 2])  — 4 samples, 2 class logits each
    assert output.shape == (4, NUM_CLASSES), \
        f"Shape mismatch: expected (4, {NUM_CLASSES}), got {output.shape}"
    print("[Sanity Check] ✓ Output shape is correct!")
    print(model)
