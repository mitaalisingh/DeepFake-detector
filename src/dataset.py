"""
================================================================================
src/dataset.py — Phase 1: Data Loading & Preprocessing
================================================================================

PURPOSE
-------
This module handles everything related to getting image data ready for the model.
It defines augmentation pipelines for training vs. validation/test sets and
returns PyTorch DataLoader objects that feed batches to the training loop.

WHY TWO SEPARATE TRANSFORMS?
  - Training   → we WANT randomness (flips, color jitter) to teach the model to
                 generalise and not memorise exact pixel patterns.
  - Val / Test → we NEVER augment at evaluation time; results must be deterministic
                 so metrics are comparable across runs.

IMAGENET NORMALISATION
  The magic numbers (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]) are the
  channel-wise statistics computed over the full ImageNet-1K dataset. Because
  MobileNetV2 was pre-trained on those normalised images, we MUST apply the same
  normalisation here so the pre-trained weights "see" inputs in the expected range.

CIFAKE FOLDER LAYOUT (expected after download & extraction)
  data/
    train/
      REAL/   ← images of real photographs
      FAKE/   ← images of AI-generated synthetics
    test/
      REAL/
      FAKE/
================================================================================
"""

import os
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# ─────────────────────────────────────────────────────────────────────────────
# 1. NORMALISATION CONSTANTS
#    Reused by the inference script (predict.py) so values stay in one place.
# ─────────────────────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Target spatial resolution accepted by MobileNetV2
INPUT_SIZE = 224


# ─────────────────────────────────────────────────────────────────────────────
# 2. TRANSFORM PIPELINES
# ─────────────────────────────────────────────────────────────────────────────

def get_train_transforms() -> transforms.Compose:
    """
    Augmentation pipeline applied ONLY during training.

    Augmentation rationale:
      - Resize(256) + RandomCrop(224)  → slight random zooming / cropping
      - RandomHorizontalFlip           → left-right mirror is label-preserving
      - ColorJitter                    → vary brightness/contrast/saturation so
                                         the model isn't fooled by lighting
      - RandomRotation(±10°)           → small tilts are common in real photos
      - ToTensor                       → converts PIL Image [0,255] → float [0,1]
      - Normalize                      → shifts to ImageNet distribution
    """
    return transforms.Compose([
        transforms.Resize(256),                         # slightly larger than target
        transforms.RandomCrop(INPUT_SIZE),              # random crop to 224×224
        transforms.RandomHorizontalFlip(p=0.5),         # 50% chance of horizontal flip
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05
        ),                                              # photometric distortion
        transforms.RandomRotation(degrees=10),          # ±10 degree rotation
        transforms.ToTensor(),                          # PIL → FloatTensor [C,H,W]
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        ),
    ])


def get_val_transforms() -> transforms.Compose:
    """
    Deterministic pipeline for validation and test sets.
    No randomness — results must be reproducible.
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(INPUT_SIZE),              # always the centre 224×224
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        ),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# 3. DATASET & DATALOADER FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def get_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    val_split: float = 0.15,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    """
    Build train / validation / test DataLoaders from the CIFAKE folder layout.

    Args
    ----
    data_dir   : Root of the dataset, e.g. "data/" — must contain train/ and test/
    batch_size : Number of images per mini-batch
    val_split  : Fraction of the *training* folder to use as validation
    num_workers: Parallel CPU workers for data loading (0 = single-threaded)

    Returns
    -------
    train_loader, val_loader, test_loader, class_to_idx mapping

    HOW THE SPLIT WORKS
    -------------------
    ImageFolder loads the entire train/ folder.  We then carve out (val_split × N)
    images for validation using random_split — guaranteeing no leakage between
    training and validation sets.  The test/ folder is kept separate as an
    unseen hold-out set (the "real exam").
    """

    train_dir = os.path.join(data_dir, "train")
    test_dir  = os.path.join(data_dir, "test")

    # ── Load the raw training folder (no transforms yet — applied below) ──────
    # ImageFolder auto-assigns integer labels based on alphabetical sub-folder order:
    #   FAKE → 0,  REAL → 1  (alphabetical)
    full_train_dataset = datasets.ImageFolder(
        root=train_dir,
        transform=get_train_transforms()
    )

    # ── Compute split sizes ───────────────────────────────────────────────────
    n_total = len(full_train_dataset)
    n_val   = int(n_total * val_split)
    n_train = n_total - n_val

    train_subset, val_subset = random_split(
        full_train_dataset,
        [n_train, n_val]
    )

    # IMPORTANT: The val subset gets its own transforms (no augmentation).
    # We wrap it in a small helper that overrides the transform at access time.
    val_subset.dataset = datasets.ImageFolder(
        root=train_dir,
        transform=get_val_transforms()
    )

    # ── Test set (fully unseen hold-out) ──────────────────────────────────────
    test_dataset = datasets.ImageFolder(
        root=test_dir,
        transform=get_val_transforms()
    )

    # ── DataLoaders: efficient batch delivery to the GPU ──────────────────────
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,               # randomise order every epoch
        num_workers=num_workers,
        pin_memory=True             # speeds up CPU→GPU transfer
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,              # order doesn't matter for evaluation
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    class_to_idx = full_train_dataset.class_to_idx
    print(f"[Dataset] Classes: {class_to_idx}")
    print(f"[Dataset] Train: {n_train}  |  Val: {n_val}  |  Test: {len(test_dataset)}")

    return train_loader, val_loader, test_loader, class_to_idx
