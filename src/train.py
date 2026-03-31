"""
================================================================================
src/train.py — Phase 3: Training Loop with Early Stopping
================================================================================

ENTRY POINT
-----------
Run from the project root:
    python src/train.py

This script orchestrates the full training pipeline:
  1. Load dataset → DataLoaders
  2. Build model → move to GPU/CPU
  3. Define loss, optimiser, LR scheduler
  4. Loop over epochs:
       a. Train one epoch (forward, backward, update)
       b. Validate one epoch (no gradients)
       c. Check early stopping
       d. Log metrics
  5. Load best checkpoint
  6. Evaluate on held-out test set + print metrics report

LOSS FUNCTION: Cross-Entropy Loss
  The standard loss for multi-class classification.  Internally applies
  Softmax to the raw logits from our model head, then computes the negative
  log-likelihood of the true class.  Minimising this pushes the model to
  assign high probability to the correct label.

  Formula: L = -Σ y_i · log(ŷ_i)   where y_i is the one-hot true label

OPTIMISER: Adam (Adaptive Moment Estimation)
  Adam maintains a per-parameter adaptive learning rate based on first and
  second moment estimates of the gradients.  Advantages:
  - Converges faster than plain SGD on most problems
  - Less sensitive to the initial learning rate choice
  - Handles sparse gradients well (important for frozen backbone layers)

LR SCHEDULER: ReduceLROnPlateau
  Monitors the validation loss every epoch.  If it hasn't improved for
  `patience` epochs, it multiplies the LR by `factor`.  This is a form of
  adaptive cooling — start learning fast, slow down when progress stalls.
  Prevents the optimiser from overshooting a good minimum.

EARLY STOPPING
  If validation loss does not improve for `patience` consecutive epochs:
  → Restore the weights from the best checkpoint
  → Terminate training early
  This prevents overfitting — memorising the training set at the expense
  of generalisation to unseen data.
================================================================================
"""

import os
import csv
import time
import argparse

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# Local modules
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataset import get_dataloaders
from src.model   import build_model
from src.utils   import set_seed, plot_training_curves, compute_metrics


# ─────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETERS (change these via CLI or directly)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "data_dir"       : "data",
    "batch_size"     : 32,
    "num_epochs"     : 30,
    "learning_rate"  : 1e-3,
    "weight_decay"   : 1e-4,       # L2 regularisation penalty on weights
    "patience"       : 5,          # early stopping patience (epochs)
    "lr_factor"      : 0.5,        # LR reduction factor when plateau detected
    "lr_patience"    : 3,          # LR scheduler patience (epochs)
    "checkpoint_dir" : "checkpoints",
    "log_dir"        : "logs",
    "seed"           : 42,
    "num_workers"    : 4,
    "freeze_backbone": True,
}


# ─────────────────────────────────────────────────────────────────────────────
# EARLY STOPPING CLASS
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Monitors validation loss and stops training if it stops improving.

    Attributes
    ----------
    patience      : Number of epochs to wait before stopping
    min_delta     : Minimum improvement to count as "improvement"
    best_loss     : Best validation loss seen so far
    counter       : Consecutive epochs without improvement
    best_epoch    : The epoch number where best performance occurred
    should_stop   : Flag — set to True when training should terminate
    """

    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float("inf")
        self.counter    = 0
        self.best_epoch = 0
        self.should_stop = False

    def step(self, val_loss: float, epoch: int) -> bool:
        """
        Call once per epoch with the current validation loss.
        Returns True if a new best checkpoint was achieved.
        """
        if val_loss < self.best_loss - self.min_delta:
            # ── Improvement detected ──────────────────────────────────────────
            self.best_loss  = val_loss
            self.counter    = 0
            self.best_epoch = epoch
            return True  # caller should save checkpoint
        else:
            # ── No improvement ────────────────────────────────────────────────
            self.counter += 1
            print(f"  [EarlyStopping] No improvement for {self.counter}/{self.patience} epochs "
                  f"(best val_loss={self.best_loss:.4f} @ epoch {self.best_epoch})")
            if self.counter >= self.patience:
                self.should_stop = True
            return False


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING FUNCTION — ONE EPOCH
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> tuple[float, float]:
    """
    Perform one full pass over the TRAINING data.

    The forward-backward-update loop:
      1. Forward pass  : model(inputs)  → logits (raw class scores)
      2. Compute loss  : CrossEntropy(logits, labels)
      3. Zero grads    : reset accumulated gradients from previous batch
      4. Backward pass : autograd computes ∂Loss/∂w for every parameter
      5. Optimiser step: update weights using the computed gradients

    Returns
    -------
    avg_loss : mean cross-entropy loss across all batches this epoch
    accuracy : percentage of correct predictions this epoch
    """
    model.train()   # IMPORTANT: enables Dropout and BatchNorm training behaviour

    running_loss = 0.0
    correct      = 0
    total        = 0

    progress_bar = tqdm(loader, desc=f"Epoch {epoch:02d} [Train]", leave=False)

    for inputs, labels in progress_bar:
        # ── Move data to the compute device (GPU if available) ───────────────
        inputs, labels = inputs.to(device), labels.to(device)

        # ── Zero accumulated gradients from the previous batch ───────────────
        # MUST be done before each forward pass, otherwise gradients accumulate
        optimiser.zero_grad()

        # ── Forward pass: compute model predictions (logits, NOT probabilities)
        logits = model(inputs)

        # ── Compute loss ─────────────────────────────────────────────────────
        # CrossEntropyLoss = Softmax + Negative Log Likelihood in one step.
        loss = criterion(logits, labels)

        # ── Backward pass: compute gradients via chain rule (autograd) ────────
        loss.backward()

        # ── Gradient clipping (optional but stabilises training) ──────────────
        # Prevents "exploding gradients" — caps the L2-norm of gradient vector.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # ── Optimiser step: apply gradients to update weights ─────────────────
        optimiser.step()

        # ── Accumulate statistics ─────────────────────────────────────────────
        running_loss += loss.item() * inputs.size(0)   # scale back from mean
        preds  = logits.argmax(dim=1)                  # pick class with highest logit
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

        # Update tqdm bar with live metrics
        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc":  f"{100 * correct / total:.1f}%"
        })

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION / EVALUATION FUNCTION — ONE EPOCH
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    phase: str = "Val",
    return_preds: bool = False,
) -> tuple[float, float] | tuple[float, float, list, list]:
    """
    Evaluate the model on a DataLoader WITHOUT updating weights.

    KEY DIFFERENCES from train_one_epoch:
      - model.eval() disables Dropout and switches BatchNorm to use running stats
      - torch.no_grad() skips gradient computation entirely (saves ~50% memory)
      - No optimiser.step() — weights do not change

    Args
    ----
    return_preds : If True, also return (all_labels, all_preds) lists for
                   computing the full classification report.
    """
    model.eval()    # IMPORTANT: disables Dropout for deterministic inference

    running_loss = 0.0
    correct      = 0
    total        = 0
    all_labels   = []
    all_preds    = []

    with torch.no_grad():   # context manager: no gradients computed
        progress_bar = tqdm(loader, desc=f"         [{phase}]", leave=False)

        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            logits = model(inputs)
            loss   = criterion(logits, labels)

            running_loss += loss.item() * inputs.size(0)
            preds  = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

            if return_preds:
                all_labels.extend(labels.cpu().tolist())
                all_preds.extend(preds.cpu().tolist())

            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc":  f"{100 * correct / total:.1f}%"
            })

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total

    if return_preds:
        return avg_loss, accuracy, all_labels, all_preds
    return avg_loss, accuracy


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def train(config: dict) -> None:
    """
    Full training pipeline — called with a config dict.

    Steps:
      1. Setup (seed, device, directories)
      2. Load data → DataLoaders
      3. Build model
      4. Define loss, optimiser, scheduler, early stopper
      5. Epoch loop → train → validate → checkpoint → early stop
      6. Evaluate on test set
      7. Save plots and logs
    """

    # ── 1. Setup ──────────────────────────────────────────────────────────────
    set_seed(config["seed"])

    # Device selection: prefer CUDA GPU → Apple MPS (M1/M2) → CPU fallback
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[Train] Using device: {device}")

    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    os.makedirs(config["log_dir"],        exist_ok=True)

    # ── 2. Data ───────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, class_to_idx = get_dataloaders(
        data_dir    = config["data_dir"],
        batch_size  = config["batch_size"],
        num_workers = config["num_workers"],
    )
    # Reverse mapping: {0: 'FAKE', 1: 'REAL'} for human-readable output
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # ── 3. Model ──────────────────────────────────────────────────────────────
    model = build_model(freeze_backbone=config["freeze_backbone"])
    model = model.to(device)

    # ── 4. Loss, Optimiser, Scheduler, Early Stopper ─────────────────────────

    # CrossEntropyLoss: well-suited for multi-class (and binary) classification
    criterion = nn.CrossEntropyLoss()

    # Adam optimiser only updates parameters where requires_grad=True
    # weight_decay adds L2 penalty → implicit regularisation
    optimiser = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr           = config["learning_rate"],
        weight_decay = config["weight_decay"],
    )

    # ReduceLROnPlateau: halve LR if val_loss doesn't improve for `lr_patience` epochs
    scheduler = ReduceLROnPlateau(
        optimiser,
        mode      = "min",              # we want val_loss to go DOWN
        factor    = config["lr_factor"],
        patience  = config["lr_patience"],
        verbose   = True
    )

    early_stopper   = EarlyStopping(patience=config["patience"])
    checkpoint_path = os.path.join(config["checkpoint_dir"], "best_model.pth")

    # ── 5. CSV Logger (tracks metrics for plot generation later) ──────────────
    log_csv = os.path.join(config["log_dir"], "training_log.csv")
    csv_file = open(log_csv, "w", newline="")
    csv_writer = csv.DictWriter(
        csv_file,
        fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"]
    )
    csv_writer.writeheader()

    # ── 6. EPOCH LOOP ─────────────────────────────────────────────────────────
    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []

    print(f"\n{'='*60}")
    print(f"  Starting training for max {config['num_epochs']} epochs")
    print(f"{'='*60}\n")

    start_time = time.time()

    for epoch in range(1, config["num_epochs"] + 1):

        # ── Train one epoch ───────────────────────────────────────────────────
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimiser, device, epoch
        )

        # ── Validate one epoch ────────────────────────────────────────────────
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device, phase="Val"
        )

        # ── LR Scheduler step (based on val loss) ─────────────────────────────
        scheduler.step(val_loss)
        current_lr = optimiser.param_groups[0]["lr"]

        # ── Log metrics ───────────────────────────────────────────────────────
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        csv_writer.writerow({
            "epoch"     : epoch,
            "train_loss": f"{train_loss:.4f}",
            "train_acc" : f"{train_acc:.2f}",
            "val_loss"  : f"{val_loss:.4f}",
            "val_acc"   : f"{val_acc:.2f}",
            "lr"        : f"{current_lr:.2e}",
        })
        csv_file.flush()

        print(f"Epoch {epoch:02d}/{config['num_epochs']}  |  "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.1f}%  |  "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.1f}%  |  "
              f"LR: {current_lr:.2e}")

        # ── Early stopping check + checkpoint save ────────────────────────────
        is_best = early_stopper.step(val_loss, epoch)
        if is_best:
            # Save everything needed to reconstruct the model at inference time
            torch.save({
                "epoch"        : epoch,
                "model_state"  : model.state_dict(),
                "optimiser"    : optimiser.state_dict(),
                "val_loss"     : val_loss,
                "val_acc"      : val_acc,
                "class_to_idx" : class_to_idx,
                "config"       : config,
            }, checkpoint_path)
            print(f"  ✓ Checkpoint saved (val_loss={val_loss:.4f})")

        if early_stopper.should_stop:
            print(f"\n[EarlyStopping] Triggered at epoch {epoch}. "
                  f"Best epoch was {early_stopper.best_epoch}.")
            break

    csv_file.close()

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed/60:.1f} minutes")

    # ── 7. Load best checkpoint and evaluate on test set ──────────────────────
    print(f"\n[Test] Loading best weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    test_loss, test_acc, all_labels, all_preds = evaluate(
        model, test_loader, criterion, device,
        phase="Test", return_preds=True
    )
    print(f"[Test] Loss: {test_loss:.4f}  |  Accuracy: {test_acc:.2f}%")

    class_names = [idx_to_class[i] for i in sorted(idx_to_class.keys())]
    compute_metrics(all_labels, all_preds, class_names=class_names)

    # ── 8. Save training curves ────────────────────────────────────────────────
    plot_training_curves(
        train_losses, val_losses,
        train_accs,   val_accs,
        save_path=os.path.join(config["log_dir"], "training_curves.png")
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train MobileNetV2 deepfake detector on the CIFAKE dataset"
    )
    parser.add_argument("--data_dir",        default=DEFAULT_CONFIG["data_dir"])
    parser.add_argument("--batch_size",      type=int,   default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--num_epochs",      type=int,   default=DEFAULT_CONFIG["num_epochs"])
    parser.add_argument("--learning_rate",   type=float, default=DEFAULT_CONFIG["learning_rate"])
    parser.add_argument("--weight_decay",    type=float, default=DEFAULT_CONFIG["weight_decay"])
    parser.add_argument("--patience",        type=int,   default=DEFAULT_CONFIG["patience"])
    parser.add_argument("--checkpoint_dir",  default=DEFAULT_CONFIG["checkpoint_dir"])
    parser.add_argument("--log_dir",         default=DEFAULT_CONFIG["log_dir"])
    parser.add_argument("--seed",            type=int,   default=DEFAULT_CONFIG["seed"])
    parser.add_argument("--num_workers",     type=int,   default=DEFAULT_CONFIG["num_workers"])
    parser.add_argument("--unfreeze",        action="store_true",
                        help="Fine-tune entire backbone (default: only head)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = vars(args)
    config["freeze_backbone"] = not config.pop("unfreeze")
    train(config)
