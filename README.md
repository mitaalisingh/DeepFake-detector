# 🕵️ Deepfake / Synthetic Image Detector

> **University BYOP Capstone** — Binary classifier to detect AI-generated face images using Transfer Learning on the CIFAKE dataset.

---

## 📌 Problem Statement

Web platforms (dating apps, matrimony sites, social networks) are increasingly being flooded with AI-generated profile pictures created by models like Stable Diffusion and DALL-E. This project builds a backend deep learning pipeline to classify any uploaded image as **REAL** (genuine photograph) or **FAKE** (AI-generated synthetic).

---

## 📂 Project Structure

```
CV/
├── data/                    # ← CIFAKE dataset (you download this)
│   ├── train/
│   │   ├── REAL/
│   │   └── FAKE/
│   └── test/
│       ├── REAL/
│       └── FAKE/
│
├── src/
│   ├── dataset.py           # Phase 1: Data loading & augmentation
│   ├── model.py             # Phase 2: MobileNetV2 + custom head
│   ├── train.py             # Phase 3: Training loop + early stopping
│   └── utils.py             # Shared helpers (seed, plots, metrics)
│
├── checkpoints/             # Saved model weights (auto-created)
├── logs/                    # Training curves CSV & PNG (auto-created)
│
├── predict.py               # Phase 4: CLI inference (the deliverable)
├── requirements.txt
└── README.md
```

---

## 🔧 Setup

### 1. Clone & Install

```bash
git clone <your-repo-url>
cd CV
pip install -r requirements.txt
```

### 2. Download the CIFAKE Dataset

1. Go to: https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images
2. Download and unzip to `data/` so the structure matches the layout above.

---

## 🚀 Usage

### Train the Model

```bash
# Default settings (30 epochs, batch=32, LR=1e-3, early stopping patience=5)
python src/train.py

# Custom settings
python src/train.py --num_epochs 50 --batch_size 64 --patience 7

# Fine-tune entire backbone (after initial training converges)
python src/train.py --unfreeze --learning_rate 1e-5
```

**Training output includes:**
- Per-epoch train/val loss and accuracy in the terminal
- `logs/training_log.csv` — full metrics history
- `logs/training_curves.png` — loss & accuracy plots
- `checkpoints/best_model.pth` — best checkpoint (restored by early stopping)

---

### Run Inference (The Deliverable)

```bash
python predict.py --image path/to/face.jpg
```

**Example output:**
```
┌───────────────────────────────────────────────┐
│  Deepfake Detection Result                    │
│───────────────────────────────────────────────│
│  File    : face.jpg                           │
│  Result  : 🔴 FAKE                            │
│  Score   : FAKE=98.2%  |  REAL=1.8%           │
└───────────────────────────────────────────────┘

Prediction: FAKE (98.2% confidence)
```

**Options:**
```bash
# Open the image before classifying
python predict.py --image face.jpg --show

# Use a custom checkpoint path
python predict.py --image face.jpg --checkpoint checkpoints/best_model.pth
```

---

## 🧠 Methodology

### Phase 1 — Data Loading & Augmentation (`src/dataset.py`)

| Split | Augmentation | Purpose |
|-------|-------------|---------|
| Train | RandomCrop, HorizontalFlip, ColorJitter, RandomRotation | Improve generalisation |
| Val / Test | CenterCrop only | Deterministic, comparable metrics |

All images are normalised using **ImageNet statistics** (`mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]`) because the backbone was pre-trained on ImageNet.

### Phase 2 — Model Architecture (`src/model.py`)

**Backbone:** MobileNetV2 (pre-trained on ImageNet-1K)

**Why MobileNetV2?**
- Only **~3.4M parameters** — fast inference, fits on a CPU
- Depthwise separable convolutions — 9× fewer operations than standard convolutions
- Strong generalisation from rich ImageNet features
- 1280-dimensional feature vector as input to our custom head

**Custom Classification Head:**
```
Linear(1280 → 512) → ReLU → Dropout(0.4) → Linear(512 → 2)
```

### Phase 3 — Training Loop (`src/train.py`)

| Component | Choice | Reason |
|-----------|--------|--------|
| Loss | CrossEntropyLoss | Standard for multi-class classification |
| Optimiser | Adam (lr=1e-3, wd=1e-4) | Fast convergence, adaptive LR per parameter |
| LR Scheduler | ReduceLROnPlateau | Halve LR if val loss plateaus for 3 epochs |
| Regularisation | Dropout(0.4) + L2 weight decay | Prevent co-adaptation and weight explosion |
| Early Stopping | Patience = 5 epochs | Restore best weights, prevent overfitting |

### Phase 4 — Inference Script (`predict.py`)

Standalone CLI tool that:
1. Loads the best checkpoint
2. Applies identical val-time transforms
3. Runs `Softmax` on model logits → per-class probabilities
4. Prints a formatted confidence report

---

## 📊 Detecting AI Artifacts — The Key Challenge

AI-generated images contain subtle statistical regularities that DNNs can learn to detect, but that are difficult to see with the human eye:

| Artifact | Description |
|----------|-------------|
| **Spectral anomalies** | GAN/Diffusion models produce unnatural high-frequency patterns in Fourier space |
| **Texture inconsistencies** | Skin pores, hair strands, and fabric textures are often too uniform |
| **Geometric impossibilities** | Background objects, earrings, teeth, and glasses are common failure modes |
| **Colour distribution bias** | Synthetic images often cluster in a tighter colour space than real photos |

MobileNetV2's convolutional filters learn to detect these inconsistencies through training, even without explicit engineering of these features.

---

## 📈 Expected Results

On the CIFAKE dataset with these settings:

| Metric | Expected (Head-only) | Expected (Fine-tuned) |
|--------|---------------------|----------------------|
| Test Accuracy | ~88–92% | ~93–96% |
| FAKE F1-Score | ~0.87–0.91 | ~0.93–0.96 |
| Training Time | ~15–20 min (CPU) | ~40–60 min (CPU) |

---

## ⚙️ CLI Reference

### `src/train.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | `data` | Root dir of CIFAKE dataset |
| `--batch_size` | `32` | Images per mini-batch |
| `--num_epochs` | `30` | Maximum training epochs |
| `--learning_rate` | `0.001` | Initial learning rate |
| `--weight_decay` | `0.0001` | L2 regularisation factor |
| `--patience` | `5` | Early stopping patience |
| `--unfreeze` | `False` | Fine-tune entire backbone |
| `--seed` | `42` | Random seed for reproducibility |

### `predict.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--image` | *(required)* | Path to input image |
| `--checkpoint` | `checkpoints/best_model.pth` | Model checkpoint path |
| `--show` | `False` | Open image before prediction |

---

## 📦 Dependencies

```
torch>=2.1.0
torchvision>=0.16.0
tqdm>=4.66.0
scikit-learn>=1.3.0
matplotlib>=3.8.0
Pillow>=10.0.0
numpy>=1.26.0
```

---

## 📄 License

MIT — free to use, modify, and cite in your project report.
