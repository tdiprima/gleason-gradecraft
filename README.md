# Gleason Gradecraft

Deep learning pipelines for classifying prostate cancer tissue patches by Gleason score.

## Why this exists

Prostate cancer is graded by pathologists using the **Gleason scoring system**, which categorizes tissue patterns from benign to increasingly aggressive. Doing this manually at scale is slow and subject to inter-observer variability. These tools train a computer vision model to do that classification automatically from histopathology image patches.

The dataset comes from the [SEER](https://seer.cancer.gov/) prostate pathology collection (`patches_prostate_seer_john_6classes`). Labels 4 and 5 are skipped because they have too few samples and represent overlapping subtypes — only the four main classes are used:

| Label | Class |
|-------|-------|
| 0 | Benign |
| 1 | Gleason 3 |
| 2 | Gleason 4 |
| 3 | Gleason 5-Single Cells |

Two separate implementations live in `src/` — one built around FastAI + Optuna, the other a modular pure-PyTorch version. Both reached ~92% test accuracy.

---

## Projects

### gleason-gradecraft

A single-file pipeline using **FastAI** and **Optuna** for automatic hyperparameter tuning.

Run 20 trials with different learning rates, batch sizes, and epoch counts → train a final model with the best settings → evaluate and save everything.

**When to use this:** You want to drop in data and get a trained model with minimal configuration. Optuna finds good hyperparameters for you.

### prostate-gleason

A modular **pure PyTorch** trainer split across multiple files (`model.py`, `data.py`, `train.py`, `eval.py`, `losses.py`, `config.py`).

Exposes more knobs: multiple class weighting strategies (inverse frequency, square root, effective number of samples, focal loss), configurable early stopping, and cosine annealing LR scheduling.

**When to use this:** You want to experiment with loss functions and weighting strategies, or build on top of the components individually.

---

## Requirements

- Python 3.11+
- NVIDIA GPU + CUDA (CPU works but training will be very slow)
- 8GB+ VRAM recommended

---

## Running gleason-gradecraft

### 1. Prepare your images

Images must be named with the label at the end:

```
anything_{label}.png
```

Example: `patient001_patch042_2.png` → label 2 (Gleason 4)

Put all images in:

```
./gleason_images/
```

### 2. Install and run

```bash
cd src/gleason-gradecraft
uv sync
uv run gleason_classifier.py
```

Optuna will run 20 trials, then train a final model with the best hyperparameters.

### 3. Output

Everything lands in `./output/`:

| File | What it is |
|------|-----------|
| `best_model.pkl` | Trained FastAI model (exportable for inference) |
| `confusion_matrix.csv` | Confusion matrix on test set |
| `per_class_metrics.csv` | Precision / recall / F1 per class |
| `training_log_*.txt` | Full training logs with timestamps |

To visualize the confusion matrix after training:

```bash
uv run visualize_confusion_matrix.py
```

### Configuration

Edit the `Config` class at the top of `gleason_classifier.py`:

```python
DATA_PATH = Path("./gleason_images")   # your image folder
BATCH_SIZE = 32                         # lower if GPU runs out of memory
NUM_TRIALS = 20                         # Optuna trials (more = better but slower)
```

---

## Running prostate-gleason

### 1. Prepare your images

Images must end with `_{label}.png` or `-{label}.png`. Flat directory or nested — both work.

Point `DATA_ROOT` at your image folder in `config.py`:

```python
DATA_ROOT = "./patches_prostate_seer_john_6classes"
```

### 2. Run

```bash
cd src/prostate-gleason
pip install torch torchvision scikit-learn matplotlib seaborn pillow
python train.py
```

### 3. Output

| File | What it is |
|------|-----------|
| `best_model.pth` | PyTorch state dict of the best checkpoint |
| `confusion_matrix.png` | Heatmap saved as PNG |
| `training_log.txt` | Full training log |

### Configuration

All settings are in `config.py`. Key options:

```python
BATCH_SIZE = 64
LEARNING_RATE = 0.0001
NUM_EPOCHS = 25
TRAIN_SPLIT = 0.8          # 80% train, 20% val

# Class imbalance strategy — pick one:
# "none" | "inverse" | "inverse_sqrt" | "effective" | "focal"
WEIGHTING_STRATEGY = "none"

# Early stopping
EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_METRIC = "accuracy"   # "accuracy" | "f1" | "loss"
```

---

## Results

Best run from `gleason-gradecraft` (Experiment 2, GPU-enabled):

| Metric | Value |
|--------|-------|
| Test accuracy | 92.3% |
| Weighted F1 | 0.923 |
| Benign F1 | 0.955 |
| Gleason 3 F1 | 0.929 |
| Gleason 4 F1 | 0.841 |
| Gleason 5-Single Cells F1 | 0.785 |

Class weighting kept Gleason 5 recall high (91.5%) despite only 815 training samples. Most errors occur between adjacent Gleason grades, which mirrors inter-pathologist disagreement on the same cases.

See `docs/experiments.md` for full experiment logs.

---

## Common failures

| Symptom | Fix |
|---------|-----|
| CUDA out of memory | Lower `BATCH_SIZE` |
| No GPU detected | Check CUDA installation |
| No valid images found | Verify filenames end in `_{label}.png` |
| Slow training | You're on CPU — get a GPU or use a cloud VM |
