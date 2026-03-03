# prostate-gleason

Modular PyTorch trainer for prostate cancer Gleason grade classification.

## Files

| File | Purpose |
|------|---------|
| `config.py` | All settings: paths, hyperparameters, loss strategy, early stopping |
| `data.py` | Dataset class and DataLoaders |
| `model.py` | ResNet50 with custom classification head |
| `losses.py` | Loss functions: CrossEntropy, weighted CE, Focal Loss |
| `eval.py` | Accuracy, precision, recall, F1 on validation set |
| `train.py` | Main training loop |

## Setup

```bash
pip install torch torchvision scikit-learn matplotlib seaborn pillow
```

## Data format

Images must be named with the label at the end:

```
anything_{label}.png   or   anything-{label}.png
```

Flat directories and nested subdirectories both work.

Labels used: **0** (Benign), **1** (Gleason 3), **2** (Gleason 4), **3** (Gleason 5-Single Cells).
Labels 4 and 5 are automatically skipped.

## Run

```bash
python train.py
```

Outputs:
- `best_model.pth` — PyTorch state dict of the best checkpoint
- `confusion_matrix.png` — heatmap of predictions vs. true labels
- `training_log.txt` — full epoch-by-epoch training log

## Configuration

Edit `config.py` before running.

### Paths and basics

```python
DATA_ROOT = "./patches_prostate_seer_john_6classes"
BATCH_SIZE = 64
LEARNING_RATE = 0.0001
NUM_EPOCHS = 25
TRAIN_SPLIT = 0.8    # 80% train, 20% val
```

### Class imbalance

The dataset is heavily imbalanced (Benign: 19K, Gleason 5: 815). Choose a weighting strategy:

```python
# Options: "none" | "inverse" | "inverse_sqrt" | "effective" | "focal"
WEIGHTING_STRATEGY = "none"
```

| Strategy | Description |
|----------|-------------|
| `none` | Standard cross-entropy, no reweighting |
| `inverse` | Weight inversely proportional to class frequency |
| `inverse_sqrt` | Softer version — square root of inverse frequency |
| `effective` | Class-Balanced Loss (Cui et al. 2019) |
| `focal` | Focal Loss — down-weights easy examples |

You can also override weights manually:

```python
MANUAL_WEIGHTS = [1.0, 1.0, 2.0, 10.0]   # per class, or None to use strategy
USE_WEIGHTED_SAMPLING = False               # oversample minority classes instead
```

### Early stopping

```python
EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_METRIC = "accuracy"   # "accuracy" | "f1" | "loss"
```

## Batch size vs GPU memory

| VRAM | Suggested batch size |
|------|---------------------|
| 8 GB | 32–64 |
| 16 GB | 64–128 |
| 24 GB+ | 128+ |
