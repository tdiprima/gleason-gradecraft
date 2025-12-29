# Gleason Score Image Classifier

A machine learning pipeline for classifying prostate cancer histopathology images by Gleason score.

## Overview

This pipeline uses:

- **FastAI** - Makes deep learning accessible and fast
- **Optuna** - Automatically finds the best hyperparameters
- **ResNet** - Proven image classification architecture

## Gleason Score Labels

| Label | Description |
|-------|-------------|
| 0 | Benign |
| 1 | Gleason 3 |
| 2 | Gleason 4 |
| 3 | Gleason 5-Single Cells |
| 4 | Gleason 5-Secretions (SKIPPED) |
| 5 | Gleason 5 (SKIPPED) |

## Requirements

- Python 3.9+
- NVIDIA GPU with CUDA support
- At least 8GB GPU memory recommended

## Installation

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Data Setup

1. Place images in the `./gleason_images` folder
2. Images should be named with the pattern: `*-{label}.png`
   - Example: `patient001_patch042-2.png` (label = 2, Gleason 4)

## Usage

```bash
# Run the training pipeline
python gleason_classifier.py
```

## Configuration

Edit the `Config` class in `gleason_classifier.py` to change:

```python
class Config:
    DATA_PATH = Path("./gleason_images") # Image folder
    OUTPUT_PATH = Path("./output")    # Where results are saved
    NUM_TRIALS = 20                   # Optuna optimization trials
    BATCH_SIZE = 32                   # Reduce if GPU runs out of memory
```

## Output Files

After training, you'll find in `./output/`:

- `best_model.pkl` - The trained model
- `training_log_*.txt` - Detailed training logs
- `confusion_matrix.csv` - Model predictions vs actual
- `per_class_metrics.csv` - Precision, recall, F1 per class

## Understanding the Code

### Key Concepts

1. **Class Weights**: Since the data is imbalanced (some classes have more images), we give higher weights to rare classes so the model learns them better.

2. **Data Split**: 
   - Training (70%): Model learns from these
   - Validation (15%): Checks performance during training
   - Test (15%): Final evaluation on unseen data

3. **Hyperparameters**: Settings like learning rate that affect training. Optuna finds the best ones automatically.

4. **Transfer Learning**: We start with a model pre-trained on ImageNet, then fine-tune it for our task. This is faster and works better with limited data.

### Pipeline Steps

1. **Setup** - Initialize logging, check GPU
2. **Load Data** - Read images, extract labels, split data
3. **Calculate Weights** - Handle class imbalance
4. **Hyperparameter Search** - Try different settings
5. **Train Final Model** - Use best settings
6. **Evaluate** - Test on held-out data
7. **Save** - Export model and metrics

## Loading a Trained Model

```python
from fastai.vision.all import load_learner

# Load the saved model
learn = load_learner("./output/best_model.pkl")

# Make predictions
predictions = learn.predict("path/to/image.png")
print(f"Predicted class: {predictions[0]}")
print(f"Confidence: {predictions[2].max():.2%}")
```

## Troubleshooting

### "CUDA out of memory"
- Reduce `BATCH_SIZE` in Config (try 16 or 8)
- Use ResNet34 instead of ResNet50

### "No GPU found"
- Install CUDA toolkit
- Check `torch.cuda.is_available()` returns True

### Corrupt images cause errors
- The pipeline automatically skips corrupt images
- Check logs for which files were skipped

<br>
