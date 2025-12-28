# Gleason Score Image Classifier

A complete pipeline for training a Gleason score classifier using **FastAI**, **Optuna**, and **MLflow**.

## Features

- ✅ **Automatic corrupt image detection** - skips invalid PNGs
- ✅ **Label extraction from filenames** - parses `_0.png` through `_5.png` suffixes
- ✅ **Stratified train/validation/test splits** - maintains class distribution
- ✅ **Hyperparameter optimization** - Optuna finds optimal architecture, learning rate, batch size, etc.
- ✅ **Experiment tracking** - MLflow logs all runs, parameters, and artifacts
- ✅ **Comprehensive evaluation** - confusion matrix, per-class metrics, classification report
- ✅ **Model export** - saves best model as `.pkl` for inference

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Organize Your Images

Place all your PNG images in a single folder. The script extracts labels from filename suffixes:

| Suffix | Label |
|--------|-------|
| `*_0.png` | benign |
| `*_1.png` | gleason_3_3 |
| `*_2.png` | gleason_3_4 |
| `*_3.png` | gleason_4_3 |
| `*_4.png` | gleason_4_4 |
| `*_5.png` | gleason_4_5_5_4_5_5 |

### 3. Configure the Script

Edit the `Config` class in `gleason_classifier.py`:

```python
class Config:
    IMAGE_DIR = Path("/path/to/your/gleason_images")  # <-- Update this
    OUTPUT_DIR = Path("./output")
    
    # Adjust label names if needed
    LABEL_MAP = {
        0: "benign",
        1: "gleason_3_3",
        # ...
    }
    
    # Optuna settings
    N_OPTUNA_TRIALS = 20  # More trials = better hyperparameters (but slower)
```

### 4. Run Training

```bash
python gleason_classifier.py
```

### 5. View Results

**MLflow UI:**

```bash
mlflow ui --backend-store-uri ./mlruns
# Open http://localhost:5000 in browser
```

**Output files in `./output/`:**
- `gleason_classifier_final.pkl` - Trained model (use with `fastai.load_learner()`)
- `confusion_matrix.png` - Visual confusion matrix
- `classification_report.json` - Per-class precision/recall/F1
- `train_split.csv`, `valid_split.csv`, `test_split.csv` - Data splits for reproducibility

## Using the Trained Model

```python
from fastai.vision.all import *

# Load model
learn = load_learner('output/gleason_classifier_final.pkl')

# Predict on new image
pred_class, pred_idx, probs = learn.predict('path/to/new_image.png')
print(f"Predicted: {pred_class} (confidence: {probs[pred_idx]:.2%})")
```

## Hyperparameters Tuned by Optuna

| Parameter | Search Space |
|-----------|--------------|
| Architecture | ResNet18, ResNet34, ResNet50 |
| Batch size | 16, 32, 64 |
| Learning rate | 1e-5 to 1e-2 (log scale) |
| Epochs | 5 to 15 |
| Freeze epochs | 1 to 3 |
| Weight decay | 1e-4 to 1e-1 (log scale) |
| Mixup alpha | 0.0 to 0.4 |
| Image size | 224, 256, 320 |

## Handling Class Imbalance

Your dataset has significant class imbalance (815 `_3.png` vs 26,816 `_1.png`). The script addresses this with:

1. **Stratified splitting** - ensures all classes represented in train/val/test
2. **Data augmentation** - flips, rotations, zoom help minority classes
3. **Mixup regularization** - Optuna can enable this for better generalization
4. **Macro-averaged metrics** - F1, precision, recall treat all classes equally

For severe imbalance, consider adding weighted loss (modify `train_model()`):

```python
# Add to train_model() after creating learner
class_weights = torch.tensor([...])  # Higher weights for rare classes
learn.loss_func = CrossEntropyLossFlat(weight=class_weights)
```

## GPU Support

The script automatically uses GPU if available (CUDA). For CPU-only training, it works but will be slower.

Check GPU availability:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
```

## Troubleshooting

**Out of memory:**

- Reduce `batch_size` in Config or Optuna search space
- Use smaller `image_size` (e.g., 192)
- Use smaller architecture (ResNet18)

**Training too slow:**

- Reduce `N_OPTUNA_TRIALS`
- Reduce `epochs` range in Optuna
- Use fewer images for initial testing

**Poor accuracy on minority classes:**

- Increase `N_OPTUNA_TRIALS` for better hyperparameters
- Try adding class weights (see above)
- Consider oversampling minority classes

<br>
