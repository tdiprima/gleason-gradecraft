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
uv sync
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

### Option 1: FastAI (.pkl) - Easy Python inference

```python
from fastai.vision.all import *

# Load model
learn = load_learner('output/gleason_classifier_final.pkl')

# Predict on new image
pred_class, pred_idx, probs = learn.predict('path/to/new_image.png')
print(f"Predicted: {pred_class} (confidence: {probs[pred_idx]:.2%})")
```

### Option 2: TorchScript (.pt) - Production deployment

TorchScript models are better for production:

- No FastAI/Python dependency at inference
- Can run in C++, Java, mobile (via PyTorch Mobile)
- Optimized and smaller file size

```python
import torch
from torchvision import transforms
from PIL import Image
import json

# Load model and labels
model = torch.jit.load('output/gleason_classifier_final.pt')
model.eval()

with open('output/class_labels.json') as f:
    labels = json.load(f)

# Preprocessing (must match training)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Predict
img = Image.open('test_image.png').convert('RGB')
input_tensor = transform(img).unsqueeze(0)

with torch.no_grad():
    output = model(input_tensor)
    pred_idx = output.argmax(dim=1).item()
    confidence = torch.softmax(output, dim=1)[0, pred_idx].item()

print(f"Predicted: {labels[str(pred_idx)]} ({confidence:.1%})")
```

### Option 3: GPU inference with TorchScript

```python
model = torch.jit.load('output/gleason_classifier_final.pt')
model = model.cuda()
input_tensor = transform(img).unsqueeze(0).cuda()

with torch.no_grad():
    output = model(input_tensor)
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

The script is **optimized for GPU training** with:

- **Mixed precision (FP16)** - 2x faster training on modern GPUs
- **cuDNN benchmark mode** - automatically finds fastest convolution algorithms
- **TF32 support** - enabled for Ampere+ GPUs (RTX 30xx, A100, etc.)
- **Large batch sizes** - Optuna searches 32, 64, 128, 256

### Hardware Configuration

The script is configured for **64 CPU workers** for data loading. Edit these in `Config` if needed:

```python
class Config:
    NUM_WORKERS = 64       # CPU workers for data loading (set to cores - 4)
    PIN_MEMORY = True      # Faster GPU transfer
    PREFETCH_FACTOR = 4    # Batches to prefetch per worker
    USE_FP16 = True        # Mixed precision training
```

### Verify GPU detection

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

At startup, the script prints hardware info:

```
🖥️  GPU: NVIDIA A100-SXM4-80GB (80.0 GB) x 1
💻 CPU cores available: 72
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
