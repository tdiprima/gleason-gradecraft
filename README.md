# Gleason Gradecraft 🧙‍♀️

## TL;DR

**What it is:**  
A pipeline that trains a deep-learning model to classify prostate cancer image patches into Gleason scores.

**What you do:**  
Drop labeled image files in a folder → run one command → get a trained model + metrics.

## Run it

1. **You need**

   * Python 3.11+
   * NVIDIA GPU + CUDA
   * Preferably 8GB+ VRAM

2. **Your data must be named like this**

   ```
   anything_{label}.png
   ```

   Example:

   ```
   patient001_patch042_2.png
   ```

   Labels used: **0–3 only**  
   (4 & 5 exist but are skipped)

3. **Put images here**

   ```
   ./gleason_images/
   ```

4. **Install + run**

   ```bash
   uv sync
   uv run gleason_classifier.py
   ```

Go do something else. Training takes a bit.

## What comes out the other end

In `./output/` you get:

* `best_model.pkl` → the trained model
* confusion matrix
* per-class precision/recall/F1
* full training logs

Translation: **model + receipts**.

## How it works (no fluff)

* Uses **ResNet (pretrained)** so it's not learning from scratch
* Uses **Optuna** to auto-tune training settings
* Handles **class imbalance** so rare cancer types aren't ignored
* Splits data: 70% train / 15% val / 15% test
* Saves everything reproducibly

## The only knob you'll probably touch

```python
BATCH_SIZE = 32
```

If your GPU cries → lower it.

## Common failure modes

* **CUDA OOM** → lower batch size
* **No GPU detected** → CUDA install issue
* **Bad images** → skipped automatically (check logs)

<br>
