# 🧠🧪 Gleason Score Image Classifier

**Teaching a computer to read prostate cancer slides (so we don't have to squint forever)**

This repo is a **machine learning pipeline** that looks at histopathology images and predicts **Gleason scores**.

No magic. No buzzwords. Just computers being trained to notice patterns.

---

## ✨ What this actually does (in plain English)

* You give it microscope image patches
* It learns what *benign vs aggressive cancer* looks like
* It spits out predictions + receipts (metrics, logs, confusion matrices)

---

## 🧰 What's under the hood

We use some battle-tested tools:

* **FastAI** → makes deep learning way less painful
* **Optuna** → automatically tries different settings so *you don't have to guess*
* **ResNet** → a proven image-recognition backbone that actually works

You don't need to be an ML wizard. This pipeline handles the boring stuff.

---

## 🏷️ Gleason Labels (aka "what the numbers mean")

| Label | Meaning                              |
| ----: | ------------------------------------ |
|     0 | Benign                               |
|     1 | Gleason 3                            |
|     2 | Gleason 4                            |
|     3 | Gleason 5 – Single Cells             |
|     4 | Gleason 5 – Secretions (**SKIPPED**) |
|     5 | Gleason 5 (**SKIPPED**)              |

⚠️ Labels 4 and 5 exist in theory but are **not used** in training.

---

## 💻 Requirements (aka “will this run on my machine?”)

* Python **3.9+**
* NVIDIA GPU with CUDA
* **8GB+ GPU memory** recommended  
  (More is better. Your GPU will thank you.)

---

## ⚙️ Installation (2 minutes, promise)

```bash
# Create a virtual environment (strongly recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install everything
uv sync
```

If this fails, it's usually CUDA or driver-related. See troubleshooting below.

---

## 🖼️ Data Setup (important!)

1. Put your images in:

   ```
   ./gleason_images/
   ```

2. File naming **must** look like this:

   ```
   anything_you_want_{label}.png
   ```

   Example:

   ```
   patient001_patch042_2.png
   ```

   → label = `2` → Gleason 4

If filenames are wrong, the model will be confused. And confused models do bad things.

---

## 🚀 Run It

```bash
uv run gleason_classifier.py
```

That's it.  
Go get coffee. This part takes time.

---

## 🛠️ Configuration (the knobs you're allowed to turn)

Open `gleason_classifier.py` and find:

```python
class Config:
    DATA_PATH = Path("./gleason_images")
    OUTPUT_PATH = Path("./output")
    NUM_TRIALS = 20
    BATCH_SIZE = 32
```

What these do:

* **DATA_PATH** → where your images live
* **OUTPUT_PATH** → where results go
* **NUM_TRIALS** → how hard Optuna tries to find good settings
* **BATCH_SIZE** → lower this if your GPU screams

---

## 📦 What you get after training

Everything lands in `./output/`:

* 🧠 `best_model.pkl` → the trained model
* 📝 `training_log_*.txt` → full training history
* 📊 `confusion_matrix.csv` → what it got right vs wrong
* 📈 `per_class_metrics.csv` → precision, recall, F1 per class

Translation: **you get proof, not vibes**.

---

## 🧩 How the code thinks (no math)

### Key ideas you should know

**1️⃣ Class Weights**
Some Gleason classes are rare.
We tell the model:

"Hey, don't ignore the rare stuff."

**2️⃣ Data Split**

* **70% training** → learns patterns
* **15% validation** → checks itself while learning
* **15% test** → final exam (never seen before)

**3️⃣ Hyperparameters**
Training settings like learning rate.
Optuna tries different combos so you don't have to guess.

**4️⃣ Transfer Learning**
We start with a model that already knows how images work,
then teach it pathology.
Faster. Better. Less data needed.

---

## 🔁 Pipeline flow (big picture)

1. Setup logging + GPU checks
2. Load images & labels
3. Handle class imbalance
4. Hyperparameter search (Optuna)
5. Train final model
6. Evaluate on unseen data
7. Save everything

No step is skipped. No vibes-only training.

---

## 🔮 Load a trained model later

```python
from fastai.vision.all import load_learner

learn = load_learner("./output/best_model.pkl")

pred = learn.predict("path/to/image.png")
print(f"Predicted class: {pred[0]}")
print(f"Confidence: {pred[2].max():.2%}")
```

Yes, it always gives an answer.  
No, that doesn't mean it's always right.

---

## 🧯 Troubleshooting (aka “when things break”)

### ❌ CUDA out of memory

* Lower `BATCH_SIZE` (try 16 or 8)
* Switch to **ResNet34** instead of ResNet50

### ❌ No GPU found

* Install CUDA toolkit
* Verify:

  ```python
  torch.cuda.is_available()
  ```

### ❌ Crashes on bad images

* Corrupt images are **automatically skipped**
* Check logs to see which ones were ignored

---

## 🧠 Final vibe check

This repo is:

* practical
* reproducible
* not pretending ML is magic

If you're here to **actually understand what your model is doing**,  
you're in the right place.

<br>
