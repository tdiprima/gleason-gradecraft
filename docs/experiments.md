## Experiment "1a"

```c
📊 Final Test Metrics:
   Accuracy:   0.7695
   F1 (macro): 0.5974
   Precision:  0.5882
   Recall:     0.6117
```

<br>

---

## Experiment "1"

| Metric | Value | Assessment |
|--------|-------|------------|
| **Accuracy** | 77% | Good overall |
| **F1 Macro** | 60% | Moderate - affected by class imbalance |
| **Precision** | 59% | Some false positives |
| **Recall** | 61% | Missing some true positives |

The gap between accuracy (77%) and F1 macro (60%) suggests the model does well on majority classes (Benign, Gleason\_3) but struggles with minority classes (Gleason\_4, Gleason\_5\_Single_Cells).

Duration: 1.3h

gleason_classifier.py commit 266b917

best\_learning_rate:  
0.00011989375667768034

best\_mixup:  
0.00043255619424744653

best\_weight_decay:  
0.018226241522626342

best\_architecture:  
resnet50

---

## Experiment "2"

gleason_classifier.py commit 5c160ce (started over)

Found the GPU, accidentally didn't use it. :p  
Total execution time: 6h 28m 7.9s

### HYPERPARAMETER OPTIMIZATION COMPLETE

After Trial 20 completes, the script will:

1. Pick the best hyperparameters from all 20 trials
1. Train one final model with those settings
1. Evaluate on the test set
1. Save the model and output confusion matrix/metrics

```c
Best trial accuracy: [wrong value]

Best hyperparameters:
  learning_rate: 0.007750014647242031
  batch_size: 16
  epochs: 15
```

### TRAINING FINAL MODEL WITH BEST HYPERPARAMETERS

Final model validation accuracy: 96.8372

### EVALUATING MODEL ON TEST SET

Weighted F1 Score: 0.9234

```c
(Rows = True Labels, Columns = Predicted Labels)
            Benign Gleason  Gleason  Gleason
  Benign:     2917      103        9        5
Gleason :      141     3694      163       10
Gleason :       18      149     1008       34
Gleason :        1        1        8      108
```

```c
Class                 Precision     Recall   F1-Score    Support
--------------------------------------------------------------
Benign                    0.948      0.961      0.955       3034
Gleason 3                 0.936      0.922      0.929       4008
Gleason 4                 0.848      0.834      0.841       1209
Gleason 5-Single Cells    0.688      0.915      0.785        118
```

### TRAINING COMPLETE - SUMMARY

```c
Best validation accuracy: 96.8372
Test accuracy: 0.9233
Test F1 score: 0.9234
```

This one did much better than the previous ones. 🥳

---

These are **excellent results** for a medical imaging classifier! Here's my analysis:

## Overall Performance
- **92.3% test accuracy** is very strong for a 4-class histopathology problem
- Train (96.8%) vs Test (92.3%) gap is reasonable - not overfitting badly

## Per-Class Breakdown

| Class | Verdict | Notes |
|-------|---------|-------|
| **Benign** | ✅ Excellent | 95.5% F1, rarely confused with cancer grades |
| **Gleason 3** | ✅ Very Good | 92.9% F1, some confusion with Gleason 4 (expected - they're adjacent grades) |
| **Gleason 4** | ⚠️ Good | 84.1% F1, hardest to classify (gets confused with both G3 and G5) |
| **Gleason 5-Single Cells** | 🎉 Surprising | 91.5% recall despite only 815 training samples! The class weights worked. |

## Key Observations

1. **The class weighting worked beautifully** - Gleason 5-Single Cells had the fewest samples (815) but achieved 91.5% recall. Without weighting, this class would likely be ignored.

2. **Confusion makes clinical sense** - Most errors are between adjacent Gleason grades (3↔4, 4↔5), which is exactly where human pathologists also disagree.

3. **Low precision on Gleason 5 (68.8%)** - The model over-predicts this class sometimes (49 false positives). This is a tradeoff from the class weighting pushing it to not miss rare cases.

## Is This Good Enough?

For a **screening/research tool**: Yes, this is very usable.

For **clinical diagnosis**: You'd want a pathologist to review, especially for Gleason 4/5 predictions. The model is better as a "second opinion" than a replacement.

<br>
