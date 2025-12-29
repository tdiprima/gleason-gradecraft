Solid initial results! Here's a breakdown:

## Current Performance

| Metric | Value | Assessment |
|--------|-------|------------|
| **Accuracy** | 77% | Good overall |
| **F1 Macro** | 60% | Moderate - affected by class imbalance |
| **Precision** | 59% | Some false positives |
| **Recall** | 61% | Missing some true positives |

The gap between accuracy (77%) and F1 macro (60%) suggests the model does well on majority classes (Benign, Gleason\_3) but struggles with minority classes (Gleason\_4, Gleason\_5\_Single_Cells).

Duration: 1.3h

gleason_classifier.py commit 266b917

best\_learning_rate  
0.00011989375667768034

best\_mixup  
0.00043255619424744653

best\_weight_decay  
0.018226241522626342

best\_architecture  
resnet50

<br>
