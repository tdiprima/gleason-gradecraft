## Experiment "1a"

```c
📊 Final Test Metrics:
   Accuracy:   0.7695
   F1 (macro): 0.5974
   Precision:  0.5882
   Recall:     0.6117
```

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

## Experiment "2"

gleason_classifier.py commit 5c160ce (started over)

Found the GPU, accidentally didn't use it. :p  
Execution time: 

HYPERPARAMETER OPTIMIZATION COMPLETE

```c
Best trial accuracy: [wrong value]

Best hyperparameters:
  learning_rate: 0.007750014647242031
  batch_size: 16
  epochs: 15
```

TRAINING FINAL MODEL WITH BEST HYPERPARAMETERS



<br>
