#!/usr/bin/env python3
"""
Gleason Score Image Classifier
==============================
Uses FastAI for training, Optuna for hyperparameter tuning, and MLflow for tracking.

Features:
- Skips corrupt images automatically
- Extracts labels from filename suffix (_0.png through _5.png)
- Splits data into train/validation/test sets
- Hyperparameter optimization with Optuna
- Experiment tracking with MLflow
- Outputs confusion matrix, per-class metrics, and saves best model
"""

import os
import sys
import warnings
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List
import json

import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

from fastai.vision.all import *
from fastai.callback.tracker import SaveModelCallback
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import optuna
from optuna.integration import FastAIPruningCallback
import mlflow
import mlflow.fastai

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Central configuration for the training pipeline."""
    
    # Data paths - UPDATE THIS TO YOUR IMAGE FOLDER
    IMAGE_DIR = Path("./gleason_images")  # Change to your actual path
    OUTPUT_DIR = Path("./output")
    
    # Label mapping from filename suffix
    LABEL_MAP = {
        0: "benign",
        1: "gleason_3_3",
        2: "gleason_3_4", 
        3: "gleason_4_3",
        4: "gleason_4_4",
        5: "gleason_4_5_5_4_5_5"
    }
    
    # You can also use numeric labels if preferred
    USE_NUMERIC_LABELS = False
    
    # Data splits
    TEST_SIZE = 0.15       # 15% for final test
    VALID_SIZE = 0.15      # 15% of remaining for validation
    RANDOM_SEED = 42
    
    # Training defaults (Optuna will override these)
    DEFAULT_BATCH_SIZE = 32
    DEFAULT_LR = 1e-3
    DEFAULT_EPOCHS = 10
    DEFAULT_ARCH = "resnet34"
    IMAGE_SIZE = 224
    
    # Optuna settings
    N_OPTUNA_TRIALS = 20
    OPTUNA_TIMEOUT = 3600 * 2  # 2 hours max
    
    # MLflow settings
    MLFLOW_EXPERIMENT = "gleason_classifier"
    MLFLOW_TRACKING_URI = "./mlruns"


# =============================================================================
# DATA PREPARATION
# =============================================================================

def is_valid_image(filepath: Path) -> bool:
    """Check if an image file is valid and not corrupt."""
    try:
        with Image.open(filepath) as img:
            img.verify()
        # Re-open after verify (verify() makes file unusable)
        with Image.open(filepath) as img:
            img.load()
        return True
    except Exception as e:
        print(f"  ⚠ Corrupt image skipped: {filepath.name} ({e})")
        return False


def extract_label_from_filename(filename: str) -> Optional[int]:
    """Extract the Gleason label from filename suffix like 'image_0.png'."""
    stem = Path(filename).stem
    for suffix in range(6):  # _0 through _5
        if stem.endswith(f"_{suffix}"):
            return suffix
    return None


def prepare_dataset(image_dir: Path, config: Config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Scan image directory, validate images, extract labels, and split into train/val/test.
    
    Returns:
        Tuple of (train_df, valid_df, test_df)
    """
    print("\n" + "="*60)
    print("📁 PREPARING DATASET")
    print("="*60)
    
    # Find all PNG files
    all_images = list(image_dir.glob("*.png"))
    print(f"\nFound {len(all_images)} PNG files in {image_dir}")
    
    if len(all_images) == 0:
        print(f"\n❌ No PNG files found in {image_dir}")
        print("Please update Config.IMAGE_DIR to point to your image folder.")
        sys.exit(1)
    
    # Validate images and extract labels
    print("\nValidating images and extracting labels...")
    valid_data = []
    corrupt_count = 0
    label_counts = {i: 0 for i in range(6)}
    
    for i, img_path in enumerate(all_images):
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i+1}/{len(all_images)} images...")
        
        label = extract_label_from_filename(img_path.name)
        if label is None:
            continue
            
        if is_valid_image(img_path):
            label_name = config.LABEL_MAP[label] if not config.USE_NUMERIC_LABELS else str(label)
            valid_data.append({
                "filepath": str(img_path),
                "label": label_name,
                "label_num": label
            })
            label_counts[label] += 1
        else:
            corrupt_count += 1
    
    print(f"\n✓ Valid images: {len(valid_data)}")
    print(f"✗ Corrupt/skipped: {corrupt_count}")
    
    print("\n📊 Class distribution:")
    for label_num, count in label_counts.items():
        label_name = config.LABEL_MAP[label_num]
        pct = 100 * count / len(valid_data) if valid_data else 0
        print(f"   {label_num} ({label_name:20s}): {count:6d} ({pct:5.1f}%)")
    
    # Create DataFrame
    df = pd.DataFrame(valid_data)
    
    # Stratified split: first separate test set
    train_val_df, test_df = train_test_split(
        df, 
        test_size=config.TEST_SIZE,
        stratify=df['label'],
        random_state=config.RANDOM_SEED
    )
    
    # Then split train_val into train and validation
    train_df, valid_df = train_test_split(
        train_val_df,
        test_size=config.VALID_SIZE / (1 - config.TEST_SIZE),
        stratify=train_val_df['label'],
        random_state=config.RANDOM_SEED
    )
    
    print(f"\n📂 Data splits:")
    print(f"   Train:      {len(train_df):6d} ({100*len(train_df)/len(df):.1f}%)")
    print(f"   Validation: {len(valid_df):6d} ({100*len(valid_df)/len(df):.1f}%)")
    print(f"   Test:       {len(test_df):6d} ({100*len(test_df)/len(df):.1f}%)")
    
    return train_df, valid_df, test_df


# =============================================================================
# MODEL TRAINING
# =============================================================================

def get_architecture(arch_name: str):
    """Get FastAI architecture from string name."""
    architectures = {
        "resnet18": resnet18,
        "resnet34": resnet34,
        "resnet50": resnet50,
        "densenet121": densenet121,
        "efficientnet_b0": efficientnet_b0 if hasattr(sys.modules[__name__], 'efficientnet_b0') else resnet34,
    }
    return architectures.get(arch_name, resnet34)


def create_dataloaders(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    config: Config,
    batch_size: int = 32,
    image_size: int = 224
) -> DataLoaders:
    """Create FastAI DataLoaders from DataFrames."""
    
    # Combine for DataBlock
    combined_df = pd.concat([
        train_df.assign(is_valid=False),
        valid_df.assign(is_valid=True)
    ]).reset_index(drop=True)
    
    # Define DataBlock
    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_x=ColReader('filepath'),
        get_y=ColReader('label'),
        splitter=ColSplitter('is_valid'),
        item_tfms=Resize(image_size),
        batch_tfms=[
            *aug_transforms(
                mult=1.0,
                do_flip=True,
                flip_vert=True,  # Pathology images can be any orientation
                max_rotate=15.0,
                max_zoom=1.1,
                max_warp=0.1,
                max_lighting=0.2,
            ),
            Normalize.from_stats(*imagenet_stats)
        ]
    )
    
    return dblock.dataloaders(combined_df, bs=batch_size)


def train_model(
    dls: DataLoaders,
    arch_name: str = "resnet34",
    lr: float = 1e-3,
    epochs: int = 10,
    freeze_epochs: int = 1,
    wd: float = 0.01,
    mixup: float = 0.0,
    save_path: Optional[Path] = None,
    trial: Optional[optuna.Trial] = None
) -> Tuple[Learner, dict]:
    """
    Train a FastAI model with the given hyperparameters.
    
    Returns:
        Tuple of (learner, metrics_dict)
    """
    arch = get_architecture(arch_name)
    
    # Build learner with metrics
    cbs = []
    if trial is not None:
        cbs.append(FastAIPruningCallback(trial, monitor='valid_loss'))
    
    if save_path:
        cbs.append(SaveModelCallback(monitor='valid_loss', fname='best_model'))
    
    learn = cnn_learner(
        dls,
        arch,
        metrics=[
            accuracy,
            F1Score(average='macro'),
            Precision(average='macro'),
            Recall(average='macro')
        ],
        wd=wd,
        cbs=cbs
    )
    
    # Optional mixup
    if mixup > 0:
        learn = learn.to_fp16().add_cb(MixUp(mixup))
    
    # Train: freeze first, then unfreeze
    learn.freeze()
    learn.fit_one_cycle(freeze_epochs, lr)
    
    learn.unfreeze()
    learn.fit_one_cycle(epochs - freeze_epochs, slice(lr/100, lr/10))
    
    # Get final metrics
    val_loss, acc, f1, prec, rec = learn.validate()
    
    metrics = {
        "valid_loss": float(val_loss),
        "accuracy": float(acc),
        "f1_macro": float(f1),
        "precision_macro": float(prec),
        "recall_macro": float(rec)
    }
    
    return learn, metrics


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_model(
    learn: Learner,
    test_df: pd.DataFrame,
    config: Config,
    output_dir: Path
) -> dict:
    """
    Evaluate model on test set and generate reports.
    
    Returns:
        Dictionary of test metrics
    """
    print("\n" + "="*60)
    print("📊 EVALUATING ON TEST SET")
    print("="*60)
    
    # Create test dataloader
    test_dl = learn.dls.test_dl(test_df['filepath'].tolist())
    
    # Get predictions
    preds, _ = learn.get_preds(dl=test_dl)
    pred_labels = preds.argmax(dim=1).numpy()
    
    # Map predictions back to class names
    vocab = learn.dls.vocab
    pred_names = [vocab[i] for i in pred_labels]
    true_names = test_df['label'].tolist()
    
    # Classification report
    print("\n📋 Classification Report:")
    print("-" * 60)
    report = classification_report(true_names, pred_names, output_dict=True, zero_division=0)
    print(classification_report(true_names, pred_names, zero_division=0))
    
    # Save report
    report_path = output_dir / "classification_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n✓ Report saved to {report_path}")
    
    # Confusion Matrix
    cm = confusion_matrix(true_names, pred_names, labels=vocab)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=vocab,
        yticklabels=vocab
    )
    plt.title('Confusion Matrix - Test Set', fontsize=14)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    cm_path = output_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"✓ Confusion matrix saved to {cm_path}")
    
    # Per-class metrics summary
    print("\n📈 Per-Class Metrics:")
    print("-" * 60)
    print(f"{'Class':<25} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("-" * 60)
    
    for class_name in vocab:
        if class_name in report:
            m = report[class_name]
            print(f"{class_name:<25} {m['precision']:>10.3f} {m['recall']:>10.3f} {m['f1-score']:>10.3f} {m['support']:>10.0f}")
    
    print("-" * 60)
    print(f"{'Macro Avg':<25} {report['macro avg']['precision']:>10.3f} {report['macro avg']['recall']:>10.3f} {report['macro avg']['f1-score']:>10.3f}")
    print(f"{'Weighted Avg':<25} {report['weighted avg']['precision']:>10.3f} {report['weighted avg']['recall']:>10.3f} {report['weighted avg']['f1-score']:>10.3f}")
    
    return {
        "test_accuracy": report['accuracy'],
        "test_f1_macro": report['macro avg']['f1-score'],
        "test_precision_macro": report['macro avg']['precision'],
        "test_recall_macro": report['macro avg']['recall']
    }


# =============================================================================
# OPTUNA HYPERPARAMETER OPTIMIZATION
# =============================================================================

def optuna_objective(trial: optuna.Trial, train_df: pd.DataFrame, valid_df: pd.DataFrame, config: Config) -> float:
    """Optuna objective function for hyperparameter optimization."""
    
    # Suggest hyperparameters
    arch_name = trial.suggest_categorical('architecture', ['resnet18', 'resnet34', 'resnet50'])
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    epochs = trial.suggest_int('epochs', 5, 15)
    freeze_epochs = trial.suggest_int('freeze_epochs', 1, 3)
    wd = trial.suggest_float('weight_decay', 1e-4, 1e-1, log=True)
    mixup = trial.suggest_float('mixup', 0.0, 0.4)
    image_size = trial.suggest_categorical('image_size', [224, 256, 320])
    
    # Log trial parameters
    print(f"\n🔬 Trial {trial.number}: arch={arch_name}, bs={batch_size}, lr={lr:.2e}, epochs={epochs}")
    
    try:
        # Create dataloaders with trial params
        dls = create_dataloaders(train_df, valid_df, config, batch_size, image_size)
        
        # Train model
        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
            mlflow.log_params(trial.params)
            
            learn, metrics = train_model(
                dls=dls,
                arch_name=arch_name,
                lr=lr,
                epochs=epochs,
                freeze_epochs=freeze_epochs,
                wd=wd,
                mixup=mixup,
                trial=trial
            )
            
            mlflow.log_metrics(metrics)
            
            # Clean up GPU memory
            del learn
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return metrics['f1_macro']
            
    except Exception as e:
        print(f"  ❌ Trial failed: {e}")
        return 0.0


def run_hyperparameter_search(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    config: Config
) -> dict:
    """Run Optuna hyperparameter optimization."""
    
    print("\n" + "="*60)
    print("🔍 HYPERPARAMETER OPTIMIZATION (OPTUNA)")
    print("="*60)
    print(f"\nRunning {config.N_OPTUNA_TRIALS} trials (timeout: {config.OPTUNA_TIMEOUT}s)")
    
    # Create Optuna study
    study = optuna.create_study(
        direction='maximize',
        study_name='gleason_classifier',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=2)
    )
    
    # Optimize
    study.optimize(
        lambda trial: optuna_objective(trial, train_df, valid_df, config),
        n_trials=config.N_OPTUNA_TRIALS,
        timeout=config.OPTUNA_TIMEOUT,
        show_progress_bar=True
    )
    
    # Report results
    print("\n" + "="*60)
    print("🏆 BEST HYPERPARAMETERS")
    print("="*60)
    print(f"\nBest F1 Score: {study.best_value:.4f}")
    print("\nBest parameters:")
    for key, value in study.best_params.items():
        print(f"   {key}: {value}")
    
    return study.best_params


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    """Main training pipeline."""
    
    print("\n" + "="*60)
    print("🔬 GLEASON SCORE CLASSIFIER")
    print("    FastAI + Optuna + MLflow")
    print("="*60)
    
    config = Config()
    
    # Create output directory
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Setup MLflow
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
    
    # Prepare dataset
    train_df, valid_df, test_df = prepare_dataset(config.IMAGE_DIR, config)
    
    # Save splits for reproducibility
    train_df.to_csv(config.OUTPUT_DIR / "train_split.csv", index=False)
    valid_df.to_csv(config.OUTPUT_DIR / "valid_split.csv", index=False)
    test_df.to_csv(config.OUTPUT_DIR / "test_split.csv", index=False)
    print(f"\n✓ Data splits saved to {config.OUTPUT_DIR}")
    
    # Start main MLflow run
    with mlflow.start_run(run_name=f"gleason_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        # Log config
        mlflow.log_params({
            "test_size": config.TEST_SIZE,
            "valid_size": config.VALID_SIZE,
            "n_train": len(train_df),
            "n_valid": len(valid_df),
            "n_test": len(test_df),
            "n_optuna_trials": config.N_OPTUNA_TRIALS
        })
        
        # Run hyperparameter search
        best_params = run_hyperparameter_search(train_df, valid_df, config)
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        
        # Train final model with best parameters
        print("\n" + "="*60)
        print("🚀 TRAINING FINAL MODEL WITH BEST PARAMETERS")
        print("="*60)
        
        dls = create_dataloaders(
            train_df, valid_df, config,
            batch_size=best_params.get('batch_size', 32),
            image_size=best_params.get('image_size', 224)
        )
        
        learn, metrics = train_model(
            dls=dls,
            arch_name=best_params.get('architecture', 'resnet34'),
            lr=best_params.get('learning_rate', 1e-3),
            epochs=best_params.get('epochs', 10),
            freeze_epochs=best_params.get('freeze_epochs', 1),
            wd=best_params.get('weight_decay', 0.01),
            mixup=best_params.get('mixup', 0.0),
            save_path=config.OUTPUT_DIR
        )
        
        mlflow.log_metrics({f"final_{k}": v for k, v in metrics.items()})
        
        # Evaluate on test set
        test_metrics = evaluate_model(learn, test_df, config, config.OUTPUT_DIR)
        mlflow.log_metrics(test_metrics)
        
        # Save final model
        model_path = config.OUTPUT_DIR / "gleason_classifier_final.pkl"
        learn.export(model_path)
        print(f"\n✓ Model saved to {model_path}")
        mlflow.log_artifact(str(model_path))
        
        # Log confusion matrix
        mlflow.log_artifact(str(config.OUTPUT_DIR / "confusion_matrix.png"))
        mlflow.log_artifact(str(config.OUTPUT_DIR / "classification_report.json"))
        
        # Summary
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETE")
        print("="*60)
        print(f"\n📊 Final Test Metrics:")
        print(f"   Accuracy:  {test_metrics['test_accuracy']:.4f}")
        print(f"   F1 (macro): {test_metrics['test_f1_macro']:.4f}")
        print(f"   Precision: {test_metrics['test_precision_macro']:.4f}")
        print(f"   Recall:    {test_metrics['test_recall_macro']:.4f}")
        print(f"\n📁 Outputs saved to: {config.OUTPUT_DIR}")
        print(f"🔗 MLflow UI: run 'mlflow ui --backend-store-uri {config.MLFLOW_TRACKING_URI}'")


if __name__ == "__main__":
    main()
