"""
Gleason Score Image Classification Pipeline
============================================
This script trains a deep learning model to classify prostate cancer images
by their Gleason score using FastAI and Optuna for hyperparameter tuning.

Gleason Scores (labels):
    0: Benign
    1: Gleason 3
    2: Gleason 4
    3: Gleason 5-Single Cells
    4: Gleason 5-Secretions (SKIPPED)
    5: Gleason 5 (SKIPPED)

Author: Generated for educational purposes
"""

import os
import re
import time
import logging
import warnings
from pathlib import Path
from datetime import datetime
from collections import Counter

import torch
import numpy as np
import pandas as pd
import optuna
from PIL import Image

from fastai.vision.all import (
    ImageDataLoaders,
    Learner,
    accuracy,
    error_rate,
    vision_learner,
    resnet50,
    CrossEntropyLossFlat,
    ClassificationInterpretation,
    SaveModelCallback,
    EarlyStoppingCallback,
    set_seed,
)
from fastai.torch_core import default_device, set_default_device
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
)

# Suppress unnecessary warnings for cleaner output
warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION - Edit these settings for your setup
# =============================================================================

class Config:
    """All settings in one place for easy modification."""
    
    # Path to your image folder (change this to your data location)
    DATA_PATH = Path("./gleason_images")
    
    # Where to save results
    OUTPUT_PATH = Path("./output")
    
    # Labels we want to use (skipping 4 and 5)
    VALID_LABELS = [0, 1, 2, 3]
    
    # Human-readable names for each label
    LABEL_NAMES = {
        0: "Benign",
        1: "Gleason 3",
        2: "Gleason 4",
        3: "Gleason 5-Single Cells",
    }
    
    # Class counts from your data (used for calculating weights)
    CLASS_COUNTS = {
        0: 19695,
        1: 26816,
        2: 8461,
        3: 815,
    }
    
    # Data split ratios
    TRAIN_RATIO = 0.70  # 70% for training
    VALID_RATIO = 0.15  # 15% for validation
    TEST_RATIO = 0.15   # 15% for testing
    
    # Training settings
    IMAGE_SIZE = 224           # Standard size for pretrained models
    BATCH_SIZE = 32            # Adjust based on your GPU memory
    NUM_WORKERS = 4            # Parallel data loading workers
    
    # Optuna settings
    NUM_TRIALS = 20            # Number of hyperparameter combinations to try
    
    # Random seed for reproducibility
    SEED = 42
    
    # Skip image validation for faster loading (set False if you have corrupt images)
    SKIP_IMAGE_VALIDATION = True


# =============================================================================
# SETUP LOGGING
# =============================================================================

def setup_logging(output_path: Path) -> logging.Logger:
    """
    Set up logging to both file and console.
    
    This helps you track what's happening during training
    and review results later.
    """
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_path / f"training_log_{timestamp}.txt"
    
    # Configure logging format
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    
    # Set up handlers for file and console
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),      # Save to file
            logging.StreamHandler()              # Print to console
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to: {log_file}")
    
    return logger


# =============================================================================
# GPU SETUP
# =============================================================================

def setup_gpu(logger: logging.Logger) -> torch.device:
    """
    Check for GPU availability and set up the device.
    
    GPUs make training much faster - this function ensures
    we're using one if available and sets it as the default.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"✓ GPU Found: {gpu_name}")
        logger.info(f"  GPU Memory: {gpu_memory:.1f} GB")
        
        # THIS IS THE KEY: Set default device for FastAI
        set_default_device(device)
        logger.info(f"  Default device set to: {default_device()}")
    else:
        logger.error("✗ No GPU found! Training will be very slow.")
        logger.error("  Please ensure CUDA is installed and a GPU is available.")
        device = torch.device("cpu")
        set_default_device(device)
    
    return device


# =============================================================================
# DATA LOADING AND PREPARATION
# =============================================================================

def extract_label_from_filename(filename: str) -> int:
    """
    Extract the label from an image filename.
    
    Expected format: anything_{label}.png
    Example: "001738-000001_01_20180504-multires.tif_17922_54662_239_250_0.png" -> label 0
    """
    # Find the pattern "_{digit}.png" at the end of filename
    match = re.search(r"_(\d)\.png$", filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return -1  # Invalid label


def is_valid_image(filepath: Path, logger: logging.Logger) -> bool:
    """
    Check if an image file is valid and not corrupted.
    
    This prevents training from crashing on bad files.
    """
    try:
        with Image.open(filepath) as img:
            # Load the image data to verify it's not corrupted
            # (verify() alone doesn't always catch issues)
            img.load()
        return True
    except Exception as e:
        logger.warning(f"Skipping corrupt image: {filepath.name} - {e}")
        return False


def load_and_split_data(
    data_path: Path,
    logger: logging.Logger
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load image paths, extract labels, and split into train/valid/test sets.
    
    Returns three DataFrames containing file paths and labels.
    """
    logger.info(f"Loading data from: {data_path}")
    
    # Find all PNG files
    all_files = list(data_path.glob("**/*.png"))
    logger.info(f"Found {len(all_files)} total PNG files")
    
    # Show a sample filename for debugging
    if all_files:
        sample = all_files[0].name
        sample_label = extract_label_from_filename(sample)
        logger.info(f"Sample filename: {sample}")
        logger.info(f"Extracted label: {sample_label}")
    
    # Filter and validate images
    valid_data = []
    skipped_labels = Counter()
    corrupt_count = 0
    
    for filepath in all_files:
        # Extract label from filename
        label = extract_label_from_filename(filepath.name)
        
        # Skip invalid labels (4 and 5, or unrecognized)
        if label not in Config.VALID_LABELS:
            skipped_labels[label] += 1
            continue
        
        # Optionally skip corrupt images (slower but safer)
        if not Config.SKIP_IMAGE_VALIDATION:
            if not is_valid_image(filepath, logger):
                corrupt_count += 1
                continue
        
        valid_data.append({
            "filepath": str(filepath),
            "label": label,
            "label_name": Config.LABEL_NAMES[label]
        })
    
    # Log summary
    logger.info(f"Valid images: {len(valid_data)}")
    if not Config.SKIP_IMAGE_VALIDATION:
        logger.info(f"Corrupt images skipped: {corrupt_count}")
    for label, count in sorted(skipped_labels.items()):
        if label == -1:
            logger.info(f"Skipped (unrecognized format): {count} images")
        else:
            logger.info(f"Skipped label {label}: {count} images")
    
    # Create DataFrame and shuffle
    df = pd.DataFrame(valid_data)
    df = df.sample(frac=1, random_state=Config.SEED).reset_index(drop=True)
    
    # Calculate split indices
    n = len(df)
    train_end = int(n * Config.TRAIN_RATIO)
    valid_end = int(n * (Config.TRAIN_RATIO + Config.VALID_RATIO))
    
    # Split the data
    train_df = df[:train_end].reset_index(drop=True)
    valid_df = df[train_end:valid_end].reset_index(drop=True)
    test_df = df[valid_end:].reset_index(drop=True)
    
    # Log split sizes
    logger.info(f"Train set: {len(train_df)} images ({Config.TRAIN_RATIO*100:.0f}%)")
    logger.info(f"Valid set: {len(valid_df)} images ({Config.VALID_RATIO*100:.0f}%)")
    logger.info(f"Test set: {len(test_df)} images ({Config.TEST_RATIO*100:.0f}%)")
    
    # Check if we have any data
    if len(train_df) == 0:
        logger.error("No valid training images found! Check your data path and filename format.")
        logger.error(f"Expected filename format: *_{{label}}.png where label is in {Config.VALID_LABELS}")
        raise ValueError("No valid training images found")
    
    # Log class distribution in training set
    logger.info("Training set class distribution:")
    for label in Config.VALID_LABELS:
        count = len(train_df[train_df["label"] == label])
        logger.info(f"  {Config.LABEL_NAMES[label]}: {count}")
    
    return train_df, valid_df, test_df


# =============================================================================
# CLASS WEIGHTS FOR IMBALANCED DATA
# =============================================================================

def calculate_class_weights(logger: logging.Logger) -> torch.Tensor:
    """
    Calculate soft class weights to handle imbalanced data.
    
    Classes with fewer samples get higher weights so the model
    pays more attention to them during training.
    
    We use "soft" weights (square root) to avoid over-emphasizing
    rare classes too much.
    """
    counts = [Config.CLASS_COUNTS[i] for i in Config.VALID_LABELS]
    total = sum(counts)
    
    # Calculate inverse frequency weights
    # More samples = lower weight, fewer samples = higher weight
    weights = [total / count for count in counts]
    
    # Apply "softening" with square root to avoid extreme weights
    soft_weights = [w ** 0.5 for w in weights]
    
    # Normalize so weights sum to number of classes
    weight_sum = sum(soft_weights)
    normalized_weights = [w * len(counts) / weight_sum for w in soft_weights]
    
    # Convert to tensor and move to GPU
    weights_tensor = torch.tensor(normalized_weights, dtype=torch.float32)
    
    if torch.cuda.is_available():
        weights_tensor = weights_tensor.cuda()
    
    # Log the weights
    logger.info("Calculated soft class weights:")
    for i, label in enumerate(Config.VALID_LABELS):
        logger.info(f"  {Config.LABEL_NAMES[label]}: {normalized_weights[i]:.3f}")
    
    return weights_tensor


# =============================================================================
# CREATE DATA LOADERS
# =============================================================================

def create_dataloaders(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    batch_size: int = Config.BATCH_SIZE,
    image_size: int = Config.IMAGE_SIZE
) -> ImageDataLoaders:
    """
    Create FastAI DataLoaders for training and validation.
    
    DataLoaders handle batching, shuffling, and data augmentation.
    """
    # Combine train and valid for FastAI's expected format
    # We'll mark which rows are for validation
    train_df = train_df.copy()
    valid_df = valid_df.copy()
    train_df["is_valid"] = False
    valid_df["is_valid"] = True
    combined_df = pd.concat([train_df, valid_df], ignore_index=True)
    
    # Create DataLoaders with explicit device
    dls = ImageDataLoaders.from_df(
        df=combined_df,
        path="",                          # Base path (empty since we have full paths)
        fn_col="filepath",                # Column with file paths
        label_col="label",                # Column with labels
        valid_col="is_valid",             # Column marking validation set
        item_tfms=None,                   # Item transforms (resizing done automatically)
        batch_tfms=None,                  # Batch transforms (augmentation)
        bs=batch_size,                    # Batch size
        num_workers=Config.NUM_WORKERS,   # Parallel data loading workers
        seed=Config.SEED,                 # Random seed
        device=default_device(),          # USE GPU!
    )
    
    return dls


# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_model(
    dls: ImageDataLoaders,
    class_weights: torch.Tensor,
    learning_rate: float,
    epochs: int,
    logger: logging.Logger
) -> tuple[Learner, float]:
    """
    Train a model with the given hyperparameters.
    
    Returns the trained Learner and the best validation accuracy.
    """
    # Use ResNet50 - deeper network for complex histopathology patterns
    arch = resnet50
    
    # Create weighted loss function
    loss_func = CrossEntropyLossFlat(weight=class_weights)
    
    # Create the learner (model + data + optimizer)
    # FastAI will automatically use the default device we set earlier
    learn = vision_learner(
        dls,
        arch,
        metrics=[accuracy, error_rate],
        loss_func=loss_func,
        pretrained=True  # Use pretrained ImageNet weights
    ).to_fp16()  # Use mixed precision for faster training on GPU
    
    # Train with callbacks for saving best model and early stopping
    learn.fine_tune(
        epochs,
        base_lr=learning_rate,
        cbs=[
            SaveModelCallback(monitor="valid_loss", fname="best_model"),
            EarlyStoppingCallback(monitor="valid_loss", patience=5)
        ]
    )
    
    # Get validation accuracy by running validation
    # This is the most reliable way to get the actual accuracy
    val_loss, val_acc, val_err = learn.validate()
    best_accuracy = float(val_acc)
    
    return learn, best_accuracy


# =============================================================================
# HYPERPARAMETER OPTIMIZATION WITH OPTUNA
# =============================================================================

def objective(
    trial: optuna.Trial,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    class_weights: torch.Tensor,
    logger: logging.Logger
) -> float:
    """
    Optuna objective function - tries different hyperparameter combinations.
    
    Returns the validation accuracy to maximize.
    """
    # Suggest hyperparameters to try
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    epochs = trial.suggest_int("epochs", 5, 20)
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Trial {trial.number + 1}")
    logger.info(f"  Learning rate: {learning_rate:.6f}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Architecture: ResNet50")
    logger.info(f"{'='*50}")
    
    try:
        # Create data loaders with this batch size
        dls = create_dataloaders(train_df, valid_df, batch_size=batch_size)
        
        # Train the model
        learn, best_accuracy = train_model(
            dls=dls,
            class_weights=class_weights,
            learning_rate=learning_rate,
            epochs=epochs,
            logger=logger
        )
        
        logger.info(f"Trial {trial.number + 1} - Best Accuracy: {best_accuracy:.4f}")
        
        # Clean up GPU memory
        del learn
        torch.cuda.empty_cache()
        
        return best_accuracy
        
    except Exception as e:
        logger.error(f"Trial {trial.number + 1} failed: {e}")
        return 0.0


def run_hyperparameter_search(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    class_weights: torch.Tensor,
    logger: logging.Logger
) -> dict:
    """
    Run Optuna hyperparameter optimization.
    
    Returns the best hyperparameters found.
    """
    logger.info("\n" + "="*60)
    logger.info("STARTING HYPERPARAMETER OPTIMIZATION")
    logger.info("="*60)
    
    # Create Optuna study (we want to MAXIMIZE accuracy)
    study = optuna.create_study(
        direction="maximize",
        study_name="gleason_classification"
    )
    
    # Run optimization
    study.optimize(
        lambda trial: objective(trial, train_df, valid_df, class_weights, logger),
        n_trials=Config.NUM_TRIALS,
        show_progress_bar=True
    )
    
    # Log results
    logger.info("\n" + "="*60)
    logger.info("HYPERPARAMETER OPTIMIZATION COMPLETE")
    logger.info("="*60)
    logger.info(f"Best trial accuracy: {study.best_value:.4f}")
    logger.info("Best hyperparameters:")
    for key, value in study.best_params.items():
        logger.info(f"  {key}: {value}")
    
    return study.best_params


# =============================================================================
# MODEL EVALUATION
# =============================================================================

def evaluate_model(
    learn: Learner,
    test_df: pd.DataFrame,
    logger: logging.Logger,
    output_path: Path
) -> dict:
    """
    Evaluate the trained model on the test set.
    
    Returns a dictionary with all evaluation metrics.
    """
    logger.info("\n" + "="*60)
    logger.info("EVALUATING MODEL ON TEST SET")
    logger.info("="*60)
    
    # Create test dataloader
    test_dl = learn.dls.test_dl(test_df["filepath"].tolist())
    
    # Get predictions
    preds, targets = learn.get_preds(dl=test_dl)
    predicted_labels = preds.argmax(dim=1).cpu().numpy()
    true_labels = test_df["label"].values
    
    # Calculate metrics
    test_accuracy = (predicted_labels == true_labels).mean()
    f1 = f1_score(true_labels, predicted_labels, average="weighted")
    
    # Confusion Matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    
    # Per-class metrics
    class_report = classification_report(
        true_labels,
        predicted_labels,
        target_names=[Config.LABEL_NAMES[i] for i in Config.VALID_LABELS],
        output_dict=True
    )
    
    # Log results
    logger.info(f"\nTest Accuracy: {test_accuracy:.4f}")
    logger.info(f"Weighted F1 Score: {f1:.4f}")
    
    logger.info("\nConfusion Matrix:")
    logger.info("(Rows = True Labels, Columns = Predicted Labels)")
    header = "          " + " ".join([f"{Config.LABEL_NAMES[i][:8]:>8}" for i in Config.VALID_LABELS])
    logger.info(header)
    for i, row in enumerate(cm):
        row_str = f"{Config.LABEL_NAMES[Config.VALID_LABELS[i]][:8]:>8}: " + " ".join([f"{val:>8}" for val in row])
        logger.info(row_str)
    
    logger.info("\nPer-Class Metrics:")
    logger.info(f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    logger.info("-" * 62)
    for label in Config.VALID_LABELS:
        name = Config.LABEL_NAMES[label]
        metrics = class_report[name]
        logger.info(f"{name:<20} {metrics['precision']:>10.3f} {metrics['recall']:>10.3f} {metrics['f1-score']:>10.3f} {int(metrics['support']):>10}")
    
    # Save confusion matrix as CSV
    cm_df = pd.DataFrame(
        cm,
        index=[Config.LABEL_NAMES[i] for i in Config.VALID_LABELS],
        columns=[Config.LABEL_NAMES[i] for i in Config.VALID_LABELS]
    )
    cm_path = output_path / "confusion_matrix.csv"
    cm_df.to_csv(cm_path)
    logger.info(f"\nConfusion matrix saved to: {cm_path}")
    
    # Save per-class metrics as CSV
    metrics_df = pd.DataFrame(class_report).transpose()
    metrics_path = output_path / "per_class_metrics.csv"
    metrics_df.to_csv(metrics_path)
    logger.info(f"Per-class metrics saved to: {metrics_path}")
    
    return {
        "accuracy": test_accuracy,
        "f1_score": f1,
        "confusion_matrix": cm,
        "class_report": class_report
    }


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """
    Main function that runs the entire training pipeline.
    """
    # Start timing
    start_time = time.time()
    
    # Setup
    logger = setup_logging(Config.OUTPUT_PATH)
    logger.info("="*60)
    logger.info("GLEASON SCORE CLASSIFICATION PIPELINE")
    logger.info("="*60)
    
    # Check GPU
    device = setup_gpu(logger)
    
    # Set random seeds for reproducibility
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    
    # Load and split data
    train_df, valid_df, test_df = load_and_split_data(Config.DATA_PATH, logger)
    
    # Calculate class weights for imbalanced data
    class_weights = calculate_class_weights(logger)
    
    # Run hyperparameter optimization
    best_params = run_hyperparameter_search(
        train_df, valid_df, class_weights, logger
    )
    
    # Train final model with best hyperparameters
    logger.info("\n" + "="*60)
    logger.info("TRAINING FINAL MODEL WITH BEST HYPERPARAMETERS")
    logger.info("="*60)
    
    # Create data loaders with best batch size
    dls = create_dataloaders(
        train_df, valid_df,
        batch_size=best_params["batch_size"]
    )
    
    # Train final model
    final_learner, final_accuracy = train_model(
        dls=dls,
        class_weights=class_weights,
        learning_rate=best_params["learning_rate"],
        epochs=best_params["epochs"],
        logger=logger
    )
    
    logger.info(f"Final model validation accuracy: {final_accuracy:.4f}")
    
    # Evaluate on test set
    eval_results = evaluate_model(final_learner, test_df, logger, Config.OUTPUT_PATH)
    
    # Save the best model
    model_path = Config.OUTPUT_PATH / "best_model.pkl"
    final_learner.export(model_path)
    logger.info(f"\nBest model saved to: {model_path}")
    
    # Calculate total time
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # Final summary
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE - SUMMARY")
    logger.info("="*60)
    logger.info(f"Total execution time: {int(hours)}h {int(minutes)}m {seconds:.1f}s")
    logger.info(f"Best validation accuracy: {final_accuracy:.4f}")
    logger.info(f"Test accuracy: {eval_results['accuracy']:.4f}")
    logger.info(f"Test F1 score: {eval_results['f1_score']:.4f}")
    logger.info(f"Model saved to: {model_path}")
    logger.info(f"Logs saved to: {Config.OUTPUT_PATH}")
    logger.info("="*60)
    
    return final_learner, eval_results


# =============================================================================
# RUN THE SCRIPT
# =============================================================================

if __name__ == "__main__":
    # This block runs when you execute the script directly
    learner, results = main()
