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

import json
import multiprocessing as mp
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import mlflow
import optuna
import pandas as pd
import seaborn as sns
import torch
from fastai.callback.tracker import SaveModelCallback
from fastai.vision.all import *
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# NOTE: We don't use optuna.integration.FastAIPruningCallback because it has
# pickling issues with CUDA tensors. We implement a custom callback below.

warnings.filterwarnings("ignore")

# Fix CUDA multiprocessing issue - must be set before CUDA initialization
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

# =============================================================================
# GPU AND CPU CONFIGURATION
# =============================================================================


def setup_hardware():
    """Configure GPU and CPU settings for optimal performance."""

    # GPU Setup
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"🖥️  GPU: {gpu_name} ({gpu_memory:.1f} GB) x {gpu_count}")

        # Enable cuDNN optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

        # NOTE: Don't set default tensor type to CUDA - causes pickling issues with DataLoader workers
        # Instead, explicitly move models/data to GPU as needed (e.g., dls.cuda(), learn.to_fp16())
        # torch.set_default_tensor_type("torch.cuda.FloatTensor")  # DISABLED - breaks multiprocessing

        # Enable TF32 for Ampere+ GPUs (faster matrix ops)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        print("⚠️  No GPU detected - training will be slow!")

    # CPU Setup for data loading
    n_cpu = mp.cpu_count()
    print(f"💻 CPU cores available: {n_cpu}")

    return torch.cuda.is_available()


# Run hardware setup at import
HAS_GPU = setup_hardware()

# =============================================================================
# CONFIGURATION
# =============================================================================


class Config:
    """Central configuration for the training pipeline."""

    # Data paths - UPDATE THIS TO YOUR IMAGE FOLDER
    IMAGE_DIR = Path("./gleason_images")  # Change to your actual path
    OUTPUT_DIR = Path("./output")

    # Label mapping from filename suffix
    # Labels 4 and 5 are skipped during training
    LABEL_MAP = {
        0: "Benign",
        1: "Gleason_3",
        2: "Gleason_4",
        3: "Gleason_5_Single_Cells",
        # 4: "Gleason_5_Secretions",  # SKIPPED
        # 5: "Gleason_5",              # SKIPPED
    }

    # Labels to skip during training
    SKIP_LABELS = {4, 5}

    # You can also use numeric labels if preferred
    USE_NUMERIC_LABELS = False

    # Data splits
    TEST_SIZE = 0.15  # 15% for final test
    VALID_SIZE = 0.15  # 15% of remaining for validation
    RANDOM_SEED = 42

    # Hardware configuration
    NUM_WORKERS = 64  # Number of CPU workers for data loading (you have 68 cores)
    PIN_MEMORY = True  # Faster GPU transfer when True
    PREFETCH_FACTOR = 4  # Batches to prefetch per worker

    # Training defaults (Optuna will override these)
    DEFAULT_BATCH_SIZE = 64  # Larger batch for GPU efficiency
    DEFAULT_LR = 1e-3
    DEFAULT_EPOCHS = 10
    DEFAULT_ARCH = "resnet34"
    IMAGE_SIZE = 224

    # Mixed precision training (faster on modern GPUs)
    USE_FP16 = True

    # Optuna settings
    N_OPTUNA_TRIALS = 20
    OPTUNA_TIMEOUT = 3600 * 2  # 2 hours max

    # MLflow settings
    MLFLOW_EXPERIMENT = "gleason_classifier"
    MLFLOW_TRACKING_URI = "./mlruns"


# =============================================================================
# CUSTOM OPTUNA PRUNING CALLBACK (avoids CUDA pickling issues)
# =============================================================================


class OptunaPruningCallback(Callback):
    """
    FastAI callback for Optuna pruning that avoids CUDA tensor pickling issues.
    
    The official FastAIPruningCallback from optuna-integration can fail with
    'Cannot pickle CUDA storage' errors when used with GPU training and
    multiple DataLoader workers. This implementation avoids that by only
    passing Python floats to Optuna.
    """
    
    def __init__(self, trial: optuna.Trial, monitor: str = "valid_loss"):
        self.trial = trial
        self.monitor = monitor
    
    def after_epoch(self):
        # Get the monitored value and convert to Python float (not CUDA tensor)
        value = self.recorder.values[-1][self.recorder.metric_names.index(self.monitor)]
        
        # Ensure it's a plain Python float, not a tensor
        if hasattr(value, 'item'):
            value = value.item()
        value = float(value)
        
        # Report to Optuna
        self.trial.report(value, step=self.epoch)
        
        # Check if trial should be pruned
        if self.trial.should_prune():
            raise optuna.TrialPruned(f"Trial pruned at epoch {self.epoch}")


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
    except Exception:
        return False


def validate_image_worker(filepath: str) -> Tuple[str, bool, Optional[int]]:
    """Worker function for parallel image validation."""
    path = Path(filepath)
    label = extract_label_from_filename(path.name)
    if label is None:
        return filepath, False, None
    is_valid = is_valid_image(path)
    return filepath, is_valid, label


def extract_label_from_filename(filename: str) -> Optional[int]:
    """Extract the Gleason label from filename suffix like 'image_0.png'."""
    stem = Path(filename).stem
    for suffix in range(6):  # _0 through _5
        if stem.endswith(f"_{suffix}"):
            return suffix
    return None


def prepare_dataset(
    image_dir: Path, config: Config
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Scan image directory, validate images, extract labels, and split into train/val/test.
    Uses parallel processing for fast validation on many CPU cores.

    Returns:
        Tuple of (train_df, valid_df, test_df)
    """
    print("\n" + "=" * 60)
    print("📁 PREPARING DATASET")
    print("=" * 60)

    # Find all PNG files
    all_images = list(image_dir.glob("*.png"))
    print(f"\nFound {len(all_images)} PNG files in {image_dir}")

    if len(all_images) == 0:
        print(f"\n❌ No PNG files found in {image_dir}")
        print("Please update Config.IMAGE_DIR to point to your image folder.")
        sys.exit(1)

    # Parallel validation using all available cores
    print(f"\nValidating images using {config.NUM_WORKERS} CPU cores...")
    print(f"Skipping labels: {config.SKIP_LABELS}")
    valid_data = []
    corrupt_count = 0
    skipped_label_count = 0
    label_counts = {i: 0 for i in range(6)}

    # Use ProcessPoolExecutor for parallel image validation
    image_paths = [str(p) for p in all_images]

    with ProcessPoolExecutor(max_workers=config.NUM_WORKERS) as executor:
        futures = {executor.submit(validate_image_worker, p): p for p in image_paths}

        completed = 0
        for future in as_completed(futures):
            completed += 1
            if completed % 5000 == 0:
                print(f"  Validated {completed}/{len(all_images)} images...")

            filepath, is_valid, label = future.result()

            if label is None:
                continue

            # Track all labels for reporting
            label_counts[label] += 1

            # Skip labels 4 and 5
            if label in config.SKIP_LABELS:
                skipped_label_count += 1
                continue

            if is_valid:
                label_name = (
                    config.LABEL_MAP[label]
                    if not config.USE_NUMERIC_LABELS
                    else str(label)
                )
                valid_data.append(
                    {"filepath": filepath, "label": label_name, "label_num": label}
                )
            else:
                corrupt_count += 1

    print(f"\n✓ Valid images for training: {len(valid_data)}")
    print(f"✗ Corrupt images: {corrupt_count}")
    print(f"⊘ Skipped (labels 4,5): {skipped_label_count}")

    print("\n📊 Full dataset class distribution:")
    all_label_names = {
        0: "Benign",
        1: "Gleason_3",
        2: "Gleason_4",
        3: "Gleason_5_Single_Cells",
        4: "Gleason_5_Secretions",
        5: "Gleason_5",
    }
    total_images = sum(label_counts.values())
    for label_num, count in label_counts.items():
        label_name = all_label_names[label_num]
        pct = 100 * count / total_images if total_images else 0
        skip_marker = " [SKIPPED]" if label_num in config.SKIP_LABELS else ""
        print(
            f"   {label_num} ({label_name:24s}): {count:6d} ({pct:5.1f}%){skip_marker}"
        )

    print("\n📊 Training class distribution (after filtering):")
    training_counts = {
        k: v for k, v in label_counts.items() if k not in config.SKIP_LABELS
    }
    for label_num, count in training_counts.items():
        label_name = config.LABEL_MAP[label_num]
        pct = 100 * count / len(valid_data) if valid_data else 0
        print(f"   {label_num} ({label_name:24s}): {count:6d} ({pct:5.1f}%)")

    # Create DataFrame
    df = pd.DataFrame(valid_data)

    # Stratified split: first separate test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=config.TEST_SIZE,
        stratify=df["label"],
        random_state=config.RANDOM_SEED,
    )

    # Then split train_val into train and validation
    train_df, valid_df = train_test_split(
        train_val_df,
        test_size=config.VALID_SIZE / (1 - config.TEST_SIZE),
        stratify=train_val_df["label"],
        random_state=config.RANDOM_SEED,
    )

    print("\n📂 Data splits:")
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
        "efficientnet_b0": (
            efficientnet_b0
            if hasattr(sys.modules[__name__], "efficientnet_b0")
            else resnet34
        ),
    }
    return architectures.get(arch_name, resnet34)


def create_dataloaders(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    config: Config,
    batch_size: int = 64,
    image_size: int = 224,
) -> DataLoaders:
    """Create FastAI DataLoaders from DataFrames with optimized GPU/CPU settings."""

    # Combine for DataBlock
    combined_df = pd.concat(
        [train_df.assign(is_valid=False), valid_df.assign(is_valid=True)]
    ).reset_index(drop=True)

    # Define DataBlock
    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_x=ColReader("filepath"),
        get_y=ColReader("label"),
        splitter=ColSplitter("is_valid"),
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
            Normalize.from_stats(*imagenet_stats),
        ],
    )

    # Create dataloaders with optimized settings for GPU training
    dls = dblock.dataloaders(
        combined_df,
        bs=batch_size,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        prefetch_factor=config.PREFETCH_FACTOR,
        persistent_workers=True,  # Keep workers alive between epochs
    )

    # NOTE: Don't call dls.cuda() here - it creates CUDA state that breaks pickling with workers
    # FastAI's learner will automatically move batches to GPU during training
    # The pin_memory=True setting handles efficient CPU->GPU transfer

    return dls


def train_model(
    dls: DataLoaders,
    config: Config,
    arch_name: str = "resnet34",
    lr: float = 1e-3,
    epochs: int = 10,
    freeze_epochs: int = 1,
    wd: float = 0.01,
    mixup: float = 0.0,
    save_path: Optional[Path] = None,
    trial: Optional[optuna.Trial] = None,
) -> Tuple[Learner, dict]:
    """
    Train a FastAI model with the given hyperparameters.
    Uses mixed precision (FP16) for faster GPU training.

    Returns:
        Tuple of (learner, metrics_dict)
    """
    arch = get_architecture(arch_name)

    # Build learner with metrics
    cbs = []
    if trial is not None:
        cbs.append(OptunaPruningCallback(trial, monitor="valid_loss"))

    if save_path:
        cbs.append(SaveModelCallback(monitor="valid_loss", fname="best_model"))

    learn = cnn_learner(
        dls,
        arch,
        metrics=[
            accuracy,
            F1Score(average="macro"),
            Precision(average="macro"),
            Recall(average="macro"),
        ],
        wd=wd,
        cbs=cbs,
    )

    # Enable mixed precision training for faster GPU performance
    if config.USE_FP16 and HAS_GPU:
        learn = learn.to_fp16()

    # Optional mixup
    if mixup > 0:
        learn = learn.add_cb(MixUp(mixup))

    # Train: freeze first, then unfreeze
    learn.freeze()
    learn.fit_one_cycle(freeze_epochs, lr)

    learn.unfreeze()
    learn.fit_one_cycle(epochs - freeze_epochs, slice(lr / 100, lr / 10))

    # Get final metrics
    val_loss, acc, f1, prec, rec = learn.validate()

    metrics = {
        "valid_loss": float(val_loss),
        "accuracy": float(acc),
        "f1_macro": float(f1),
        "precision_macro": float(prec),
        "recall_macro": float(rec),
    }

    return learn, metrics


# =============================================================================
# EVALUATION
# =============================================================================


def evaluate_model(
    learn: Learner, test_df: pd.DataFrame, config: Config, output_dir: Path
) -> dict:
    """
    Evaluate model on test set and generate reports.

    Returns:
        Dictionary of test metrics
    """
    print("\n" + "=" * 60)
    print("📊 EVALUATING ON TEST SET")
    print("=" * 60)

    # Create test dataloader
    test_dl = learn.dls.test_dl(test_df["filepath"].tolist())

    # Get predictions
    preds, _ = learn.get_preds(dl=test_dl)
    pred_labels = preds.argmax(dim=1).numpy()

    # Map predictions back to class names
    vocab = learn.dls.vocab
    pred_names = [vocab[i] for i in pred_labels]
    true_names = test_df["label"].tolist()

    # Classification report
    print("\n📋 Classification Report:")
    print("-" * 60)
    report = classification_report(
        true_names, pred_names, output_dict=True, zero_division=0
    )
    print(classification_report(true_names, pred_names, zero_division=0))

    # Save report
    report_path = output_dir / "classification_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n✓ Report saved to {report_path}")

    # Confusion Matrix
    cm = confusion_matrix(true_names, pred_names, labels=vocab)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=vocab, yticklabels=vocab
    )
    plt.title("Confusion Matrix - Test Set", fontsize=14)
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("True", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    cm_path = output_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"✓ Confusion matrix saved to {cm_path}")

    # Per-class metrics summary
    print("\n📈 Per-Class Metrics:")
    print("-" * 60)
    print(
        f"{'Class':<25} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}"
    )
    print("-" * 60)

    for class_name in vocab:
        if class_name in report:
            m = report[class_name]
            print(
                f"{class_name:<25} {m['precision']:>10.3f} {m['recall']:>10.3f} {m['f1-score']:>10.3f} {m['support']:>10.0f}"
            )

    print("-" * 60)
    print(
        f"{'Macro Avg':<25} {report['macro avg']['precision']:>10.3f} {report['macro avg']['recall']:>10.3f} {report['macro avg']['f1-score']:>10.3f}"
    )
    print(
        f"{'Weighted Avg':<25} {report['weighted avg']['precision']:>10.3f} {report['weighted avg']['recall']:>10.3f} {report['weighted avg']['f1-score']:>10.3f}"
    )

    return {
        "test_accuracy": report["accuracy"],
        "test_f1_macro": report["macro avg"]["f1-score"],
        "test_precision_macro": report["macro avg"]["precision"],
        "test_recall_macro": report["macro avg"]["recall"],
    }


# =============================================================================
# TORCHSCRIPT EXPORT
# =============================================================================


def export_to_torchscript(
    learn: Learner, output_dir: Path, image_size: int = 224
) -> Optional[Path]:
    """
    Export FastAI model to TorchScript format for production deployment.

    TorchScript advantages:
    - No Python dependency at inference time
    - Can run in C++/Java/other runtimes
    - Optimized for production serving
    - Smaller file size (no training state)

    Returns:
        Path to exported .pt file, or None if export failed
    """
    print("\n" + "-" * 60)
    print("📦 EXPORTING TO TORCHSCRIPT")
    print("-" * 60)

    try:
        # Get the PyTorch model from FastAI learner
        model = learn.model.eval()

        # Move to CPU for broader compatibility (can also export CUDA version)
        model_cpu = model.cpu()

        # Create example input tensor (batch_size=1, channels=3, height, width)
        example_input = torch.randn(1, 3, image_size, image_size)

        # Trace the model (records operations during forward pass)
        with torch.no_grad():
            traced_model = torch.jit.trace(model_cpu, example_input)

        # Optimize for inference
        traced_model = torch.jit.optimize_for_inference(traced_model)

        # Save TorchScript model
        torchscript_path = output_dir / "gleason_classifier_final.pt"
        traced_model.save(str(torchscript_path))

        # Also save class labels for inference
        labels_path = output_dir / "class_labels.json"
        class_labels = {i: label for i, label in enumerate(learn.dls.vocab)}
        with open(labels_path, "w") as f:
            json.dump(class_labels, f, indent=2)

        # Report file sizes
        pkl_size = (output_dir / "gleason_classifier_final.pkl").stat().st_size / 1e6
        pt_size = torchscript_path.stat().st_size / 1e6

        print(f"✓ TorchScript model saved to {torchscript_path}")
        print(f"✓ Class labels saved to {labels_path}")
        print(f"  FastAI .pkl size: {pkl_size:.1f} MB")
        print(f"  TorchScript .pt size: {pt_size:.1f} MB")

        # Move model back to GPU if available
        if HAS_GPU:
            learn.model.cuda()

        return torchscript_path

    except Exception as e:
        print(f"⚠ TorchScript export failed: {e}")
        print("  The .pkl model is still available for inference via FastAI.")
        return None


def load_torchscript_model(model_path: str, labels_path: str):
    """
    Example function showing how to load and use the TorchScript model.

    Usage:
        model, labels, transform = load_torchscript_model(
            'output/gleason_classifier_final.pt',
            'output/class_labels.json'
        )

        # Load and preprocess image
        from PIL import Image
        img = Image.open('test_image.png').convert('RGB')
        input_tensor = transform(img).unsqueeze(0)

        # Predict
        with torch.no_grad():
            output = model(input_tensor)
            pred_idx = output.argmax(dim=1).item()
            pred_label = labels[str(pred_idx)]
            confidence = torch.softmax(output, dim=1)[0, pred_idx].item()

        print(f"Predicted: {pred_label} ({confidence:.1%})")
    """
    from torchvision import transforms

    # Load model
    model = torch.jit.load(model_path)
    model.eval()

    # Load labels
    with open(labels_path) as f:
        labels = json.load(f)

    # ImageNet normalization (same as training)
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return model, labels, transform


# =============================================================================
# OPTUNA HYPERPARAMETER OPTIMIZATION
# =============================================================================


def optuna_objective(
    trial: optuna.Trial, train_df: pd.DataFrame, valid_df: pd.DataFrame, config: Config
) -> float:
    """Optuna objective function for hyperparameter optimization."""

    # Suggest hyperparameters - larger batches for GPU
    arch_name = trial.suggest_categorical(
        "architecture", ["resnet18", "resnet34", "resnet50"]
    )
    batch_size = trial.suggest_categorical(
        "batch_size", [32, 64, 128, 256]
    )  # Larger batches for GPU
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    epochs = trial.suggest_int("epochs", 5, 15)
    freeze_epochs = trial.suggest_int("freeze_epochs", 1, 3)
    wd = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
    mixup = trial.suggest_float("mixup", 0.0, 0.4)
    image_size = trial.suggest_categorical("image_size", [224, 256, 320])

    # Log trial parameters
    print(
        f"\n🔬 Trial {trial.number}: arch={arch_name}, bs={batch_size}, lr={lr:.2e}, epochs={epochs}"
    )

    try:
        # Create dataloaders with trial params
        dls = create_dataloaders(train_df, valid_df, config, batch_size, image_size)

        # Train model
        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
            mlflow.log_params(trial.params)

            learn, metrics = train_model(
                dls=dls,
                config=config,
                arch_name=arch_name,
                lr=lr,
                epochs=epochs,
                freeze_epochs=freeze_epochs,
                wd=wd,
                mixup=mixup,
                trial=trial,
            )

            mlflow.log_metrics(metrics)

            # Clean up GPU memory
            del learn
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            return metrics["f1_macro"]

    except optuna.TrialPruned:
        # Re-raise pruning exception so Optuna handles it properly
        raise
    except Exception as e:
        print(f"  ❌ Trial failed: {e}")
        return 0.0


def run_hyperparameter_search(
    train_df: pd.DataFrame, valid_df: pd.DataFrame, config: Config
) -> dict:
    """Run Optuna hyperparameter optimization."""

    print("\n" + "=" * 60)
    print("🔍 HYPERPARAMETER OPTIMIZATION (OPTUNA)")
    print("=" * 60)
    print(
        f"\nRunning {config.N_OPTUNA_TRIALS} trials (timeout: {config.OPTUNA_TIMEOUT}s)"
    )

    # Create Optuna study
    study = optuna.create_study(
        direction="maximize",
        study_name="gleason_classifier",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=2),
    )

    # Optimize
    study.optimize(
        lambda trial: optuna_objective(trial, train_df, valid_df, config),
        n_trials=config.N_OPTUNA_TRIALS,
        timeout=config.OPTUNA_TIMEOUT,
        show_progress_bar=True,
    )

    # Report results
    print("\n" + "=" * 60)
    print("🏆 BEST HYPERPARAMETERS")
    print("=" * 60)
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

    print("\n" + "=" * 60)
    print("🔬 GLEASON SCORE CLASSIFIER")
    print("    FastAI + Optuna + MLflow")
    print("=" * 60)

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
    with mlflow.start_run(
        run_name=f"gleason_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ):

        # Log config including hardware
        mlflow.log_params(
            {
                "test_size": config.TEST_SIZE,
                "valid_size": config.VALID_SIZE,
                "n_train": len(train_df),
                "n_valid": len(valid_df),
                "n_test": len(test_df),
                "n_optuna_trials": config.N_OPTUNA_TRIALS,
                "num_workers": config.NUM_WORKERS,
                "use_fp16": config.USE_FP16,
                "gpu_available": HAS_GPU,
                "gpu_name": torch.cuda.get_device_name(0) if HAS_GPU else "None",
            }
        )

        # Run hyperparameter search
        best_params = run_hyperparameter_search(train_df, valid_df, config)
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})

        # Train final model with best parameters
        print("\n" + "=" * 60)
        print("🚀 TRAINING FINAL MODEL WITH BEST PARAMETERS")
        print("=" * 60)

        dls = create_dataloaders(
            train_df,
            valid_df,
            config,
            batch_size=best_params.get("batch_size", 32),
            image_size=best_params.get("image_size", 224),
        )

        learn, metrics = train_model(
            dls=dls,
            config=config,
            arch_name=best_params.get("architecture", "resnet34"),
            lr=best_params.get("learning_rate", 1e-3),
            epochs=best_params.get("epochs", 10),
            freeze_epochs=best_params.get("freeze_epochs", 1),
            wd=best_params.get("weight_decay", 0.01),
            mixup=best_params.get("mixup", 0.0),
            save_path=config.OUTPUT_DIR,
        )

        mlflow.log_metrics({f"final_{k}": v for k, v in metrics.items()})

        # Evaluate on test set
        test_metrics = evaluate_model(learn, test_df, config, config.OUTPUT_DIR)
        mlflow.log_metrics(test_metrics)

        # Save final model as FastAI .pkl
        model_path = config.OUTPUT_DIR / "gleason_classifier_final.pkl"
        learn.export(model_path)
        print(f"\n✓ FastAI model saved to {model_path}")
        mlflow.log_artifact(str(model_path))

        # Export to TorchScript for production deployment
        torchscript_path = export_to_torchscript(
            learn, config.OUTPUT_DIR, config.IMAGE_SIZE
        )
        if torchscript_path:
            mlflow.log_artifact(str(torchscript_path))

        # Log confusion matrix
        mlflow.log_artifact(str(config.OUTPUT_DIR / "confusion_matrix.png"))
        mlflow.log_artifact(str(config.OUTPUT_DIR / "classification_report.json"))

        # Summary
        print("\n" + "=" * 60)
        print("✅ TRAINING COMPLETE")
        print("=" * 60)
        print("\n📊 Final Test Metrics:")
        print(f"   Accuracy:  {test_metrics['test_accuracy']:.4f}")
        print(f"   F1 (macro): {test_metrics['test_f1_macro']:.4f}")
        print(f"   Precision: {test_metrics['test_precision_macro']:.4f}")
        print(f"   Recall:    {test_metrics['test_recall_macro']:.4f}")
        print(f"\n📁 Outputs saved to: {config.OUTPUT_DIR}")
        print(
            f"🔗 MLflow UI: run 'mlflow ui --backend-store-uri {config.MLFLOW_TRACKING_URI}'"
        )


if __name__ == "__main__":
    main()
