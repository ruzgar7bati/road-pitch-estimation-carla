import os
import csv
import math
import time
import random
import argparse
from dataclasses import dataclass

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    import torchvision
    from torchvision import transforms
except ImportError as exc:
    raise ImportError(
        "torchvision is required for this script. Install it with:\n"
        "pip install torchvision"
    ) from exc

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


# =========================
# Default settings
# =========================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_DATASET_DIR = os.path.join(SCRIPT_DIR, "dataset_pitch")
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "training_output")
DEFAULT_MODEL_DIR = os.path.join(SCRIPT_DIR, "models")

DEFAULT_TRAIN_CSV = "train.csv"
DEFAULT_VAL_CSV = "val.csv"

# Use true vehicle pitch because that is what we decided to train on.
# Keep waypoint_pitch_deg available for quick switching.
DEFAULT_TARGET_COLUMN = "true_vehicle_pitch_deg"
# DEFAULT_TARGET_COLUMN = "waypoint_pitch_deg"

IMAGE_SIZE = 224

BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4

RANDOM_SEED = 42

MODEL_NAME = "mobilenet_v3_small"
BEST_MODEL_NAME = "mobilenet_pitch_best.pth"
LAST_MODEL_NAME = "mobilenet_pitch_last.pth"

USE_AMP = True  # automatic mixed precision if CUDA is available


# =========================
# Reproducibility
# =========================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Good enough reproducibility without slowing training too much.
    torch.backends.cudnn.benchmark = True


# =========================
# Dataset
# =========================

class PitchMaskDataset(Dataset):
    def __init__(self, dataset_dir, csv_name, target_column, image_size=224):
        self.dataset_dir = dataset_dir
        self.csv_path = os.path.join(dataset_dir, csv_name)
        self.target_column = target_column

        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        self.rows = []

        with open(self.csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)

            if target_column not in reader.fieldnames:
                raise ValueError(
                    f"Target column '{target_column}' not found in {self.csv_path}.\n"
                    f"Available columns: {reader.fieldnames}"
                )

            if "mask_path" not in reader.fieldnames:
                raise ValueError(
                    f"'mask_path' column not found in {self.csv_path}.\n"
                    f"Available columns: {reader.fieldnames}"
                )

            for row in reader:
                target_value = row.get(target_column, "")

                if target_value is None or target_value == "":
                    continue

                try:
                    target_float = float(target_value)
                except ValueError:
                    continue

                mask_rel_path = row["mask_path"]
                mask_abs_path = os.path.join(dataset_dir, mask_rel_path)

                if not os.path.exists(mask_abs_path):
                    # Skip missing image rows instead of crashing during training.
                    continue

                row["_target_float"] = target_float
                row["_mask_abs_path"] = mask_abs_path
                self.rows.append(row)

        if len(self.rows) == 0:
            raise RuntimeError(
                f"No usable rows found in {self.csv_path}. "
                f"Check image paths and target column."
            )

        # MobileNet expects 3-channel images.
        # Our saved masks are already 3-channel:
        #   B = road
        #   G = road line
        #   R = road + road line
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            # Do NOT use ImageNet normalization here.
            # The input is semantic masks, not natural RGB.
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]

        image = Image.open(row["_mask_abs_path"]).convert("RGB")
        image_tensor = self.transform(image)

        target = torch.tensor([row["_target_float"]], dtype=torch.float32)

        metadata = {
            "sample_index": row.get("sample_index", ""),
            "split": row.get("split", ""),
            "mask_path": row.get("mask_path", ""),
            "semantic_frame": row.get("semantic_frame", ""),
            "sim_time": row.get("sim_time", ""),
            "true_vehicle_pitch_deg": row.get("true_vehicle_pitch_deg", ""),
            "waypoint_pitch_deg": row.get("waypoint_pitch_deg", ""),
        }

        return image_tensor, target, metadata


# =========================
# Model
# =========================

def build_mobilenet_regressor():
    """
    MobileNetV3-small regression model.

    weights=None avoids internet downloads and makes the script self-contained.
    The final classifier output is replaced with one value: pitch in degrees.
    """
    model = torchvision.models.mobilenet_v3_small(weights=None)

    # torchvision MobileNetV3 classifier usually:
    # Sequential(
    #   Linear(...),
    #   Hardswish,
    #   Dropout,
    #   Linear(..., 1000)
    # )
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 1)

    return model


# =========================
# Metrics
# =========================

def compute_regression_metrics(predictions, targets):
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)

    errors = predictions - targets
    abs_errors = np.abs(errors)

    mse = np.mean(errors ** 2)
    rmse = math.sqrt(mse)
    mae = np.mean(abs_errors)
    max_abs_error = np.max(abs_errors)
    mean_error = np.mean(errors)
    std_error = np.std(errors)

    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "max_abs_error": float(max_abs_error),
        "mean_error": float(mean_error),
        "std_error": float(std_error),
    }


# =========================
# Train / validate loops
# =========================

def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler=None):
    model.train()

    running_loss = 0.0
    count = 0

    for images, targets, _ in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(images)
                loss = loss_fn(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        count += batch_size

    return running_loss / max(count, 1)


@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()

    running_loss = 0.0
    count = 0

    predictions = []
    targets_all = []
    metadata_all = []

   

        outputs = model(images)
        loss = loss_fn(outputs, targets)

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        count += batch_size

        outputs_np = outputs.detach().cpu().numpy().reshape(-1)
        targets_np = targets.detach().cpu().numpy().reshape(-1)

        predictions.extend(outputs_np.tolist())
        targets_all.extend(targets_np.tolist())

        # Default DataLoader collate turns metadata dict values into lists.
        for i in range(batch_size):
            item = {}
            for key, value in metadata.items():
                if isinstance(value, (list, tuple)):
                    item[key] = value[i]
                else:
                    # Some versions may collate strings differently.
                    try:
                        item[key] = value[i]
                    except Exception:
                        item[key] = value
            metadata_all.append(item)

    val_loss = running_loss / max(count, 1)
    metrics = compute_regression_metrics(predictions, targets_all)

    return val_loss, metrics, predictions, targets_all, metadata_all


# =========================
# Output helpers
# =========================

def write_history_csv(history, output_path):
    if not history:
        return

    fieldnames = list(history[0].keys())

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def write_predictions_csv(output_path, predictions, targets, metadata_all, target_column):
    fieldnames = [
        "sample_index",
        "split",
        "mask_path",
        "semantic_frame",
        "sim_time",
        "target_column",
        "target_pitch_deg",
        "predicted_pitch_deg",
        "error_deg",
        "abs_error_deg",
        "true_vehicle_pitch_deg",
        "waypoint_pitch_deg",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for pred, target, metadata in zip(predictions, targets, metadata_all):
            error = pred - target

            writer.writerow({
                "sample_index": metadata.get("sample_index", ""),
                "split": metadata.get("split", ""),
                "mask_path": metadata.get("mask_path", ""),
                "semantic_frame": metadata.get("semantic_frame", ""),
                "sim_time": metadata.get("sim_time", ""),
                "target_column": target_column,
                "target_pitch_deg": target,
                "predicted_pitch_deg": pred,
                "error_deg": error,
                "abs_error_deg": abs(error),
                "true_vehicle_pitch_deg": metadata.get("true_vehicle_pitch_deg", ""),
                "waypoint_pitch_deg": metadata.get("waypoint_pitch_deg", ""),
            })


def save_loss_curve(history, output_path):
    if plt is None or not history:
        return False

    epochs = [row["epoch"] for row in history]
    train_losses = [row["train_loss"] for row in history]
    val_losses = [row["val_loss"] for row in history]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label="Train loss")
    plt.plot(epochs, val_losses, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("MobileNet pitch regression loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return True


def save_prediction_scatter(predictions, targets, output_path):
    if plt is None:
        return False

    predictions = np.asarray(predictions)
    targets = np.asarray(targets)

    min_value = min(float(np.min(predictions)), float(np.min(targets)))
    max_value = max(float(np.max(predictions)), float(np.max(targets)))

    plt.figure(figsize=(8, 8))
    plt.scatter(targets, predictions, s=8, alpha=0.5)
    plt.plot([min_value, max_value], [min_value, max_value], linestyle="--")
    plt.xlabel("True pitch (deg)")
    plt.ylabel("Predicted pitch (deg)")
    plt.title("Validation prediction vs true pitch")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return True


def save_error_histogram(predictions, targets, output_path):
    if plt is None:
        return False

    predictions = np.asarray(predictions)
    targets = np.asarray(targets)
    errors = predictions - targets

    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50)
    plt.xlabel("Prediction error (deg)")
    plt.ylabel("Count")
    plt.title("Validation pitch error histogram")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return True


def save_checkpoint(
    output_path,
    model,
    optimizer,
    epoch,
    target_column,
    image_size,
    train_loss,
    val_loss,
    val_metrics,
    args,
):
    checkpoint = {
        "model_name": MODEL_NAME,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "target_column": target_column,
        "image_size": image_size,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_metrics": val_metrics,
        "args": vars(args),
        "note": (
            "MobileNetV3-small regression model. "
            "Input is 3-channel CARLA semantic road + road-line mask. "
            "Output is pitch angle in degrees."
        ),
    }

    torch.save(checkpoint, output_path)


# =========================
# Main
# =========================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train MobileNet pitch regressor from CARLA road + road-line masks."
    )

    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)

    parser.add_argument("--train-csv", default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--val-csv", default=DEFAULT_VAL_CSV)

    parser.add_argument("--target-column", default=DEFAULT_TARGET_COLUMN)

    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)

    parser.add_argument(
        "--loss",
        choices=["smooth_l1", "mse"],
        default="smooth_l1",
        help="Regression loss. smooth_l1 is more robust to occasional noisy labels.",
    )

    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable CUDA automatic mixed precision.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\nPitch model training")
    print("====================")
    print(f"Dataset dir:   {os.path.abspath(args.dataset_dir)}")
    print(f"Train CSV:     {args.train_csv}")
    print(f"Val CSV:       {args.val_csv}")
    print(f"Target column: {args.target_column}")
    print(f"Output dir:    {os.path.abspath(args.output_dir)}")
    print(f"Model dir:     {os.path.abspath(args.model_dir)}")
    print(f"Device:        {device}")

    if torch.cuda.is_available():
        print(f"CUDA device:   {torch.cuda.get_device_name(0)}")
        print(f"CUDA version:  {torch.version.cuda}")

    train_dataset = PitchMaskDataset(
        dataset_dir=args.dataset_dir,
        csv_name=args.train_csv,
        target_column=args.target_column,
        image_size=args.image_size,
    )

    val_dataset = PitchMaskDataset(
        dataset_dir=args.dataset_dir,
        csv_name=args.val_csv,
        target_column=args.target_column,
        image_size=args.image_size,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")

    # pin_memory helps CPU -> GPU transfer.
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    model = build_mobilenet_regressor()
    model.to(device)

    if args.loss == "smooth_l1":
        loss_fn = nn.SmoothL1Loss(beta=0.5)
    else:
        loss_fn = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=4,
    )

    scaler = None
    if USE_AMP and torch.cuda.is_available() and not args.no_amp:
        scaler = torch.amp.GradScaler("cuda")
        print("AMP:           enabled")
    else:
        print("AMP:           disabled")

    best_val_mae = float("inf")
    best_epoch = -1
    history = []

    best_model_path = os.path.join(args.model_dir, BEST_MODEL_NAME)
    last_model_path = os.path.join(args.model_dir, LAST_MODEL_NAME)

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            scaler=scaler,
        )

        val_loss, val_metrics, predictions, targets, metadata_all = validate(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=device,
        )

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mae_deg": val_metrics["mae"],
            "val_rmse_deg": val_metrics["rmse"],
            "val_max_abs_error_deg": val_metrics["max_abs_error"],
            "val_mean_error_deg": val_metrics["mean_error"],
            "val_std_error_deg": val_metrics["std_error"],
            "lr": current_lr,
        }

        history.append(row)

        is_best = val_metrics["mae"] < best_val_mae

        if is_best:
            best_val_mae = val_metrics["mae"]
            best_epoch = epoch

            save_checkpoint(
                output_path=best_model_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                target_column=args.target_column,
                image_size=args.image_size,
                train_loss=train_loss,
                val_loss=val_loss,
                val_metrics=val_metrics,
                args=args,
            )

            write_predictions_csv(
                output_path=os.path.join(args.output_dir, "val_predictions_best.csv"),
                predictions=predictions,
                targets=targets,
                metadata_all=metadata_all,
                target_column=args.target_column,
            )

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f} | "
            f"MAE={val_metrics['mae']:.4f} deg | "
            f"RMSE={val_metrics['rmse']:.4f} deg | "
            f"max_abs={val_metrics['max_abs_error']:.4f} deg | "
            f"lr={current_lr:.2e}"
            f"{' | best' if is_best else ''}"
        )

        # Save history every epoch so you still have it if training is interrupted.
        write_history_csv(
            history,
            os.path.join(args.output_dir, "train_history.csv")
        )

    # Save final checkpoint.
    save_checkpoint(
        output_path=last_model_path,
        model=model,
        optimizer=optimizer,
        epoch=args.epochs,
        target_column=args.target_column,
        image_size=args.image_size,
        train_loss=train_loss,
        val_loss=val_loss,
        val_metrics=val_metrics,
        args=args,
    )

    # Re-run validation with the final model and write final predictions.
    val_loss, val_metrics, predictions, targets, metadata_all = validate(
        model=model,
        loader=val_loader,
        loss_fn=loss_fn,
        device=device,
    )

    write_predictions_csv(
        output_path=os.path.join(args.output_dir, "val_predictions_last.csv"),
        predictions=predictions,
        targets=targets,
        metadata_all=metadata_all,
        target_column=args.target_column,
    )

    save_loss_curve(
        history,
        os.path.join(args.output_dir, "loss_curve.png")
    )

    save_prediction_scatter(
        predictions,
        targets,
        os.path.join(args.output_dir, "val_prediction_vs_true_last.png")
    )

    save_error_histogram(
        predictions,
        targets,
        os.path.join(args.output_dir, "val_error_histogram_last.png")
    )

    elapsed = time.time() - start_time

    print("\nTraining complete")
    print("=================")
    print(f"Best epoch:       {best_epoch}")
    print(f"Best val MAE:     {best_val_mae:.4f} deg")
    print(f"Best model saved: {os.path.abspath(best_model_path)}")
    print(f"Last model saved: {os.path.abspath(last_model_path)}")
    print(f"History CSV:      {os.path.abspath(os.path.join(args.output_dir, 'train_history.csv'))}")
    print(f"Elapsed time:     {elapsed / 60.0:.2f} min")


if __name__ == "__main__":
    main()
