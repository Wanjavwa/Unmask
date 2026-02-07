"""
Training script for the fairness (real vs fake) classifier.
Loads dataset, splits 80/20, trains with validation, saves best checkpoint and training log.
"""

import os
import sys
import argparse
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add src to path so we can run from fairness_model/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.config import (
    REAL_DATA_DIR,
    FAKE_DATA_DIR,
    MODEL_SAVE_DIR,
    MODEL_PATH,
    MODEL_BEST_PATH,
    TRAINING_LOG_PATH,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    TRAIN_VAL_SPLIT,
    RANDOM_SEED,
    LR_SCHEDULER,
    STEPLR_STEP_SIZE,
    STEPLR_GAMMA,
    PLATEAU_PATIENCE,
    PLATEAU_FACTOR,
)
from src.dataset_loader import build_dataset_splits
from src.model_builder import build_fairness_model
from src.metrics import compute_metrics, format_metrics
from src.utils import get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Train fairness model (real vs fake)")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--train-ratio", type=float, default=TRAIN_VAL_SPLIT, help="Train/val split ratio (e.g. 0.8)")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed")
    return parser.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Run one training epoch; return average loss."""
    model.train()
    total_loss = 0.0
    n = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / n if n else 0.0


@torch.no_grad()
def evaluate(model, loader, device):
    """Run validation; return loss and list of (y_true, y_pred)."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    n = 0
    all_preds = []
    all_labels = []
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        loss = criterion(logits, targets)
        total_loss += loss.item()
        n += 1
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(targets.cpu().numpy().tolist())
    avg_loss = total_loss / n if n else 0.0
    return avg_loss, all_labels, all_preds


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = get_device()
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    # Load dataset (prints real/fake/total counts)
    print("Loading dataset...")
    train_dataset, val_dataset = build_dataset_splits(
        real_dir=REAL_DATA_DIR,
        fake_dir=FAKE_DATA_DIR,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("Error: no samples in train or validation set. Check data paths and UTKFace filename filter (_1_ for race).")
        sys.exit(1)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = build_fairness_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if LR_SCHEDULER == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEPLR_STEP_SIZE, gamma=STEPLR_GAMMA)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=PLATEAU_FACTOR, patience=PLATEAU_PATIENCE
        )

    best_val_loss = float("inf")
    log_lines = []
    log_lines.append(f"Training started at {datetime.now().isoformat()}")
    log_lines.append(f"Epochs={args.epochs} batch_size={args.batch_size} lr={args.lr} train_ratio={args.train_ratio}")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_labels, val_preds = evaluate(model, val_loader, device)
        if LR_SCHEDULER == "plateau":
            scheduler.step(val_loss)
        else:
            scheduler.step()
        metrics = compute_metrics(val_labels, val_preds)
        line = (
            f"Epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_acc={metrics['accuracy']:.4f}  val_f1={metrics['f1']:.4f}"
        )
        print(line)
        log_lines.append(line)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "val_loss": val_loss}, MODEL_BEST_PATH)
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  -> Saved best model (val_loss={val_loss:.4f})")
            log_lines.append(f"  -> Saved best model (val_loss={val_loss:.4f})")

    # Final evaluation and metrics
    model.load_state_dict(torch.load(MODEL_BEST_PATH, map_location=device)["state_dict"])
    _, final_labels, final_preds = evaluate(model, val_loader, device)
    final_metrics = compute_metrics(final_labels, final_preds)
    print("\n--- Final validation metrics ---")
    print(format_metrics(final_metrics))
    log_lines.append("\n--- Final validation metrics ---")
    log_lines.append(format_metrics(final_metrics))
    log_lines.append(f"\nTraining finished at {datetime.now().isoformat()}")

    with open(TRAINING_LOG_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    print(f"\nTraining log saved to {TRAINING_LOG_PATH}")
    print(f"Best model saved to {MODEL_BEST_PATH} and {MODEL_PATH}")


if __name__ == "__main__":
    main()
