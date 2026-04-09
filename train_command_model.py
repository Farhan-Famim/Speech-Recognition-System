from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from command_model import (
    CommandCNN,
    CommandDataset,
    build_label_mapping,
    evaluate_model,
    load_manifest,
    save_label_mapping,
    split_rows,
)


def build_parser():
    parser = argparse.ArgumentParser(description="Train a custom speech-command classifier from your own dataset.")
    parser.add_argument("--manifest", default="custom_dataset/metadata.csv", help="Path to your metadata CSV.")
    parser.add_argument("--output-dir", default="artifacts/command_model", help="Where to save checkpoints.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=8, help="Mini-batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Optimizer learning rate.")
    return parser


def main():
    args = build_parser().parse_args()
    base_dir = Path(__file__).resolve().parent
    manifest_path = base_dir / args.manifest
    output_dir = base_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_manifest(manifest_path)
    train_rows = split_rows(rows, "train")
    val_rows = split_rows(rows, "val")
    test_rows = split_rows(rows, "test")

    if not train_rows:
        raise ValueError("No train rows found in the manifest.")
    if not val_rows:
        raise ValueError("No val rows found in the manifest.")
    if not test_rows:
        raise ValueError("No test rows found in the manifest.")

    label_to_index = build_label_mapping(rows)
    save_label_mapping(label_to_index, output_dir / "labels.json")

    train_dataset = CommandDataset(train_rows, label_to_index, base_dir)
    val_dataset = CommandDataset(val_rows, label_to_index, base_dir)
    test_dataset = CommandDataset(test_rows, label_to_index, base_dir)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CommandCNN(num_classes=len(label_to_index)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    best_val_accuracy = -1.0
    best_path = output_dir / "best_command_model.pt"

    print(f"Training on device: {device}")
    print(f"Train/Val/Test rows: {len(train_rows)}/{len(val_rows)}/{len(test_rows)}")
    print(f"Classes: {len(label_to_index)}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_count = 0

        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(features)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total_correct += int((logits.argmax(dim=1) == labels).sum().item())
            total_count += labels.size(0)

        train_loss = total_loss / total_count
        train_accuracy = total_correct / total_count
        val_metrics = evaluate_model(model, val_loader, loss_fn, device)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_accuracy:.4f} "
            f"val_loss={val_metrics.loss:.4f} "
            f"val_acc={val_metrics.accuracy:.4f}"
        )

        if val_metrics.accuracy > best_val_accuracy:
            best_val_accuracy = val_metrics.accuracy
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "label_to_index": label_to_index,
                },
                best_path,
            )

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate_model(model, test_loader, loss_fn, device)

    print("\nBest checkpoint saved to:", best_path)
    print(f"Final test accuracy: {test_metrics.accuracy:.4f}")
    print(f"Final test loss: {test_metrics.loss:.4f}")


if __name__ == "__main__":
    main()
