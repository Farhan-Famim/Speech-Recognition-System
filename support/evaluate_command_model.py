from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .command_model import CommandCNN, CommandDataset, evaluate_model, load_manifest

DEFAULT_EVALUATION_CHECKPOINT = "artifacts/command_model_resplit/best_command_model.pt"


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate a trained speech-command classifier.")
    parser.add_argument("--manifest", default="data/custom_dataset/metadata.csv", help="Path to your metadata CSV.")
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_EVALUATION_CHECKPOINT,
        help="Saved model checkpoint. Use command_model_resplit for report accuracy; command_model is the train-all demo model.",
    )
    parser.add_argument("--split", default="test", help="Which split to evaluate.")
    parser.add_argument("--batch-size", type=int, default=8, help="Mini-batch size.")
    return parser


def main():
    args = build_parser().parse_args()
    base_dir = Path(__file__).resolve().parent.parent
    manifest_path = base_dir / args.manifest
    checkpoint_path = base_dir / args.checkpoint

    if checkpoint_path.parent.name == "command_model":
        print(
            "WARNING: You are evaluating the train-all demo checkpoint. "
            "Do not report this as held-out test accuracy.\n"
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    label_to_index = checkpoint["label_to_index"]
    index_to_label = {value: key for key, value in label_to_index.items()}

    rows = [row for row in load_manifest(manifest_path) if row["split"] == args.split]
    if not rows:
        raise ValueError(f"No rows found for split '{args.split}'.")

    dataset = CommandDataset(rows, label_to_index, base_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CommandCNN(num_classes=len(label_to_index)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    loss_fn = nn.CrossEntropyLoss()

    metrics = evaluate_model(model, loader, loss_fn, device)
    print(f"Accuracy on split '{args.split}': {metrics.accuracy:.4f}")
    print(f"Loss on split '{args.split}': {metrics.loss:.4f}\n")

    model.eval()
    with torch.no_grad():
        for row, (features, _) in zip(rows, loader.dataset):
            logits = model(features.unsqueeze(0).to(device))
            predicted_index = int(logits.argmax(dim=1).item())
            predicted_label = index_to_label[predicted_index]
            print(f"FILE: {row['audio_path']}")
            print(f"TRUE: {row['command_id']}")
            print(f"PRED: {predicted_label}")
            print("-" * 40)


if __name__ == "__main__":
    main()
