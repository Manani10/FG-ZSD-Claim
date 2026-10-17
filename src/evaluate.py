"""
Load both finished checkpoints (no retraining) and write results/results.json.

This is the "give me the numbers" step: it does not train anything, it
just loads whatever checkpoints/baseline.pt and checkpoints/hicl.pt already
contain and reports their test accuracy, plus the difference. If a
checkpoint is not yet at --epochs, it says so rather than reporting a
partial number as final.
"""

import argparse
import json
import os

import torch

from src.train import build_loaders, evaluate, get_device
from src.model import make_model


def load_and_evaluate(run_name: str, checkpoint_dir: str, data_dir: str, epochs: int) -> dict:
    device = get_device()
    ckpt_path = os.path.join(checkpoint_dir, f"{run_name}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"No checkpoint at {ckpt_path}. Run "
            f"`python -m src.train --run {run_name}` first."
        )
    ckpt = torch.load(ckpt_path, map_location=device)
    finished_epoch = ckpt["epoch"]

    # weights=None: the real weights come from the checkpoint, not from
    # re-downloading ImageNet pretrained weights.
    model = make_model(num_classes=200, pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])

    _, test_loader, _ = build_loaders(data_dir)
    acc = evaluate(model, test_loader, device)

    return {
        "run": run_name,
        "finished_epoch": finished_epoch,
        "requested_epochs": epochs,
        "complete": finished_epoch >= epochs,
        "test_accuracy": round(acc, 2),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--results-path", default="results/results.json")
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()

    baseline = load_and_evaluate("baseline", args.checkpoint_dir, args.data_dir, args.epochs)
    hicl = load_and_evaluate("hicl", args.checkpoint_dir, args.data_dir, args.epochs)

    results = {
        "baseline": baseline,
        "hicl": hicl,
        "difference_points": round(hicl["test_accuracy"] - baseline["test_accuracy"], 2),
    }

    os.makedirs(os.path.dirname(args.results_path), exist_ok=True)
    with open(args.results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
