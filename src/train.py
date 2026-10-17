"""
Checkpointed training loop for both the baseline (cross-entropy only) and
treatment (cross-entropy + MSA-hicl) runs.

Resumability: after every epoch we save model + optimizer state (and, for
the hicl run, the momentum-cache prototypes) to
checkpoints/<run_name>.pt. If that file already exists and reports a
finished epoch >= --epochs, training is skipped entirely; if it exists but
is partway through, we resume from the next epoch. This is what let the
original Colab runs survive free-tier session disconnects without losing
work, and it is what makes `scripts/run_all.sh` safe to re-run after an
interruption instead of re-training from scratch every time.

Usage
-----
    python -m src.train --run baseline --data-dir data --epochs 15
    python -m src.train --run hicl     --data-dir data --epochs 15 --hicl-weight 0.1
"""

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data import CUBDataset, parse_cub_metadata
from src.download_data import ensure_cub_dataset
from src.hierarchy import build_taxonomy
from src.losses import MomentumCache, msa_hicl_loss
from src.model import make_model


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():  # Apple Silicon, for completeness
        return "mps"
    return "cpu"


def build_loaders(data_dir: str, batch_size_train: int = 32, batch_size_test: int = 64):
    cub_root = ensure_cub_dataset(data_dir)
    meta = parse_cub_metadata(cub_root)
    train_set = CUBDataset(cub_root, meta, train=True)
    test_set = CUBDataset(cub_root, meta, train=False)
    train_loader = DataLoader(train_set, batch_size=batch_size_train, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size_test, shuffle=False, num_workers=2)
    class_names = [meta["class_names"][i] for i in range(len(meta["class_names"]))]
    return train_loader, test_loader, class_names


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total


def train_model(
    run_name: str,
    use_hicl: bool,
    data_dir: str,
    checkpoint_dir: str,
    epochs: int,
    hicl_weight: float,
    lr: float,
    seed: int,
) -> float:
    torch.manual_seed(seed)
    device = get_device()
    print(f"Device: {device}")

    train_loader, test_loader, class_names = build_loaders(data_dir)
    model = make_model(num_classes=200, pretrained=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ce_loss_fn = nn.CrossEntropyLoss()

    taxonomy = None
    cache = None
    if use_hicl:
        taxonomy = build_taxonomy(cache_dir=data_dir, cub_class_names=class_names, num_classes=200)
        cache = MomentumCache(taxonomy.node_index, feature_dim=2048, momentum=0.99, device=device)

    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, f"{run_name}.pt")
    start_epoch = 1

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if use_hicl and "prototypes" in ckpt:
            cache.prototypes = ckpt["prototypes"].to(device)
        start_epoch = ckpt["epoch"] + 1
        print(f"Resuming {run_name} from epoch {start_epoch} "
              f"(checkpoint says epoch {ckpt['epoch']} finished).")
        if start_epoch > epochs:
            print(f"{run_name}: already finished {ckpt['epoch']} epochs >= requested {epochs}. "
                  "Skipping training.")
            return evaluate(model, test_loader, device)

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        running_ce, running_hicl, n_batches = 0.0, 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            if use_hicl:
                logits, features = model(images, return_features=True)
                ce = ce_loss_fn(logits, labels)
                hicl, leaf_node_ids = msa_hicl_loss(features, labels, taxonomy, cache)
                loss = ce + hicl_weight * hicl
                cache.update(leaf_node_ids, features.detach())
                running_hicl += hicl.item()
            else:
                logits = model(images)
                ce = ce_loss_fn(logits, labels)
                loss = ce

            loss.backward()
            optimizer.step()
            running_ce += ce.item()
            n_batches += 1

        msg = f"[{run_name}] epoch {epoch}/{epochs}  CE={running_ce / n_batches:.4f}"
        if use_hicl:
            msg += f"  hicl={running_hicl / n_batches:.4f}"
        print(msg)

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }
        if use_hicl:
            ckpt["prototypes"] = cache.prototypes.detach().cpu()
        torch.save(ckpt, ckpt_path)

    return evaluate(model, test_loader, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", choices=["baseline", "hicl"], required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--hicl-weight", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    acc = train_model(
        run_name=args.run,
        use_hicl=(args.run == "hicl"),
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        hicl_weight=args.hicl_weight,
        lr=args.lr,
        seed=args.seed,
    )
    print(f"{args.run} test accuracy: {acc:.2f}%")
