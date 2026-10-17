"""
Reproduce (a scaled-down version of) the paper's Figure 6: a t-SNE
projection of test-set features, side by side for the baseline and hicl
models, to check the qualitative claim that MSA-hicl produces features
that cluster by genus rather than only by species.

We pick 8 species across 4 real-taxonomy genus groups (not 4
common-name groups) so the "does it cluster by genus" question is
answered against the same taxonomy the loss itself uses.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE

from src.data import CUBDataset, parse_cub_metadata
from src.download_data import ensure_cub_dataset
from src.model import make_model
from src.train import get_device

# 8 species, 4 genus pairs, matched to the real cvjena taxonomy's immediate
# parent for each (see src/hierarchy.py: Taxonomy.genus_of).
SPECIES_SUBSET = [
    "Black_footed_Albatross", "Laysan_Albatross",
    "Crested_Auklet", "Least_Auklet",
    "Brewer_Blackbird", "Rusty_Blackbird",
    "Indigo_Bunting", "Lazuli_Bunting",
]


@torch.no_grad()
def extract_features(model, dataset, class_names, device):
    wanted_ids = {
        i for i, name in enumerate(class_names)
        if any(name.endswith(s) or s in name for s in SPECIES_SUBSET)
    }
    feats, labels = [], []
    for image, label in dataset:
        if label not in wanted_ids:
            continue
        _, feature = model(image.unsqueeze(0).to(device), return_features=True)
        feats.append(feature.squeeze(0).cpu().numpy())
        labels.append(class_names[label])
    return feats, labels


def make_panel(ax, feats, labels, title):
    projected = TSNE(n_components=2, random_state=0, init="pca").fit_transform(feats)
    unique_labels = sorted(set(labels))
    for lbl in unique_labels:
        idx = [i for i, l in enumerate(labels) if l == lbl]
        xs = [projected[i, 0] for i in idx]
        ys = [projected[i, 1] for i in idx]
        ax.scatter(xs, ys, label=lbl, s=12)
    ax.set_title(title)
    ax.legend(fontsize=6, loc="best")


def main(checkpoint_dir: str, data_dir: str, out_path: str):
    device = get_device()
    cub_root = ensure_cub_dataset(data_dir)
    meta = parse_cub_metadata(cub_root)
    class_names = [meta["class_names"][i] for i in range(len(meta["class_names"]))]
    test_set = CUBDataset(cub_root, meta, train=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, run_name, title in [
        (axes[0], "baseline", "Cross-entropy only"),
        (axes[1], "hicl", "Cross-entropy + MSA-hicl"),
    ]:
        ckpt_path = os.path.join(checkpoint_dir, f"{run_name}.pt")
        ckpt = torch.load(ckpt_path, map_location=device)
        model = make_model(num_classes=200, pretrained=False).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        feats, labels = extract_features(model, test_set, class_names, device)
        make_panel(ax, feats, labels, title)

    fig.suptitle("t-SNE of test-set features, 8 species / 4 genus groups")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--out", default="figures/tsne_comparison.png")
    args = parser.parse_args()
    main(args.checkpoint_dir, args.data_dir, args.out)
