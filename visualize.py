"""
UMAP visualization of SimCLR learned representations across training epochs.

Loads per-epoch checkpoints from an experiment directory, extracts backbone
features on CIFAR-10 test images, projects them with UMAP, and saves a
grid of scatter plots (one per epoch) coloured by class label.

Usage:
    python visualize.py <experiment_dir> [--cfg simclr/config/simclr_config.yml]
                                         [--n-samples 2000]
                                         [--output umap_grid.png]

Example:
    python visualize.py artifacts/logs/exp_20250501-120000
"""

import argparse
import glob
import math
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from simclr.model.simclr_model import SimCLR
from simclr.utils.misc import IDX_TO_CLASS, load_config


# ── CIFAR-10 loader (minimal, no augmentations) ──────────────────────────────
# We reuse the raw data loading logic from the project's dataset but apply only
# a deterministic resize + to-tensor so that representations are comparable
# across checkpoints.

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import pickle


class CIFAR10Eval(Dataset):
    """Deterministic CIFAR-10 loader (no random augmentations)."""

    def __init__(self, cfg, split="test"):
        self.base_path = cfg.DATA.root_dir
        if "cifar-10-batches-py" not in self.base_path and os.path.exists(
            os.path.join(self.base_path, "cifar-10-batches-py")
        ):
            self.data_dir = os.path.join(self.base_path, "cifar-10-batches-py")
        else:
            self.data_dir = self.base_path

        self.data = []
        self.labels = []

        file_names = (
            [f"data_batch_{i}" for i in range(1, 6)] if split == "train" else ["test_batch"]
        )
        for fname in file_names:
            with open(os.path.join(self.data_dir, fname), "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                self.labels.extend(entry.get("labels", entry.get("fine_labels")))

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32).transpose((0, 2, 3, 1))

        self.transform = transforms.Compose(
            [
                transforms.Resize((cfg.DATA.img_size, cfg.DATA.img_size)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.fromarray(self.data[idx])
        return self.transform(img), self.labels[idx]


# ── Feature extraction ────────────────────────────────────────────────────────


@torch.no_grad()
def extract_features(model, dataloader, device):
    """Run images through the backbone (without the projection head) and
    return (features, labels) as numpy arrays."""
    all_feats, all_labels = [], []
    model.eval()
    for imgs, labels in dataloader:
        imgs = imgs.to(device)
        feats = model.backbone(imgs)  # 512-d backbone output
        all_feats.append(feats.cpu().numpy())
        all_labels.append(labels.numpy() if isinstance(labels, torch.Tensor) else np.array(labels))
    return np.concatenate(all_feats), np.concatenate(all_labels)


# ── Checkpoint discovery ──────────────────────────────────────────────────────


def find_checkpoints(experiment_dir):
    """Return a sorted list of (epoch_number, filepath) tuples."""
    pattern = os.path.join(experiment_dir, "model_epoch_*.pth")
    paths = glob.glob(pattern)
    if not paths:
        sys.exit(f"No checkpoints found matching {pattern}")

    pairs = []
    for p in paths:
        m = re.search(r"model_epoch_(\d+)\.pth$", p)
        if m:
            pairs.append((int(m.group(1)), p))
    pairs.sort(key=lambda x: x[0])
    return pairs


# ── Plotting ──────────────────────────────────────────────────────────────────

# Distinguishable colour palette for 10 CIFAR-10 classes
CLASS_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
]


def plot_umap_grid(embeddings_per_epoch, labels, output_path):
    """Create a grid of UMAP scatter plots, one per epoch.

    Parameters
    ----------
    embeddings_per_epoch : list[(epoch_num, 2-d array)]
    labels : 1-d int array  (same for every epoch)
    output_path : str
    """
    n = len(embeddings_per_epoch)
    cols = min(6, n)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), dpi=120)
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for idx, (epoch, emb) in enumerate(embeddings_per_epoch):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        for cls_idx in range(10):
            mask = labels == cls_idx
            ax.scatter(
                emb[mask, 0],
                emb[mask, 1],
                c=CLASS_COLORS[cls_idx],
                label=IDX_TO_CLASS[cls_idx],
                s=4,
                alpha=0.6,
            )
        ax.set_title(f"Epoch {epoch}", fontsize=10, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    # Turn off any unused subplot axes
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].axis("off")

    # Shared legend
    handles, leg_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        leg_labels,
        loc="lower center",
        ncol=5,
        fontsize=8,
        markerscale=3,
        frameon=False,
    )

    fig.suptitle("UMAP of SimCLR Backbone Features Across Training", fontsize=14, y=1.0)
    fig.tight_layout(rect=[0, 0.04, 1, 0.98])
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    print(f"Saved grid → {output_path}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="UMAP visualisation of SimCLR representations per epoch."
    )
    parser.add_argument(
        "experiment_dir",
        type=str,
        help="Path to experiment directory containing model_epoch_*.pth checkpoints.",
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default="simclr/config/simclr_config.yml",
        help="Path to the YAML config used during training.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=2000,
        help="Number of test images to embed (default 2000; fewer = faster).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output image path. Defaults to <experiment_dir>/umap_grid.png",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for feature extraction.",
    )
    args = parser.parse_args()

    # ── setup ─────────────────────────────────────────────────────────────
    try:
        import umap
    except ImportError:
        sys.exit("umap-learn is required.  Install it with:  pip install umap-learn")

    cfg = load_config(args.cfg)
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    # ── dataset ───────────────────────────────────────────────────────────
    full_dataset = CIFAR10Eval(cfg, split="test")
    n_samples = min(args.n_samples, len(full_dataset))

    # Deterministic subset so every epoch is projected from the same images
    rng = np.random.RandomState(42)
    indices = rng.choice(len(full_dataset), size=n_samples, replace=False)
    subset = Subset(full_dataset, indices)
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # ── discover checkpoints ──────────────────────────────────────────────
    checkpoints = find_checkpoints(args.experiment_dir)
    print(f"Found {len(checkpoints)} checkpoints in {args.experiment_dir}")

    # ── extract features per epoch ────────────────────────────────────────
    all_feats = []  # list of (epoch, features_array)
    labels = None

    for epoch_num, ckpt_path in checkpoints:
        print(f"  Loading epoch {epoch_num} … ", end="", flush=True)
        model = SimCLR(cfg).to(device)
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state)

        feats, lbls = extract_features(model, loader, device)
        all_feats.append((epoch_num, feats))
        if labels is None:
            labels = lbls
        print(f"features shape {feats.shape}")

        # free memory
        del model, state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── UMAP ──────────────────────────────────────────────────────────────
    # Fit one shared UMAP on the concatenated features from all epochs so
    # the axes are comparable across panels.
    print("Running UMAP on concatenated features …")
    concat = np.concatenate([f for _, f in all_feats], axis=0)
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    all_emb = reducer.fit_transform(concat)

    # Split back into per-epoch arrays
    embeddings_per_epoch = []
    offset = 0
    for epoch_num, feats in all_feats:
        n = feats.shape[0]
        embeddings_per_epoch.append((epoch_num, all_emb[offset : offset + n]))
        offset += n

    # ── plot ───────────────────────────────────────────────────────────────
    output_path = args.output or os.path.join(args.experiment_dir, "umap_grid.png")
    plot_umap_grid(embeddings_per_epoch, labels, output_path)
    print("Done.")


if __name__ == "__main__":
    main()