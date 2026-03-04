"""
DDPM Training for Microplastics
================================
Trains a class-label-conditioned denoising diffusion model (DDPM) on
individual microplastic object crops extracted from the COCO-annotated dataset.

Usage:
    python train_diffusion.py --num-epochs 300 --batch-size 16
    python train_diffusion.py --num-epochs 50 --batch-size 4          # CPU test
    python train_diffusion.py --resume diffusion_checkpoints/checkpoint-100
"""

import argparse
import math
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler, UNet2DModel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from generate_synthetic import extract_all_objects, load_coco

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

CLASS_NAMES = {1: "Fiber", 2: "Fragment", 3: "Film"}
# Map category IDs (1,2,3) to contiguous label indices (0,1,2)
CATID_TO_LABEL = {1: 0, 2: 1, 3: 2}
NUM_CLASSES = 3


class MicroplasticCropDataset(Dataset):
    """Yields square, resized object crops with class labels.

    Each crop is the RGB image masked by the object mask (black background),
    padded to square, resized to ``image_size x image_size``, and normalised
    to [-1, 1].  Minority classes are oversampled so every class appears the
    same number of times per epoch.
    """

    def __init__(self, object_pool: dict[int, list], image_size: int = 128):
        self.image_size = image_size
        self.samples: list[tuple[np.ndarray, int]] = []  # (crop_uint8_HWC, label)

        # Convert RGBA patches → masked RGB crops
        for cat_id, patches in object_pool.items():
            label = CATID_TO_LABEL[cat_id]
            for patch in patches:
                rgba = patch.image_crop  # H,W,4
                mask = patch.mask_crop   # H,W
                rgb = rgba[:, :, :3].copy()
                rgb[mask == 0] = 0
                self.samples.append((rgb, label))

        # Oversample minority classes to equalise counts
        counts = {}
        for _, label in self.samples:
            counts[label] = counts.get(label, 0) + 1
        max_count = max(counts.values())

        oversampled: list[tuple[np.ndarray, int]] = []
        by_label: dict[int, list[tuple[np.ndarray, int]]] = {}
        for s in self.samples:
            by_label.setdefault(s[1], []).append(s)

        rng = np.random.default_rng(42)
        for label, items in by_label.items():
            oversampled.extend(items)
            deficit = max_count - len(items)
            if deficit > 0:
                extras = [items[i % len(items)] for i in rng.integers(0, len(items), size=deficit)]
                oversampled.extend(extras)

        self.samples = oversampled
        rng.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        crop, label = self.samples[idx]
        crop = self._pad_and_resize(crop)
        # HWC uint8 → CHW float [-1, 1]
        tensor = torch.from_numpy(crop).permute(2, 0, 1).float() / 127.5 - 1.0
        return tensor, label

    def _pad_and_resize(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        size = max(h, w)
        padded = np.zeros((size, size, 3), dtype=np.uint8)
        y_off = (size - h) // 2
        x_off = (size - w) // 2
        padded[y_off:y_off + h, x_off:x_off + w] = img
        resized = cv2.resize(padded, (self.image_size, self.image_size),
                             interpolation=cv2.INTER_AREA)
        return resized


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def create_model(image_size: int = 128, num_classes: int = NUM_CLASSES) -> UNet2DModel:
    """Build a UNet2DModel sized for ``image_size`` with class-label conditioning."""
    return UNet2DModel(
        sample_size=image_size,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
        num_class_embeds=num_classes,
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- Load dataset ----
    print("Loading COCO annotations ...")
    id2img, id2cat, imgid2anns, _ = load_coco(args.annotation)

    print("Extracting object crops ...")
    object_pool = extract_all_objects(id2img, imgid2anns, args.image_dir)
    for cat_id, patches in sorted(object_pool.items()):
        name = CLASS_NAMES.get(cat_id, f"id={cat_id}")
        print(f"  {name}: {len(patches)} patches")

    dataset = MicroplasticCropDataset(object_pool, image_size=args.image_size)
    print(f"Dataset size (after oversampling): {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    # ---- Model & scheduler ----
    start_epoch = 0
    if args.resume:
        print(f"Resuming from {args.resume} ...")
        model = UNet2DModel.from_pretrained(args.resume)
        # Try to infer start epoch from checkpoint dir name
        dirname = os.path.basename(args.resume.rstrip("/"))
        if dirname.startswith("checkpoint-"):
            try:
                start_epoch = int(dirname.split("-")[1])
            except ValueError:
                pass
    else:
        model = create_model(image_size=args.image_size)

    model = model.to(device)

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule="squaredcos_cap_v2",
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Mixed precision
    use_amp = (device.type == "cuda") and args.fp16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ---- Training ----
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    global_step = 0

    for epoch in range(start_epoch, start_epoch + args.num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{start_epoch + args.num_epochs}")
        for images, labels in pbar:
            images = images.to(device)                  # (B, 3, H, W)
            labels = labels.to(device)                  # (B,)

            noise = torch.randn_like(images)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (images.shape[0],), device=device,
            ).long()

            noisy_images = noise_scheduler.add_noise(images, noise, timesteps)

            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(noisy_images, timesteps, class_labels=labels).sample
                loss = F.mse_loss(pred, noise)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"  Epoch {epoch + 1} — avg loss: {avg_loss:.5f}")

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == (start_epoch + args.num_epochs):
            ckpt_path = os.path.join(args.checkpoint_dir, f"checkpoint-{epoch + 1}")
            model.save_pretrained(ckpt_path)
            noise_scheduler.save_pretrained(ckpt_path)
            print(f"  Saved checkpoint → {ckpt_path}")

    print("Training complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train a class-conditioned DDPM on microplastic object crops"
    )
    parser.add_argument("--image-size", type=int, default=128,
                        help="Square crop size in pixels (default: 128)")
    parser.add_argument("--num-epochs", type=int, default=300,
                        help="Number of training epochs (default: 300)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (default: 1e-4)")
    parser.add_argument("--save-every", type=int, default=50,
                        help="Save checkpoint every N epochs (default: 50)")
    parser.add_argument("--checkpoint-dir", type=str, default="diffusion_checkpoints",
                        help="Directory for saving checkpoints (default: diffusion_checkpoints)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint directory to resume from")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--annotation", type=str, default="annotation.json",
                        help="Path to COCO annotation file")
    parser.add_argument("--image-dir", type=str, default="images",
                        help="Path to source images directory")
    parser.add_argument("--fp16", action="store_true",
                        help="Use mixed precision training (GPU only)")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
