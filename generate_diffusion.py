"""
Diffusion-based Synthetic Image Generator for Microplastics
============================================================
Generates synthetic microplastic crops using a trained DDPM checkpoint, with
an optional composition mode that places generated crops onto backgrounds to
produce full-scene images with COCO annotations.

Usage:
    # Generate crops only
    python generate_diffusion.py --checkpoint diffusion_checkpoints/checkpoint-300 \
        --num-per-class 50

    # Generate composed scenes with COCO annotations
    python generate_diffusion.py --checkpoint diffusion_checkpoints/checkpoint-300 \
        --compose --num-images 200 --background-mode inpaint
"""

import argparse
import json
import os

import cv2
import numpy as np
import torch
from diffusers import DDPMScheduler, UNet2DModel
from tqdm import tqdm

from generate_synthetic import (
    SyntheticConfig,
    ObjectPatch,
    compose_synthetic_image,
    compute_bbox,
    generate_backgrounds,
    load_coco,
    mask_to_poly,
)
from train_diffusion import CLASS_NAMES, CATID_TO_LABEL, NUM_CLASSES

# Reverse mapping: label index → category id
LABEL_TO_CATID = {v: k for k, v in CATID_TO_LABEL.items()}


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_crops(
    model: UNet2DModel,
    scheduler: DDPMScheduler,
    class_label: int,
    num_images: int,
    image_size: int,
    device: torch.device,
    batch_size: int = 16,
    seed: int | None = None,
) -> list[np.ndarray]:
    """Generate ``num_images`` crops for a single class label.

    Returns a list of uint8 RGB numpy arrays (H, W, 3).
    """
    model.eval()
    crops: list[np.ndarray] = []
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)

    remaining = num_images
    while remaining > 0:
        bs = min(batch_size, remaining)

        # Start from pure noise
        sample = torch.randn(
            (bs, 3, image_size, image_size),
            device=device,
            generator=generator,
        )
        labels = torch.full((bs,), class_label, dtype=torch.long, device=device)

        # Reverse diffusion
        scheduler.set_timesteps(scheduler.config.num_train_timesteps)
        for t in scheduler.timesteps:
            t_batch = t.expand(bs).to(device)
            pred = model(sample, t_batch, class_labels=labels).sample
            sample = scheduler.step(pred, t, sample).prev_sample

        # Convert to uint8 images
        images = (sample.clamp(-1, 1) + 1) / 2 * 255
        images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

        for img in images:
            # Convert RGB → BGR for OpenCV consistency
            crops.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        remaining -= bs

    return crops[:num_images]


def crop_to_object_patch(crop_bgr: np.ndarray, category_id: int,
                         threshold: int = 15) -> ObjectPatch | None:
    """Extract an ObjectPatch from a generated crop by thresholding.

    Pixels close to black (background) are masked out.  Returns None if the
    resulting mask is too small.
    """
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 1, cv2.THRESH_BINARY)

    # Clean up small noise
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    if mask.sum() < 50:
        return None

    # Tight crop
    bbox = compute_bbox(mask)
    x, y, bw, bh = bbox
    if bw < 4 or bh < 4:
        return None

    img_crop = crop_bgr[y:y + bh, x:x + bw].copy()
    mask_crop = mask[y:y + bh, x:x + bw].copy()

    # Build RGBA
    rgba = np.zeros((bh, bw, 4), dtype=np.uint8)
    rgba[:, :, :3] = img_crop
    rgba[:, :, 3] = mask_crop * 255

    return ObjectPatch(image_crop=rgba, mask_crop=mask_crop, category_id=category_id)


# ---------------------------------------------------------------------------
# Mode A: Save individual crops
# ---------------------------------------------------------------------------

def save_crops(crops_by_class: dict[int, list[np.ndarray]], output_dir: str):
    """Save generated crops as PNG files organised by class."""
    for cat_id, crops in crops_by_class.items():
        class_name = CLASS_NAMES.get(cat_id, f"class_{cat_id}")
        class_dir = os.path.join(output_dir, "crops", class_name)
        os.makedirs(class_dir, exist_ok=True)
        for i, crop in enumerate(crops):
            # Save with alpha channel
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
            rgba = np.zeros((*crop.shape[:2], 4), dtype=np.uint8)
            rgba[:, :, :3] = crop
            rgba[:, :, 3] = mask
            path = os.path.join(class_dir, f"{class_name}_{i + 1:04d}.png")
            cv2.imwrite(path, rgba)
    print(f"Crops saved to {os.path.join(output_dir, 'crops')}")


# ---------------------------------------------------------------------------
# Mode B: Compose full scenes
# ---------------------------------------------------------------------------

def compose_scenes(
    crops_by_class: dict[int, list[np.ndarray]],
    args,
):
    """Compose generated crops onto backgrounds, producing images + COCO JSON."""
    rng = np.random.default_rng(args.seed)

    # Build object pool from generated crops
    object_pool: dict[int, list[ObjectPatch]] = {}
    for cat_id, crops in crops_by_class.items():
        patches = []
        for crop in crops:
            patch = crop_to_object_patch(crop, cat_id)
            if patch is not None:
                patches.append(patch)
        if patches:
            object_pool[cat_id] = patches
    print(f"Usable patches from diffusion crops: "
          f"{sum(len(v) for v in object_pool.values())}")

    if not object_pool:
        print("ERROR: No usable patches extracted. Try generating more crops "
              "or lowering the threshold.")
        return

    # Load source dataset for backgrounds
    print("Loading source annotations for backgrounds ...")
    id2img, id2cat, imgid2anns, categories = load_coco(args.annotation)

    config = SyntheticConfig(
        annotation_path=args.annotation,
        image_dir=args.image_dir,
        background_mode=args.background_mode,
        num_backgrounds=min(args.num_images, 100),
    )

    print(f"Generating backgrounds ({args.background_mode}) ...")
    backgrounds = generate_backgrounds(id2img, imgid2anns, config, rng)

    # Compose
    output_dir = args.output_dir
    out_img_dir = os.path.join(output_dir, "images")
    os.makedirs(out_img_dir, exist_ok=True)

    all_images = []
    all_annotations = []
    ann_id = 1

    print(f"Composing {args.num_images} synthetic scenes ...")
    for i in tqdm(range(args.num_images), desc="Composing"):
        bg = backgrounds[rng.integers(len(backgrounds))].copy()
        image, annotations = compose_synthetic_image(bg, object_pool, config, rng)

        img_id = i + 1
        filename = f"diffusion_{img_id:06d}.jpg"

        for ann in annotations:
            ann["id"] = ann_id
            ann["image_id"] = img_id
            ann_id += 1
            all_annotations.append(ann)

        all_images.append({
            "id": img_id,
            "file_name": filename,
            "width": config.output_width,
            "height": config.output_height,
        })

        cv2.imwrite(
            os.path.join(out_img_dir, filename),
            image,
            [cv2.IMWRITE_JPEG_QUALITY, config.jpg_quality],
        )

    # Save COCO JSON
    coco_output = {
        "images": all_images,
        "categories": categories,
        "annotations": all_annotations,
    }
    ann_path = os.path.join(output_dir, "synthetic_annotation.json")
    with open(ann_path, "w") as f:
        json.dump(coco_output, f)

    print(f"\n=== Composition Summary ===")
    print(f"Images: {len(all_images)}")
    print(f"Annotations: {len(all_annotations)}")
    print(f"Output: {output_dir}")
    print(f"COCO JSON: {ann_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print(f"Loading checkpoint from {args.checkpoint} ...")
    model = UNet2DModel.from_pretrained(args.checkpoint).to(device)
    scheduler = DDPMScheduler.from_pretrained(args.checkpoint)

    image_size = model.config.sample_size

    # Determine which classes to generate
    if args.classes:
        name_to_catid = {v.lower(): k for k, v in CLASS_NAMES.items()}
        cat_ids = []
        for name in args.classes:
            cid = name_to_catid.get(name.lower())
            if cid is None:
                print(f"WARNING: Unknown class '{name}', skipping. "
                      f"Valid: {list(CLASS_NAMES.values())}")
            else:
                cat_ids.append(cid)
    else:
        cat_ids = sorted(CLASS_NAMES.keys())

    # Generate crops for each class
    crops_by_class: dict[int, list[np.ndarray]] = {}
    for cat_id in cat_ids:
        label = CATID_TO_LABEL[cat_id]
        name = CLASS_NAMES[cat_id]
        n = args.num_per_class
        print(f"Generating {n} crops for {name} (label={label}) ...")
        crops = generate_crops(
            model, scheduler, label, n, image_size, device,
            batch_size=args.batch_size,
            seed=args.seed + cat_id,
        )
        crops_by_class[cat_id] = crops
        print(f"  Generated {len(crops)} crops")

    # Output
    os.makedirs(args.output_dir, exist_ok=True)

    if args.compose:
        # Mode B: compose full scenes
        compose_scenes(crops_by_class, args)
    else:
        # Mode A: save individual crops
        save_crops(crops_by_class, args.output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic microplastic images using a trained DDPM"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to diffusers checkpoint directory")
    parser.add_argument("--num-per-class", type=int, default=50,
                        help="Number of crops to generate per class (default: 50)")
    parser.add_argument("--classes", nargs="*", default=None,
                        help="Classes to generate (default: all). "
                             "E.g. --classes Fiber Film")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Inference batch size (default: 16)")
    parser.add_argument("--output-dir", type=str, default="diffusion_output",
                        help="Output directory (default: diffusion_output)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")

    # Composition mode
    parser.add_argument("--compose", action="store_true",
                        help="Compose generated crops into full scene images")
    parser.add_argument("--num-images", type=int, default=100,
                        help="Number of composed scene images (default: 100)")
    parser.add_argument("--background-mode", type=str, default="inpaint",
                        choices=["inpaint", "crop", "noise", "blank"],
                        help="Background mode for composition (default: inpaint)")
    parser.add_argument("--annotation", type=str, default="annotation.json",
                        help="COCO annotation file (for backgrounds)")
    parser.add_argument("--image-dir", type=str, default="images",
                        help="Source images directory (for backgrounds)")

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
