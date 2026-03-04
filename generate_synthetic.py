"""
Synthetic Data Generator for Microplastics Segmentation
========================================================
Generates synthetic training images using patch-based composition:
  1. Extracts microplastic objects from real images using COCO masks
  2. Pastes them onto clean backgrounds with augmentations
  3. Produces new images + COCO-format annotations

Usage:
    python generate_synthetic.py --num-images 500
    python generate_synthetic.py --num-images 1000 --film-weight 12.0 --background-mode noise
"""

import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SyntheticConfig:
    annotation_path: str = "annotation.json"
    image_dir: str = "images"
    output_dir: str = "synthetic_output"
    output_annotation: str = "synthetic_annotation.json"

    num_images: int = 500
    seed: int = 42

    # Class weights for sampling (higher = more of that class)
    class_weights: dict = field(default_factory=lambda: {
        1: 1.0,    # Fiber   (already 69% of real data)
        2: 2.5,    # Fragment (25% of real data)
        3: 8.0,    # Film    (only 5% of real data)
    })

    min_objects: int = 1
    max_objects: int = 4

    # Augmentation parameters
    scale_range: tuple = (0.5, 1.5)
    rotation_range: tuple = (0.0, 360.0)
    flip_horizontal: bool = True
    flip_vertical: bool = True
    brightness_range: tuple = (0.8, 1.2)
    blur_probability: float = 0.3
    blur_kernel_range: tuple = (3, 7)

    # Placement constraints
    max_overlap_ratio: float = 0.15
    border_margin: int = 10
    max_placement_attempts: int = 50
    min_object_area: int = 20

    # Background generation
    background_mode: str = "inpaint"   # inpaint, crop, noise, blank
    num_backgrounds: int = 100

    # Output image specs
    output_width: int = 640
    output_height: int = 480
    jpg_quality: int = 95


# ---------------------------------------------------------------------------
# COCO I/O helpers
# ---------------------------------------------------------------------------

def load_coco(path: str):
    """Load COCO annotations and build lookup dicts."""
    with open(path, "r") as f:
        data = json.load(f)

    id2img = {img["id"]: img for img in data["images"]}
    id2cat = {cat["id"]: cat for cat in data["categories"]}
    categories = data["categories"]

    imgid2anns = {}
    for ann in data["annotations"]:
        imgid2anns.setdefault(ann["image_id"], []).append(ann)

    return id2img, id2cat, imgid2anns, categories


def poly_to_mask(segmentation: list, height: int, width: int) -> np.ndarray:
    """Convert COCO polygon segmentation to binary mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    for poly in segmentation:
        pts = np.array(poly, dtype=np.float32).reshape(-1, 2).astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)
    return mask


def mask_to_poly(mask: np.ndarray) -> list:
    """Convert binary mask to COCO polygon format."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    polygons = []
    for contour in contours:
        if contour.shape[0] >= 3:
            polygons.append(contour.flatten().tolist())
    return polygons


def compute_bbox(mask: np.ndarray) -> list:
    """Compute [x, y, w, h] bounding box from binary mask."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    x, y = int(xs.min()), int(ys.min())
    w, h = int(xs.max()) - x + 1, int(ys.max()) - y + 1
    return [x, y, w, h]


# ---------------------------------------------------------------------------
# Object patch extraction
# ---------------------------------------------------------------------------

@dataclass
class ObjectPatch:
    image_crop: np.ndarray    # RGBA (H, W, 4)
    mask_crop: np.ndarray     # binary (H, W), uint8
    category_id: int


def extract_all_objects(
    id2img: dict,
    imgid2anns: dict,
    image_dir: str,
) -> dict[int, list[ObjectPatch]]:
    """Extract all annotated objects as RGBA patches grouped by category."""
    pool: dict[int, list[ObjectPatch]] = {}
    image_cache: dict[int, np.ndarray] = {}

    all_anns = []
    for img_id, anns in imgid2anns.items():
        for ann in anns:
            all_anns.append((img_id, ann))

    for img_id, ann in tqdm(all_anns, desc="Extracting objects"):
        img_info = id2img[img_id]
        h, w = img_info["height"], img_info["width"]

        # Load image (cached)
        if img_id not in image_cache:
            path = os.path.join(image_dir, img_info["file_name"])
            img = cv2.imread(path)
            if img is None:
                continue
            image_cache[img_id] = img
        img = image_cache[img_id]

        # Use actual image dimensions (may differ from annotation metadata)
        actual_h, actual_w = img.shape[:2]

        # Convert segmentation to mask
        mask = poly_to_mask(ann["segmentation"], actual_h, actual_w)

        # Tight bounding box crop
        bbox = compute_bbox(mask)
        x, y, bw, bh = bbox
        if bw < 2 or bh < 2:
            continue

        img_crop = img[y:y+bh, x:x+bw].copy()
        mask_crop = mask[y:y+bh, x:x+bw].copy()

        # Create soft alpha edges
        alpha = create_soft_alpha(mask_crop)

        # Build RGBA
        rgba = np.zeros((bh, bw, 4), dtype=np.uint8)
        rgba[:, :, :3] = img_crop
        rgba[:, :, 3] = alpha

        cat_id = ann["category_id"]
        pool.setdefault(cat_id, []).append(
            ObjectPatch(image_crop=rgba, mask_crop=mask_crop, category_id=cat_id)
        )

    # Free image cache
    image_cache.clear()
    return pool


def create_soft_alpha(mask: np.ndarray) -> np.ndarray:
    """Create alpha channel with soft edges from binary mask."""
    # Erode slightly to get solid interior
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(mask, kernel, iterations=1)

    # Edge band = original - eroded
    edge = mask.astype(np.float32) - eroded.astype(np.float32)
    edge = np.clip(edge, 0, 1)

    # Blur the edge band for soft transition
    if edge.sum() > 0:
        edge_blurred = cv2.GaussianBlur(edge, (5, 5), sigmaX=1.0)
    else:
        edge_blurred = edge

    # Combine: solid interior + soft edges
    alpha = eroded.astype(np.float32) + edge_blurred
    alpha = np.clip(alpha, 0, 1)
    return (alpha * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Background generation
# ---------------------------------------------------------------------------

def generate_backgrounds(
    id2img: dict,
    imgid2anns: dict,
    config: SyntheticConfig,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Generate a pool of clean background images."""
    mode = config.background_mode
    h, w = config.output_height, config.output_width

    if mode == "blank":
        return _backgrounds_blank(id2img, config, rng)
    elif mode == "noise":
        return _backgrounds_noise(id2img, config, rng)
    elif mode == "inpaint":
        return _backgrounds_inpaint(id2img, imgid2anns, config, rng)
    elif mode == "crop":
        return _backgrounds_crop(id2img, imgid2anns, config, rng)
    else:
        raise ValueError(f"Unknown background mode: {mode}")


def _sample_image_stats(id2img: dict, config: SyntheticConfig, n_sample: int = 30):
    """Compute mean and std pixel values from a sample of images."""
    rng = np.random.default_rng(0)
    img_ids = list(id2img.keys())
    sample_ids = rng.choice(img_ids, size=min(n_sample, len(img_ids)), replace=False)
    pixels = []
    for img_id in sample_ids:
        path = os.path.join(config.image_dir, id2img[img_id]["file_name"])
        img = cv2.imread(path)
        if img is not None:
            pixels.append(img.mean(axis=(0, 1)))
    pixels = np.array(pixels)
    return pixels.mean(axis=0), pixels.std(axis=0)


def _backgrounds_blank(id2img, config, rng):
    """Solid color backgrounds using dataset median."""
    mean, _ = _sample_image_stats(id2img, config)
    bg = np.full((config.output_height, config.output_width, 3), mean, dtype=np.uint8)
    return [bg.copy() for _ in range(config.num_backgrounds)]


def _backgrounds_noise(id2img, config, rng):
    """Gaussian noise backgrounds matching dataset statistics."""
    mean, std = _sample_image_stats(id2img, config)
    backgrounds = []
    for _ in range(config.num_backgrounds):
        noise = rng.normal(mean, std * 3, (config.output_height, config.output_width, 3))
        backgrounds.append(np.clip(noise, 0, 255).astype(np.uint8))
    return backgrounds


def _backgrounds_inpaint(id2img, imgid2anns, config, rng):
    """Inpaint objects out of real images to get clean backgrounds."""
    img_ids = list(id2img.keys())
    rng.shuffle(img_ids)
    backgrounds = []

    for img_id in img_ids:
        if len(backgrounds) >= config.num_backgrounds:
            break

        img_info = id2img[img_id]
        path = os.path.join(config.image_dir, img_info["file_name"])
        img = cv2.imread(path)
        if img is None:
            continue

        h, w = img_info["height"], img_info["width"]
        anns = imgid2anns.get(img_id, [])

        if not anns:
            # No objects - use as-is
            backgrounds.append(img)
            continue

        # Build combined mask of all objects
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        for ann in anns:
            obj_mask = poly_to_mask(ann["segmentation"], h, w)
            combined_mask = np.maximum(combined_mask, obj_mask)

        # Dilate mask slightly for better inpainting
        kernel = np.ones((7, 7), dtype=np.uint8)
        dilated = cv2.dilate(combined_mask, kernel, iterations=2)

        # Inpaint
        inpainted = cv2.inpaint(img, dilated * 255, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
        backgrounds.append(inpainted)

    # If not enough, duplicate some
    while len(backgrounds) < config.num_backgrounds:
        backgrounds.append(backgrounds[rng.integers(len(backgrounds))].copy())

    return backgrounds


def _backgrounds_crop(id2img, imgid2anns, config, rng):
    """Crop object-free regions from source images."""
    # Fallback to inpaint if can't find enough clean regions
    return _backgrounds_inpaint(id2img, imgid2anns, config, rng)


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def augment_patch(
    patch: ObjectPatch,
    config: SyntheticConfig,
    rng: np.random.Generator,
) -> Optional[ObjectPatch]:
    """Apply random augmentations to an object patch. Returns None if result is too small."""
    img = patch.image_crop.copy()    # RGBA
    mask = patch.mask_crop.copy()

    # 1. Random rotation
    angle = rng.uniform(*config.rotation_range)
    img, mask = _rotate(img, mask, angle)

    # 2. Random scale
    scale = rng.uniform(*config.scale_range)
    new_w = max(1, int(img.shape[1] * scale))
    new_h = max(1, int(img.shape[0] * scale))
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # Check minimum area
    if np.sum(mask) < config.min_object_area:
        return None

    # 3. Random flips
    if config.flip_horizontal and rng.random() > 0.5:
        img = cv2.flip(img, 1)
        mask = cv2.flip(mask, 1)
    if config.flip_vertical and rng.random() > 0.5:
        img = cv2.flip(img, 0)
        mask = cv2.flip(mask, 0)

    # 4. Brightness jitter (RGB channels only)
    factor = rng.uniform(*config.brightness_range)
    rgb = img[:, :, :3].astype(np.float32) * factor
    img[:, :, :3] = np.clip(rgb, 0, 255).astype(np.uint8)

    # 5. Optional Gaussian blur
    if rng.random() < config.blur_probability:
        k_min, k_max = config.blur_kernel_range
        ksize = rng.choice(range(k_min, k_max + 1, 2))
        img[:, :, :3] = cv2.GaussianBlur(img[:, :, :3], (ksize, ksize), 0)

    return ObjectPatch(image_crop=img, mask_crop=mask, category_id=patch.category_id)


def _rotate(img: np.ndarray, mask: np.ndarray, angle: float):
    """Rotate image and mask, expanding canvas to avoid clipping."""
    h, w = img.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)

    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    rotated_img = cv2.warpAffine(img, M, (new_w, new_h), borderValue=(0, 0, 0, 0))
    rotated_mask = cv2.warpAffine(mask, M, (new_w, new_h), borderValue=0)

    return rotated_img, rotated_mask


# ---------------------------------------------------------------------------
# Composition engine
# ---------------------------------------------------------------------------

def compose_synthetic_image(
    background: np.ndarray,
    object_pool: dict[int, list[ObjectPatch]],
    config: SyntheticConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[dict]]:
    """Compose one synthetic image by placing augmented objects on background."""
    canvas = background.copy()
    canvas_h, canvas_w = canvas.shape[:2]

    # Track placed masks for overlap checking
    placed_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    annotations = []

    # Determine how many objects to place
    n_objects = rng.integers(config.min_objects, config.max_objects + 1)

    # Weighted class sampling
    available_cats = [c for c in config.class_weights if c in object_pool and len(object_pool[c]) > 0]
    if not available_cats:
        return canvas, []

    weights = np.array([config.class_weights.get(c, 1.0) for c in available_cats])
    weights /= weights.sum()

    # Pick one class for the entire image
    cat_id = rng.choice(available_cats, p=weights)

    for _ in range(n_objects):
        # Pick a random patch from that class
        patches = object_pool[cat_id]
        patch = patches[rng.integers(len(patches))]

        # Augment
        aug_patch = augment_patch(patch, config, rng)
        if aug_patch is None:
            continue

        ph, pw = aug_patch.mask_crop.shape[:2]

        # Check if patch fits in canvas at all
        if pw > canvas_w - 2 * config.border_margin or ph > canvas_h - 2 * config.border_margin:
            # Try to resize to fit
            max_scale = min(
                (canvas_w - 2 * config.border_margin) / pw,
                (canvas_h - 2 * config.border_margin) / ph,
            )
            if max_scale < 0.3:
                continue
            new_pw = max(1, int(pw * max_scale))
            new_ph = max(1, int(ph * max_scale))
            aug_patch = ObjectPatch(
                image_crop=cv2.resize(aug_patch.image_crop, (new_pw, new_ph), interpolation=cv2.INTER_LINEAR),
                mask_crop=cv2.resize(aug_patch.mask_crop, (new_pw, new_ph), interpolation=cv2.INTER_NEAREST),
                category_id=aug_patch.category_id,
            )
            ph, pw = new_ph, new_pw

        # Try to find a valid placement
        placed = False
        for _ in range(config.max_placement_attempts):
            x = rng.integers(config.border_margin, max(config.border_margin + 1, canvas_w - pw - config.border_margin))
            y = rng.integers(config.border_margin, max(config.border_margin + 1, canvas_h - ph - config.border_margin))

            # Check overlap with already placed objects
            region = placed_mask[y:y+ph, x:x+pw]
            new_mask = aug_patch.mask_crop
            overlap_area = np.sum((region > 0) & (new_mask > 0))
            new_area = np.sum(new_mask > 0)

            if new_area == 0:
                break

            overlap_ratio = overlap_area / new_area
            if overlap_ratio <= config.max_overlap_ratio:
                placed = True
                break

        if not placed:
            continue

        # Alpha-composite onto canvas
        alpha = aug_patch.image_crop[:, :, 3:4].astype(np.float32) / 255.0
        roi = canvas[y:y+ph, x:x+pw].astype(np.float32)
        fg = aug_patch.image_crop[:, :, :3].astype(np.float32)
        blended = fg * alpha + roi * (1.0 - alpha)
        canvas[y:y+ph, x:x+pw] = blended.astype(np.uint8)

        # Update placed mask
        placed_mask[y:y+ph, x:x+pw] = np.maximum(
            placed_mask[y:y+ph, x:x+pw], aug_patch.mask_crop
        )

        # Build full-canvas mask for this object
        full_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        full_mask[y:y+ph, x:x+pw] = aug_patch.mask_crop

        # Convert to COCO annotation
        polys = mask_to_poly(full_mask)
        if not polys:
            continue

        bbox = compute_bbox(full_mask)
        area = float(np.sum(full_mask))

        annotations.append({
            "category_id": aug_patch.category_id,
            "segmentation": polys,
            "bbox": bbox,
            "area": area,
            "iscrowd": 0,
        })

    return canvas, annotations


# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------

def generate(config: SyntheticConfig):
    """Run the full synthetic data generation pipeline."""
    rng = np.random.default_rng(config.seed)

    # Step 1: Load source dataset
    print("Loading source annotations...")
    id2img, id2cat, imgid2anns, categories = load_coco(config.annotation_path)
    print(f"  {len(id2img)} images, {sum(len(v) for v in imgid2anns.values())} annotations")

    # Step 2: Extract all object patches
    print("Extracting object patches...")
    object_pool = extract_all_objects(id2img, imgid2anns, config.image_dir)
    for cat_id, patches in sorted(object_pool.items()):
        name = id2cat.get(cat_id, {}).get("name", f"id={cat_id}")
        print(f"  {name}: {len(patches)} patches")

    # Step 3: Generate background pool
    print(f"Generating {config.num_backgrounds} backgrounds ({config.background_mode})...")
    backgrounds = generate_backgrounds(id2img, imgid2anns, config, rng)
    print(f"  {len(backgrounds)} backgrounds ready")

    # Step 4: Create output directory
    out_img_dir = os.path.join(config.output_dir, "images")
    os.makedirs(out_img_dir, exist_ok=True)

    # Step 5: Generate synthetic images
    print(f"Generating {config.num_images} synthetic images...")
    all_images = []
    all_annotations = []
    ann_id_counter = 1

    for i in tqdm(range(config.num_images), desc="Composing"):
        bg = backgrounds[rng.integers(len(backgrounds))].copy()

        image, annotations = compose_synthetic_image(bg, object_pool, config, rng)

        img_id = i + 1
        filename = f"synthetic_{img_id:06d}.jpg"

        for ann in annotations:
            ann["id"] = ann_id_counter
            ann["image_id"] = img_id
            ann_id_counter += 1
            all_annotations.append(ann)

        all_images.append({
            "id": img_id,
            "file_name": filename,
            "width": config.output_width,
            "height": config.output_height,
        })

        out_path = os.path.join(out_img_dir, filename)
        cv2.imwrite(out_path, image, [cv2.IMWRITE_JPEG_QUALITY, config.jpg_quality])

    # Step 6: Save annotations
    coco_output = {
        "images": all_images,
        "categories": categories,
        "annotations": all_annotations,
    }
    ann_path = os.path.join(config.output_dir, config.output_annotation)
    with open(ann_path, "w") as f:
        json.dump(coco_output, f)

    # Step 7: Print summary
    print("\n=== Generation Summary ===")
    print(f"Images generated: {len(all_images)}")
    print(f"Total annotations: {len(all_annotations)}")
    print(f"Output directory: {config.output_dir}")
    print(f"Annotation file: {ann_path}")

    cat_counts = {}
    for ann in all_annotations:
        cid = ann["category_id"]
        name = id2cat.get(cid, {}).get("name", f"id={cid}")
        cat_counts[name] = cat_counts.get(name, 0) + 1

    total = len(all_annotations) or 1
    print("\nAnnotation count per class:")
    for name, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"  {name:>12s}: {count:5d}  ({count/total*100:.1f}%)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic microplastics training data (patch-based composition)"
    )
    parser.add_argument("--num-images", type=int, default=500,
                        help="Number of synthetic images to generate (default: 500)")
    parser.add_argument("--output-dir", type=str, default="synthetic_output",
                        help="Output directory (default: synthetic_output)")
    parser.add_argument("--annotation", type=str, default="annotation.json",
                        help="Path to source COCO annotation file")
    parser.add_argument("--image-dir", type=str, default="images",
                        help="Path to source images directory")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--min-objects", type=int, default=1,
                        help="Minimum objects per synthetic image")
    parser.add_argument("--max-objects", type=int, default=4,
                        help="Maximum objects per synthetic image")
    parser.add_argument("--background-mode", type=str, default="inpaint",
                        choices=["inpaint", "crop", "noise", "blank"],
                        help="Background generation mode (default: inpaint)")
    parser.add_argument("--max-overlap", type=float, default=0.15,
                        help="Maximum overlap ratio between objects (default: 0.15)")
    parser.add_argument("--fiber-weight", type=float, default=1.0,
                        help="Sampling weight for Fiber class (default: 1.0)")
    parser.add_argument("--fragment-weight", type=float, default=2.5,
                        help="Sampling weight for Fragment class (default: 2.5)")
    parser.add_argument("--film-weight", type=float, default=8.0,
                        help="Sampling weight for Film class (default: 8.0)")
    parser.add_argument("--num-backgrounds", type=int, default=100,
                        help="Number of background images to pre-generate (default: 100)")
    parser.add_argument("--jpg-quality", type=int, default=95,
                        help="JPEG output quality (default: 95)")

    args = parser.parse_args()

    config = SyntheticConfig(
        annotation_path=args.annotation,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        num_images=args.num_images,
        seed=args.seed,
        min_objects=args.min_objects,
        max_objects=args.max_objects,
        background_mode=args.background_mode,
        max_overlap_ratio=args.max_overlap,
        num_backgrounds=args.num_backgrounds,
        jpg_quality=args.jpg_quality,
        class_weights={
            1: args.fiber_weight,
            2: args.fragment_weight,
            3: args.film_weight,
        },
    )

    generate(config)


if __name__ == "__main__":
    main()
