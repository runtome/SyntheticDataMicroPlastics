# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Synthetic data generation for microplastics image segmentation (U-Net / Attention U-Net). Uses patch-based composition to extract real microplastic objects and composite them onto clean backgrounds, producing new training images with COCO-format annotations.

## Dataset

- 596 source images (640x480, JPG) in `images/`
- COCO-format annotations in `annotation.json` (901 annotations, 3 classes)
- Classes: Fiber (69.4%), Fragment (25.5%), Film (5.1%) — severely imbalanced

## Commands

```bash
pip install -r requirements.txt
python generate_synthetic.py --num-images 500
python generate_synthetic.py --num-images 1000 --film-weight 12.0 --background-mode noise
```

Key CLI flags: `--num-images`, `--seed`, `--background-mode` (inpaint|crop|noise|blank), `--fiber-weight`, `--fragment-weight`, `--film-weight`, `--min-objects`, `--max-objects`

## Architecture

`generate_synthetic.py` — single self-contained script:
1. **COCO loading** — parses annotation.json, builds lookup dicts
2. **Object extraction** — converts polygon masks to RGBA patches with soft alpha edges
3. **Background generation** — inpaints objects out of real images (default), or noise/blank
4. **Augmentation** — rotation, scale, flip, brightness, blur per object patch
5. **Composition** — places augmented objects onto backgrounds with overlap control
6. **Output** — saves images + COCO JSON to `synthetic_output/`

Class weights default to Fiber=1.0, Fragment=2.5, Film=8.0 to invert the imbalance.

## Diffusion Model (Method 1)

Class-label-conditioned DDPM that generates individual object crops. Complements the patch-based composition approach above.

### Training

```bash
python train_diffusion.py --num-epochs 300 --batch-size 16           # GPU
python train_diffusion.py --num-epochs 50 --batch-size 4             # CPU test
python train_diffusion.py --resume diffusion_checkpoints/checkpoint-100
```

Key flags: `--image-size` (default 128), `--num-epochs`, `--batch-size`, `--lr`, `--save-every`, `--resume`, `--fp16`

### Inference

```bash
# Individual crops
python generate_diffusion.py --checkpoint diffusion_checkpoints/checkpoint-300 --num-per-class 50

# Full composed scenes + COCO JSON
python generate_diffusion.py --checkpoint diffusion_checkpoints/checkpoint-300 \
    --compose --num-images 200 --background-mode inpaint
```

Key flags: `--checkpoint`, `--num-per-class`, `--classes`, `--compose`, `--num-images`, `--background-mode`

### Architecture

- `train_diffusion.py` — DDPM training with `UNet2DModel` (diffusers), cosine schedule, class-label embeddings, minority class oversampling
- `generate_diffusion.py` — reverse diffusion inference with two modes: crop-only or full scene composition (reuses `generate_synthetic.py` background/composition pipeline)
