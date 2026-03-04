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
