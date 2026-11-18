# YOLO + SAM Backend for Label Studio

ML backend that combines YOLO object detection with SAM (Segment Anything Model)
for interactive polygon annotation in Label Studio.

## Features

- **Auto-detection**: YOLO detects objects, SAM generates precise polygon masks
- **Interactive annotation**: Click a point or draw a box, SAM segments the object
- **Smart tools**: Supports both KeyPointLabels and RectangleLabels triggers

## Setup

1. Copy `.env.example` to `.env` and adjust paths:
   ```bash
   cp .env.example .env
   ```

2. Install dependencies:
   ```bash
   pip install .
   ```

3. Place model weights in the parent directory:
   - `../v5.1.pt` (YOLO checkpoint)
   - `../sam_vit_b_01ec64.pth` (SAM checkpoint)

4. Run the backend:
   ```bash
   python _wsgi.py
   ```

5. Connect to Label Studio at **Settings > Machine Learning > Add Model**
   - URL: `http://localhost:9090`

## Configuration

All settings are controlled via environment variables or `.env` file.
See `.env.example` for all available options.

| Variable | Default | Description |
|----------|---------|-------------|
| `YOLO_CHECKPOINT` | `../v5.1.pt` | Path to YOLO model weights |
| `SAM_CHECKPOINT` | `../sam_vit_b_01ec64.pth` | Path to SAM model weights |
| `SAM_MODEL_TYPE` | `vit_b` | SAM model architecture |
| `YOLO_CONFIDENCE_THRESHOLD` | `0.25` | YOLO detection confidence |
| `POLYGON_SIMPLIFY_EPSILON` | `0.001` | Polygon simplification factor |
| `ML_BACKEND_HOST` | `0.0.0.0` | Server bind host |
| `ML_BACKEND_PORT` | `9090` | Server bind port |
| `LOG_LEVEL` | `INFO` | Logging level |
| `DEBUG_REQUESTS` | `false` | Log all incoming requests |

## Label Config

Use `label_config.xml` in your Label Studio project for compatible annotation setup.
