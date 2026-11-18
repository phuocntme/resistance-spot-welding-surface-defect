# Resistance Spot Welding Surface Defect Detection

YOLO-based surface-defect detection with a Label Studio **YOLO + SAM** backend for assisted polygon annotation. YOLO proposes defect regions; Segment Anything refines them from detection boxes or interactive point/rectangle prompts, then returns editable polygons to Label Studio.

## Project contents

- `training/YOLO-Segmentation-Training.ipynb` — sanitized YOLO segmentation workflow. Saved cell outputs and local/company-specific paths were removed. Configure an authorized dataset before running.
- `training/train_val_split.py` — image/label train-validation split helper.
- `backend/my_backend/` — Label Studio ML backend, settings, label configuration, and packaging metadata.
- `backend/v5.1.pt` — YOLO detection checkpoint used by the Label Studio backend. The backend loads it from this relative path by default.
- `models/v5.1.pt` — duplicate copy of the same checkpoint for local training/evaluation workflows.
- `data.example.yaml` — placeholder dataset configuration; no dataset is included.

The SAM checkpoint is not bundled. Download the official Meta SAM ViT-B checkpoint (about 375 MB) from [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything#model-checkpoints): [sam_vit_b_01ec64.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth). Place it at `backend/sam_vit_b_01ec64.pth`; keep `SAM_MODEL_TYPE=vit_b` in the backend environment.

The project owner confirms permission to distribute the included `v5.1.pt` checkpoint. The two copies are identical; keep the required paths for the backend and training workflow. Verify any applicable upstream model terms independently.

## Dataset setup

Use a dataset you are authorized to access and use. No dataset files are included in this folder. Expected YOLO layout:

```text
dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

Copy `data.example.yaml` to `data.yaml`, set the dataset root, and replace placeholder class names and IDs with the dataset's actual label schema:

```powershell
Copy-Item data.example.yaml data.yaml
```

`data.yaml`, dataset folders, training outputs, virtual environments, and real `.env` files are excluded by `.gitignore`.

## Training

Python 3.12 is recommended. Create an environment and install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install a PyTorch build compatible with the local OS, GPU driver, and CUDA runtime using the official PyTorch selector. Then open `training/YOLO-Segmentation-Training.ipynb` and run cells after setting the dataset path and class names. Example CLI command:

```bash
yolo segment train model=yolo26m-seg.pt data=data.yaml epochs=150 imgsz=1024 batch=8 device=0
```

Training outputs go under `runs/`. No accuracy claims are made here: verify dataset/split provenance, class mapping, checkpoint lineage, and evaluation protocol before reporting metrics.

## Label Studio YOLO + SAM backend

Install the package and configure local model paths in a private `.env` file:

```powershell
Copy-Item backend/my_backend/.env.example backend/my_backend/.env
cd backend/my_backend
pip install .
python _wsgi.py
```

The backend defaults to port `9090`. Connect it in Label Studio under **Settings → Machine Learning → Add Model**. `label_config.xml` defines the expected labels; keep those names consistent with the model class names and dataset schema. Use authentication and network controls before exposing the backend beyond a trusted local network.

## Reproducibility and responsible sharing

For future runs, record dataset provenance and split, class mapping, model source/checksum, package/CUDA versions, training parameters, and validation metrics. Do not publish confidential employer data, production images/labels, credentials, or model files without confirming redistribution permission.

## License

No project license has been selected. Check each dependency, pretrained checkpoint, and future dataset license independently before redistribution or commercial use; no license file is included.
