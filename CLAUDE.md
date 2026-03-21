# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI-powered lung nodule malignancy risk estimation system for the [LUNA25 Challenge](https://luna25.grand-challenge.org/). Processes low-dose chest CT scans (`.mha` format) and predicts malignancy probability of detected nodules. The repo contains both the Grand-Challenge submission pipeline and a full-stack web application.

## Commands

### Web App (Docker Compose)
```bash
docker compose build      # Build frontend + backend images
docker compose up         # Start services (frontend :3000, backend :8000)
docker compose up -d      # Start detached
```

### Grand-Challenge Submission
```bash
./do_build.sh             # Build inference Docker image
./do_test_run.sh          # Run inference locally with Docker + GPU
./do_save.sh              # Export .tar.gz for upload
```

### Frontend
```bash
cd frontend
npm install
npm run dev               # Dev server at http://localhost:3000
npm run build
```

### Backend
```bash
cd backend
python main.py            # Uvicorn at http://localhost:8000
```

### Backend Lint/Format
```bash
cd backend
bash scripts/install-deps.sh   # Install ruff, typos, mypy
bash scripts/lint.sh           # ruff check + typos
bash scripts/format.sh         # ruff format + typos
```

### ML Training
```bash
pip install -e "lung-nodule[all]"
python -m lung_nodule.classification.training.train            # Train 2D or 3D baseline
python -m lung_nodule.classification.training.train_pulse_v2   # Train Pulse 3D v2
```

### End-to-End Pipeline
```bash
pip install -e "lung-nodule[detection]"
python run_pipeline.py <path/to/scan.zip> [--device cpu|cuda]
```

> **No automated tests exist yet.** `backend/tests/` is an empty directory. `.github/workflows/lint-and-format.yml` exists but has no content.

## Architecture

### `lung-nodule/` Python Package

All AI logic lives in the installable `lung-nodule/` package (`pip install -e lung-nodule[all]`). It has two modules:

**Detection** (`lung_nodule.detection`): MONAI RetinaNet 3D nodule detector (inference only).
- `DetectionConfig` — dataclass loaded from `detection_config.json`
- `build_detector()`, `build_preprocess()`, `build_postprocess()` — MONAI pipeline builders

**Classification** (`lung_nodule.classification`): Binary malignancy classification (training + inference).
- `MalignancyProcessor` — core inference engine: patch extraction, model loading, prediction
- `NoduleProcessor` — high-level orchestrator: loads CT + coordinates, runs predictions
- `lung_nodule.classification.models` — `ResNet18`, `I3D`, `Pulse3D_v2`
- `lung_nodule.classification.data` — `extract_patch`, `clip_and_scale`, `CTCaseDataset`, `get_data_loader`
- `lung_nodule.classification.training` — training scripts and `Configuration` config class

**Shared** (`lung_nodule._io`):
- `itk_image_to_numpy()` — SimpleITK to numpy conversion (single source of truth)

### Request Flow
```
Browser (React/Vite :3000)
  -> proxy /api/* -> FastAPI (:8000)
      -> predict.py (controller)
      -> predict_service.py (business logic)
      -> predict_repository.py (file/model I/O)
      -> lung_nodule.classification.MalignancyProcessor
          -> lung_nodule.classification.models (PyTorch)
```

### Backend Layer Pattern (`backend/app/`)
- **`api/`** — HTTP routing and request/response handling
- **`service/`** — orchestration and business logic
- **`repository/`** — data access (model loading, image I/O)
- **`schemas/`** — Pydantic models for request/response validation
- **`core/config.py`** — all config via env vars with defaults (`Configuration` class)

### Key API Endpoints
- `POST /api/v1/predict/` — main endpoint: multipart upload of `.mha` CT image + nodule JSON + optional clinical JSON
- `POST /api/v1/predict/lesion` — single lesion with form-field coordinates
- `GET  /api/v1/health/` — health check; `GET /api/v1/health/ping`

### Inference Pipeline
1. CT image uploaded as `.mha`, nodule coordinates as JSON `[x, y, z]`
2. Backend flips axes `[x,y,z]` -> `[z,y,x]` before processing (`np.flip(coords, axis=1)` in `predict_repository.py`)
3. `MalignancyProcessor` extracts 64x64x64 px patches (50mm physical size) around each nodule
4. Runs model, applies sigmoid -> malignancy probability per nodule
5. Returns per-nodule probability + risk classification: Low < 0.4 <= Medium < 0.7 <= High

### Models (`lung_nodule.classification.models`)
- `model_2d.py` — ResNet18 2D baseline
- `model_3d.py` — I3D (Inflated 3D ConvNet) baseline
- `pulse_3d_v2.py` — team's model: ResNet3D-18 backbone + SE attention + Transformer (4-layer, GEGLU, DropPath, LayerScale) + CLS token. Input: `(B, 1, 64, 64, 64)`, output: `(B, 1)` logit.

Weight files are loaded from `{MODEL_PATH}/best_metric_model.pth`. `.pth` files live in `results/`, not in the package.

### Configuration (env vars)
| Variable | Default |
|---|---|
| `MODEL_PATH_2D` | `results/LUNA25-baseline-2D-20250225` |
| `MODEL_PATH_3D` | `results/LUNA25-baseline-3D-20250225` |
| `MODEL_PATH_3D_PULSE` | `results/UET-G8-LUNA25-baseline` |
| `DEFAULT_PREDICTION_MODE` | `2D` |
| `PORT` | `8000` |
| `DEBUG` | `true` |

### Docker Volumes
Docker Compose mounts `./results` from the host into the backend container (`/app/results`). Model architecture code is installed as part of the `lung-nodule` package. Weight files must exist on the host before starting the stack.

### Requirements
- `requirements-ai.txt` — AI/ML core (torch, monai, etc.)
- `requirements-backend.txt` — Web backend (fastapi, uvicorn, etc.)
- `requirements-pipeline.txt` — End-to-end pipeline (AI + dicom2nifti)
- `requirements.txt` — All-in-one convenience file

### Frontend-Backend Proxy
Vite proxies `/api` to the backend. In Docker Compose, `VITE_API_TARGET=http://backend:8000` is used at build time; locally `VITE_API_URL=http://localhost:8000`.

### Grand-Challenge vs Web App
The root-level `Dockerfile` and `inference.py` are for Grand-Challenge submission (batch inference only). Both import from the `lung-nodule` package. The `backend/` and `frontend/` subdirectories are the interactive web application.

### Root-Level Scripts (thin wrappers)
- **`inference.py`** — Grand-Challenge entrypoint. Imports `NoduleProcessor` from `lung_nodule.classification`.
- **`run_pipeline.py`** — End-to-end pipeline (ZIP -> DICOM -> Detection -> Classification). Imports from both `lung_nodule.detection` and `lung_nodule.classification`.
