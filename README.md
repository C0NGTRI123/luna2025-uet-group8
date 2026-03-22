# LUNA25 — UET Group 8

AI-powered lung nodule malignancy risk estimation for the [LUNA25 Challenge](https://luna25.grand-challenge.org/). Processes low-dose chest CT scans (`.mha` format) and predicts the malignancy probability of detected nodules.

This repository contains:
- **`lung-nodule/`** — installable Python package with all AI logic (detection + classification)
- **`backend/`** — FastAPI REST API
- **`frontend/`** — React/Vite web interface
- **`inference.py`** — Grand-Challenge batch inference entrypoint
- **`run_pipeline.py`** — end-to-end pipeline (ZIP → DICOM → detection → classification)

---

## Web Application (Docker Compose)

The easiest way to run the full stack:

```bash
docker compose build      # Build backend + frontend images
docker compose up         # Start services
docker compose up -d      # Start detached
```

| Service | URL |
|---------|-----|
| Frontend | http://localhost:3000 |
| Backend API | http://localhost:8000 |
| Swagger UI | http://localhost:8000/docs |

> Model weight files must exist under `./results/` before starting. Docker Compose mounts this directory into the backend container.

### Running Without Docker

**Backend:**
```bash
pip install -r requirements-backend.txt
pip install -e "lung-nodule[all]"
cd backend && python main.py
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev   # http://localhost:3000
```

---

## Environment Setup (ML Training / Pipeline)

```bash
conda create -n luna25 python=3.9
conda activate luna25

# AI + training dependencies
pip install -r requirements-ai.txt
pip install -e "lung-nodule[all]"
```

---

## ML Training

All training code lives in the `lung-nodule` package.

```bash
# Train 2D (ResNet18) or 3D (I3D) baseline
python -m lung_nodule.classification.training.train

# Train the team's Pulse 3D v2 model
python -m lung_nodule.classification.training.train_pulse_v2
```

Key training hyperparameters are in `lung-nodule/src/lung_nodule/classification/training/config.py`:

| Parameter | Value |
|-----------|-------|
| Patch size | 64 px / 50 mm |
| Batch size | 32 |
| Learning rate | 1e-4 |
| Epochs | 100 |
| Early stopping | patience=10 |
| Seed | 2025 |

Trained weights are saved to `results/{experiment_name}/best_metric_model.pth`.

---

## End-to-End Pipeline

Runs the full pipeline from a zipped DICOM study to malignancy predictions:

```bash
pip install -e "lung-nodule[detection]"
python run_pipeline.py <path/to/scan.zip> [--device cpu|cuda]
```

---

## Grand-Challenge Submission

```bash
./do_build.sh       # Build inference Docker image (linux/amd64)
./do_test_run.sh    # Test inference locally with Docker + GPU
./do_save.sh        # Export .tar.gz for upload to Grand-Challenge
```

The `inference.py` entrypoint reads from `test/input/` and writes predictions to `test/output/lung-nodule-malginancy-likelihoods.json`. It uses the **Pulse 3D v2** model by default (`results/UET-G8-LUNA25-baseline/`).

---

## Architecture

```
Browser (React/Vite :3000)
  → proxy /api/* → FastAPI (:8000)
      → predict_service.py
      → predict_repository.py
      → lung_nodule.classification.MalignancyProcessor
          → Pulse3D_v2 / I3D / ResNet18 (PyTorch)
```

### Models

| Mode | Model | Description |
|------|-------|-------------|
| `2D` | `ResNet18` | 2D ResNet18 baseline |
| `3D` | `I3D` | Inflated 3D ConvNet baseline |
| `3D-PULSE` | `Pulse3D_v2` | Team model — ResNet3D-18 + SE attention + 4-layer Transformer (GEGLU, DropPath, LayerScale) |

### Risk Classification

| Probability | Risk Level |
|-------------|-----------|
| < 0.4 | Low |
| 0.4 – 0.7 | Medium |
| ≥ 0.7 | High |

### Evaluation Metrics

The main evaluation metric used during training is **ROC-AUC** on the validation set.

- **Primary model-selection metric:** Validation ROC-AUC (higher is better)
- **Optimization loss:** Binary Cross-Entropy with Logits (`BCEWithLogitsLoss`)
- **Learning-rate scheduling signal:** Validation ROC-AUC (`ReduceLROnPlateau`, mode=`max`)
- **Early stopping:** Based on patience when validation ROC-AUC no longer improves

In `train_pulse_v2.py`, the best checkpoint is saved whenever validation ROC-AUC improves.

---

## Repository Structure

```
luna2025-uet-group8/
├── lung-nodule/              # Installable AI package
│   └── src/lung_nodule/
│       ├── classification/   # MalignancyProcessor, NoduleProcessor, models, training
│       └── detection/        # MONAI RetinaNet 3D detector
├── backend/                  # FastAPI application
│   └── app/
│       ├── api/              # HTTP routing
│       ├── service/          # Business logic
│       ├── repository/       # Model I/O
│       └── core/             # Config, exceptions
├── frontend/                 # React + Vite UI
│   └── src/
├── results/                  # Model weight directories (not committed)
├── inference.py              # Grand-Challenge entrypoint
├── run_pipeline.py           # End-to-end pipeline
├── docker-compose.yml        # Web app orchestration
├── Dockerfile                # Grand-Challenge submission image
├── requirements-ai.txt       # AI/ML dependencies
├── requirements-backend.txt  # Web backend dependencies
├── requirements-pipeline.txt # Pipeline dependencies
└── requirements.txt          # All-in-one convenience file
```

---

## Configuration

Backend configuration is done via environment variables (see `backend/app/core/config.py`):

| Variable | Default |
|----------|---------|
| `MODEL_PATH_2D` | `results/LUNA25-baseline-2D-20250225` |
| `MODEL_PATH_3D` | `results/LUNA25-baseline-3D-20250225` |
| `MODEL_PATH_3D_PULSE` | `results/UET-G8-LUNA25-baseline` |
| `DEFAULT_PREDICTION_MODE` | `2D` |
| `PORT` | `8000` |
| `DEBUG` | `true` |

---

## Further Reading

- [`lung-nodule/README.md`](lung-nodule/README.md) — AI package usage, model architectures, training
- [`backend/README.md`](backend/README.md) — API endpoints, request/response format, inference flow
- [`frontend/README.md`](frontend/README.md) — UI setup, API integration
- [LUNA25 Challenge](https://luna25.grand-challenge.org/)
- [Grand-Challenge Docker docs](https://grand-challenge.org/documentation/test-and-deploy-your-container/)
