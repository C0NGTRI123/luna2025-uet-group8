# Backend — LUNA25 Malignancy Prediction API

FastAPI backend for the LUNA25 lung nodule malignancy estimation web application.

## Stack

- **Python 3.9** | **FastAPI 0.121** | **Uvicorn 0.38**
- **PyTorch 2.0** + **SimpleITK** for inference
- [`lung-nodule`](../lung-nodule/) installable package for all AI logic

## Quick Start

```bash
# Install dependencies
pip install -r ../requirements-backend.txt
pip install -e "../lung-nodule[all]"

# Run development server (hot-reload)
python main.py
# → http://localhost:8000
# → Swagger UI: http://localhost:8000/docs
```

## Docker

```bash
# From repo root
docker compose build backend
docker compose up backend
```

The container mounts `../results/` to `/app/results` for model weight files.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/health/` | Service health check |
| `GET` | `/api/v1/health/ping` | Liveness probe → `{"message":"pong"}` |
| `POST` | `/api/v1/predict/` | Predict malignancy for a CT scan |
| `POST` | `/api/v1/predict/lesion` | Predict a single lesion via form fields |
| `GET` | `/api/v1/predict/status` | Model/service status |

### `POST /api/v1/predict/` — Multipart Upload

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `image_data` | file (`.mha`) | Yes | Low-dose chest CT scan |
| `nodule_locations` | file (`.json`) | Yes | Nodule coordinates |
| `clinical_information` | file (`.json`) | No | Patient metadata |

**Nodule locations JSON format:**
```json
{
  "points": [
    { "name": "nodule_1", "point": [x, y, z] }
  ]
}
```

**Response:**
```json
{
  "name": "Points of interest",
  "type": "Multiple points",
  "points": [
    { "name": "nodule_1", "point": [x, y, z], "probability": 0.82 }
  ],
  "version": { "major": 1, "minor": 0 }
}
```

### `POST /api/v1/predict/lesion` — Form Fields

| Field | Type | Required |
|-------|------|----------|
| `seriesInstanceUID` | string | Yes |
| `lesionID` | string | Yes |
| `coordX`, `coordY`, `coordZ` | float | Yes |
| `patientID`, `studyDate`, `age`, `gender` | string/int | No |

**Response:**
```json
{
  "status": "success",
  "data": {
    "probability": 0.82,
    "predictionLabel": 1,
    "processingTimeMs": 1240
  }
}
```

### Risk Classification

| Range | Level |
|-------|-------|
| probability < 0.4 | Low |
| 0.4 ≤ probability < 0.7 | Medium |
| probability ≥ 0.7 | High |

## Project Structure

```
backend/
├── main.py                     # Uvicorn entrypoint
├── app/
│   ├── app.py                  # FastAPI app factory (CORS, routers)
│   ├── api/
│   │   ├── health/v1/          # Health endpoints
│   │   └── predict/v1/         # Prediction endpoints
│   ├── service/
│   │   └── predict_service.py  # Business logic orchestration
│   ├── repository/
│   │   └── predict_repository.py  # Image I/O, coord transform, model paths
│   ├── schemas/
│   │   └── predict.py          # Pydantic request/response models
│   └── core/
│       ├── config.py           # Configuration (env vars)
│       └── exceptions.py       # Custom HTTP exceptions
└── scripts/
    ├── install-deps.sh         # Install ruff, typos, mypy
    ├── lint.sh                 # ruff check + typos
    └── format.sh               # ruff format + typos
```

## Configuration

All settings are environment variables with defaults defined in `app/core/config.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `ENV` | `development` | Environment name |
| `DEBUG` | `true` | Enable hot-reload and debug logs |
| `HOST` | `0.0.0.0` | Bind host |
| `PORT` | `8000` | Bind port |
| `LOG_LEVEL` | `INFO` | Uvicorn log level |
| `MODEL_PATH_2D` | `results/LUNA25-baseline-2D-20250225` | 2D model weights directory |
| `MODEL_PATH_3D` | `results/LUNA25-baseline-3D-20250225` | 3D model weights directory |
| `MODEL_PATH_3D_PULSE` | `results/UET-G8-LUNA25-baseline` | Pulse 3D v2 weights directory |
| `DEFAULT_PREDICTION_MODE` | `2D` | Fallback mode if not specified |

Weight files are loaded from `{MODEL_PATH}/best_metric_model.pth`.

## Inference Flow

1. Client uploads `.mha` CT image + nodule coordinates JSON.
2. `predict_repository.py` saves files, flips axes `[x,y,z]` → `[z,y,x]` (`np.flip(coords, axis=1)`).
3. `predict_service.py` calls `MalignancyProcessor` from the `lung-nodule` package.
4. `MalignancyProcessor` extracts 64×64×64 px patches (50 mm physical) around each nodule, runs the model, and applies sigmoid.
5. Response returns per-nodule probability and risk label.

The web app always uses the **Pulse 3D v2** model (`3D-PULSE` mode).

## Linting and Formatting

```bash
cd backend
bash scripts/install-deps.sh   # one-time setup
bash scripts/lint.sh           # ruff check + typos
bash scripts/format.sh         # ruff format + typos fix
```

## Error Codes

| Status | Exception | Cause |
|--------|-----------|-------|
| 400 | `InvalidFileFormatException` | Unsupported file type |
| 404 | `NotFoundException` | Resource not found |
| 422 | `ProcessingErrorException` | Model/GPU inference failure |
| 500 | `InternalServerErrorException` | Unexpected server error |
| 504 | `GatewayTimeoutException` | Processing exceeded 600 s |
