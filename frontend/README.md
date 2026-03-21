# Frontend — LUNA25 Malignancy Prediction UI

React web interface for uploading chest CT scans and visualizing lung nodule malignancy predictions.

## Stack

- **React 18** | **Vite 7** | **Axios**
- Proxies `/api` to the FastAPI backend (port 8000)

## Quick Start

```bash
npm install
npm run dev        # → http://localhost:3000
npm run build      # Production build
npm run preview    # Preview production build locally
```

## Docker

```bash
# Standalone
docker build -t luna25-frontend .
docker run -p 3000:3000 luna25-frontend

# With full stack (recommended)
# From repo root:
docker compose up
```

In Docker Compose, the frontend is built with `VITE_API_TARGET=http://backend:8000` so `/api` requests are proxied to the backend container.

## Features

- Upload chest CT images (`.mha` format)
- Upload nodule locations (`.json`)
- Upload optional clinical information (`.json`)
- View per-nodule malignancy probability with risk level badge
- Risk levels: **Low** (< 0.4) | **Medium** (0.4–0.7) | **High** (≥ 0.7)

## Project Structure

```
frontend/
├── index.html
├── vite.config.js      # Dev server config + /api proxy
├── package.json
└── src/
    ├── main.jsx        # React entry point
    ├── App.jsx         # Main component (file upload + results display)
    └── index.css       # Global styles
```

## API Integration

The frontend calls `POST /api/v1/predict/` as a `multipart/form-data` request.

**Required fields:**
- `image_data` — `.mha` CT scan file
- `nodule_locations` — JSON file with nodule coordinates

**Nodule locations format:**
```json
{
  "points": [
    { "name": "nodule_1", "point": [x, y, z] }
  ]
}
```

**Optional field:**
- `clinical_information` — JSON file with patient metadata:

```json
{ "age": 65, "gender": "M" }
```

**Response shape:**
```json
{
  "name": "Points of interest",
  "type": "Multiple points",
  "points": [
    { "name": "nodule_1", "point": [x, y, z], "probability": 0.82 }
  ]
}
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VITE_API_TARGET` | `http://localhost:8000` | Backend URL for `/api` proxy |

Set in `.env` locally or as a build arg in Docker Compose.
