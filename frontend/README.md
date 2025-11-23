# LUNA25 Baseline - Frontend

React-based web interface for LUNA25 nodule malignancy prediction.

## Features

- Upload CT images (.mha format)
- Upload nodule locations (JSON)
- Optional clinical information upload
- Choose between 2D and 3D prediction models
- View malignancy predictions with risk levels
- Clean and intuitive UI

## Development

### Install dependencies
```bash
npm install
```

### Run development server
```bash
npm run dev
```

Visit http://localhost:3000

### Build for production
```bash
npm run build
```

## Docker

Build and run with Docker:
```bash
docker build -t luna25-frontend .
docker run -p 3000:3000 luna25-frontend
```

## API Integration

The frontend connects to the backend API at `/api/v1/predict/` endpoint.

Expected JSON format for nodule locations:
```json
{
  "points": [
    {
      "name": "nodule_1",
      "point": [x, y, z]
    }
  ]
}
```

Expected JSON format for clinical information:
```json
{
  "age": 65,
  "gender": "M"
}
```
