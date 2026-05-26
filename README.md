# YOLLama — YOLO + Ollama Docker Architecture

Local AI system: **YOLO** detects objects, **Ollama** interprets and generates structured reports.

## Architecture

```
┌──────────┐     ┌──────────────┐     ┌──────────────┐
│  Client   │────▶│  Orchestrator │────▶│  YOLO Service │
│ CLI / API │     │  (FastAPI)    │     │  (ultralytics)│
└──────────┘     │  :8000        │     │  :8001        │
                  │               │     └──────────────┘
                  │               │────▶┌──────────────┐
                  │               │     │  Ollama       │
                  └──────────────┘     │  :11434       │
                                        └──────────────┘
```

| Service       | Port  | Role                            |
|---------------|-------|---------------------------------|
| **app**       | 8000  | Orchestrator (FastAPI)          |
| **yolo**      | 8001  | Object detection (YOLOv8)       |
| **ollama**    | 11434 | Vision-language model            |

## Quick Start

### 1. Start the system

```bash
docker-compose up --build -d
```

### 2. Pull an Ollama model (first time only)

```bash
docker exec yollama-ollama ollama pull qwen2.5vl
```

> Swap `qwen2.5vl` with any vision-capable model. Update `OLLAMA_MODEL` in `docker-compose.yml` to match.

### 3. Analyze an image

**CLI:**
```bash
pip install httpx
python cli.py path/to/image.jpg
```

**curl:**
```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@path/to/image.jpg" \
  -o result.json
```

**API Docs:** Open http://localhost:8000/docs for interactive Swagger UI.

### 4. Check service health

```bash
curl http://localhost:8000/health
```

## Pipeline Flow

1. Image submitted to `/analyze`
2. Image forwarded to YOLO `/detect` → bounding boxes, classes, confidence
3. Detections + image sent to Ollama → structured JSON analysis
4. Combined result saved to `data/output/` and returned

## Output Format

```json
{
  "run_id": "a1b2c3d4e5f6",
  "timestamp": "2026-03-31T12:00:00+00:00",
  "filename": "road_sample.jpg",
  "yolo": {
    "detections_count": 3,
    "detections": [...]
  },
  "analysis": {
    "summary": "Image shows a road with visible cracks and a pothole",
    "objects": [
      { "name": "crack", "confidence": 0.91, "severity": "medium" }
    ],
    "recommendation": "Schedule maintenance inspection"
  }
}
```

## Configuration

Environment variables in `docker-compose.yml`:

| Variable            | Default                  | Description                |
|---------------------|--------------------------|----------------------------|
| `YOLO_SERVICE_URL`  | `http://yolo:8001`       | YOLO service endpoint      |
| `OLLAMA_SERVICE_URL`| `http://ollama:11434`    | Ollama service endpoint    |
| `OLLAMA_MODEL`      | `qwen2.5vl`             | Ollama model to use        |

## Project Structure

```
YOLLama/
├── docker-compose.yml          # Service orchestration
├── cli.py                      # Command-line client
├── app/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── main.py                 # Orchestrator (FastAPI)
├── yolo-service/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app.py                  # YOLO detection API
└── data/
    ├── input/                  # Place images here
    └── output/                 # Analysis results
```

## API Endpoints

| Method | Endpoint              | Description                    |
|--------|-----------------------|--------------------------------|
| POST   | `/analyze`            | Full pipeline analysis         |
| GET    | `/health`             | Service health check           |
| GET    | `/results`            | List saved results             |
| GET    | `/results/{filename}` | Retrieve specific result       |
| POST   | `/detect` (yolo:8001) | Direct YOLO detection          |

## Future Extensions

- Batch image processing
- Video stream analysis
- GUI dashboard
- Database persistence
- PDF report export
- Camera feed integration
