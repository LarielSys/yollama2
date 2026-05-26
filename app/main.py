"""
YOLLama Orchestrator
Receives images, sends to YOLO for detection, then to Ollama for interpretation.
"""

import base64
import io
import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="YOLLama Orchestrator", version="1.0.0")

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


@app.get("/", response_class=HTMLResponse)
async def gui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


YOLO_URL = os.getenv("YOLO_SERVICE_URL", "http://yolo:8001")
OLLAMA_URL = os.getenv("OLLAMA_SERVICE_URL", "http://ollama:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5vl")
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/app/output"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TIMEOUT = httpx.Timeout(120.0, connect=30.0)


@app.on_event("startup")
async def preload_ollama():
    """Preload the Ollama model so the first user request is fast."""
    logger.info(f"Preloading Ollama model '{OLLAMA_MODEL}'...")
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        try:
            r = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": OLLAMA_MODEL, "prompt": "hi", "stream": False},
            )
            logger.info(f"Ollama model preloaded (status {r.status_code})")
        except Exception as e:
            logger.warning(f"Ollama preload failed (will load on first request): {e}")


def build_prompt(detections: list[dict], filename: str) -> str:
    """Build the analysis prompt for Ollama including YOLO detections."""
    detection_text = "\n".join(
        f"- {d['class_name']} — {d.get('class_description', d['class_name'])} "
        f"(confidence: {d['confidence']:.2f}) "
        f"at bbox [{d['bbox']['x1']}, {d['bbox']['y1']}, {d['bbox']['x2']}, {d['bbox']['y2']}]"
        for d in detections
    )
    if not detection_text:
        detection_text = "No objects detected by YOLO."

    return f"""You are a road infrastructure analyst. A YOLO road damage detection model (trained on RDD2022) has analyzed a road image.

The model detects these damage classes:
- D00: Longitudinal and transverse cracks
- D10: Reflection cracks from concrete joints
- D20: Fatigue/alligator cracking
- D40: Potholes and patching

Image: {filename}

YOLO Detections:
{detection_text}

Based on the image and these detections, provide a road condition assessment.
You MUST respond with ONLY valid JSON in exactly this format (no additional text):

{{
  "summary": "Brief description of the road condition observed",
  "objects": [
    {{
      "name": "damage type description",
      "confidence": 0.00,
      "severity": "none|low|medium|high|critical"
    }}
  ],
  "recommendation": "Maintenance actions recommended"
}}

Severity guidelines:
- D00 cracks: low if hairline, medium if wide, high if networked
- D10 reflection cracks: medium to high depending on extent
- D20 fatigue cracking: high to critical (indicates structural failure)
- D40 potholes: high to critical (safety hazard)
- If no detections, assess road condition from the image alone."""


def extract_json(text: str) -> dict | None:
    """Extract JSON from Ollama response, handling markdown code blocks."""
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try extracting from code block
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # Try finding first { ... } block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


@app.get("/health")
async def health():
    """Check health of all services."""
    status = {"app": "ok", "yolo": "unknown", "ollama": "unknown"}
    async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
        try:
            r = await client.get(f"{YOLO_URL}/health")
            status["yolo"] = "ok" if r.status_code == 200 else "error"
        except Exception:
            status["yolo"] = "unreachable"
        try:
            r = await client.get(f"{OLLAMA_URL}/api/tags")
            status["ollama"] = "ok" if r.status_code == 200 else "error"
        except Exception:
            status["ollama"] = "unreachable"
    return status


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    confidence: float = Query(0.25, ge=0.0, le=1.0),
    save: bool = Query(True, description="Save results to output directory"),
):
    """
    Full analysis pipeline:
    1. Send image to YOLO for detection
    2. Send image + detections to Ollama for interpretation
    3. Return and optionally save structured results
    """
    # Read image bytes
    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    import time as _time
    run_id = uuid.uuid4().hex[:12]
    logger.info(f"[{run_id}] Analyzing: {file.filename}")
    t_start = _time.time()

    # --- Step 1: YOLO Detection ---
    logger.info(f"[{run_id}] Sending to YOLO service...")
    t_yolo = _time.time()
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        try:
            yolo_response = await client.post(
                f"{YOLO_URL}/detect",
                files={"file": (file.filename, image_bytes, file.content_type or "image/jpeg")},
                params={"confidence": confidence},
            )
            yolo_response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=502, detail=f"YOLO service error: {e.response.text}")
        except httpx.ConnectError:
            raise HTTPException(status_code=503, detail="YOLO service unreachable")

    yolo_data = yolo_response.json()
    detections = yolo_data.get("detections", [])
    logger.info(f"[{run_id}] YOLO found {len(detections)} objects in {_time.time()-t_yolo:.2f}s")

    # --- Step 2: Ollama Interpretation ---
    logger.info(f"[{run_id}] Sending to Ollama ({OLLAMA_MODEL})...")
    prompt = build_prompt(detections, file.filename or "image")

    # Re-encode image: resize for faster Ollama processing + fix JPEG compat
    from PIL import Image
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        # Downscale to max 768px on longest side for faster Ollama inference
        max_dim = 768
        if max(img.size) > max_dim:
            img.thumbnail((max_dim, max_dim), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    ollama_payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
    }

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        try:
            ollama_response = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json=ollama_payload,
            )
            ollama_response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=502, detail=f"Ollama service error: {e.response.text}")
        except httpx.ConnectError:
            raise HTTPException(status_code=503, detail="Ollama service unreachable")

    t_ollama_end = _time.time()
    ollama_data = ollama_response.json()
    raw_response = ollama_data.get("response", "")
    logger.info(f"[{run_id}] Ollama responded ({len(raw_response)} chars) in {t_ollama_end-t_yolo:.2f}s total")

    # --- Step 3: Parse & structure output ---
    structured = extract_json(raw_response)
    if structured is None:
        structured = {
            "summary": raw_response[:500],
            "objects": [
                {"name": d["class_name"], "confidence": d["confidence"], "severity": "unknown"}
                for d in detections
            ],
            "recommendation": "Ollama response could not be parsed as JSON. Raw response preserved in summary.",
        }

    result = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "filename": file.filename,
        "yolo": yolo_data,
        "analysis": structured,
        "raw_ollama_response": raw_response,
    }

    # --- Step 4: Save results ---
    if save:
        # Strip large base64 image from saved JSON to keep files small
        save_data = {k: v for k, v in result.items()}
        if "yolo" in save_data and save_data["yolo"].get("annotated_image"):
            save_data["yolo"] = {k: v for k, v in save_data["yolo"].items() if k != "annotated_image"}
        stem = Path(file.filename).stem if file.filename else run_id
        out_path = OUTPUT_DIR / f"{stem}_{run_id}.json"
        out_path.write_text(json.dumps(save_data, indent=2, ensure_ascii=False), encoding="utf-8")
        result["saved_to"] = str(out_path)
        logger.info(f"[{run_id}] Results saved to {out_path}")

    result["timing"] = {"total_seconds": round(_time.time() - t_start, 2)}
    logger.info(f"[{run_id}] Total pipeline: {_time.time()-t_start:.2f}s")
    return JSONResponse(content=result)


@app.get("/results")
async def list_results():
    """List all saved analysis results."""
    files = sorted(OUTPUT_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    return [{"filename": f.name, "size": f.stat().st_size} for f in files]


@app.get("/results/{filename}")
async def get_result(filename: str):
    """Retrieve a specific saved result."""
    # Prevent path traversal
    safe_name = Path(filename).name
    filepath = OUTPUT_DIR / safe_name
    if not filepath.exists() or not filepath.suffix == ".json":
        raise HTTPException(status_code=404, detail="Result not found")
    return JSONResponse(content=json.loads(filepath.read_text(encoding="utf-8")))
