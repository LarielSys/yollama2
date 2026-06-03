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
import sqlite3
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import httpx
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Request
from fastapi.responses import FileResponse
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
SAAS_UPLOAD_DIR = OUTPUT_DIR / "saas_uploads"
SAAS_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
SAAS_DB = OUTPUT_DIR / "saas.db"

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


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(SAAS_DB)
    conn.row_factory = sqlite3.Row
    return conn


def init_saas_db() -> None:
    conn = get_db()
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS survey_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                status TEXT NOT NULL,
                confidence REAL NOT NULL,
                total_images INTEGER NOT NULL DEFAULT 0,
                positive_events INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY(project_id) REFERENCES projects(id)
            );

            CREATE TABLE IF NOT EXISTS frames (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                image_name TEXT NOT NULL,
                image_path TEXT NOT NULL,
                image_id TEXT,
                timestamp TEXT,
                lat REAL,
                lon REAL,
                elevation REAL,
                has_deterioration INTEGER NOT NULL,
                confidence REAL NOT NULL,
                detection_count INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(run_id) REFERENCES survey_runs(id)
            );

            CREATE INDEX IF NOT EXISTS idx_frames_run_id ON frames(run_id);
            CREATE INDEX IF NOT EXISTS idx_frames_geo ON frames(lat, lon);
            """
        )
        conn.commit()
    finally:
        conn.close()


@app.on_event("startup")
async def init_saas_on_startup():
    init_saas_db()


def normalize_coord(value: str) -> float | None:
    """Parse coordinates like '19.592650 N' or '98.574850 W'."""
    if not value:
        return None
    parts = value.strip().split()
    if not parts:
        return None
    try:
        coord = float(parts[0])
    except ValueError:
        return None
    if len(parts) > 1 and parts[1].upper() in {"S", "W"}:
        coord *= -1
    return coord


def parse_indice_text(content: str) -> dict[str, dict]:
    """Parse Indice de Imagenes.txt into image_id keyed metadata."""
    records: dict[str, dict] = {}
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        fields = [f.strip() for f in line.split("\t")]
        if len(fields) < 5:
            continue
        image_id = fields[0]
        timestamp = fields[1]
        lat = normalize_coord(fields[2])
        lon = normalize_coord(fields[3])
        elev_raw = fields[4].replace("MSNM", "").strip()
        try:
            elevation = float(elev_raw)
        except ValueError:
            elevation = None
        records[image_id] = {
            "timestamp": timestamp,
            "lat": lat,
            "lon": lon,
            "elevation": elevation,
        }
    return records


def extract_image_id_from_name(name: str) -> str:
    stem = Path(name).stem
    match = re.findall(r"\d{6,}", stem)
    if match:
        return match[-1].zfill(10)
    return stem


async def detect_deterioration(image_name: str, image_bytes: bytes, confidence: float) -> tuple[bool, float, int]:
    """Binary deterioration: positive if any YOLO detection exists above threshold."""
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        response = await client.post(
            f"{YOLO_URL}/detect",
            files={"file": (image_name, image_bytes, "image/jpeg")},
            params={"confidence": confidence},
        )
        response.raise_for_status()
        data = response.json()
        detections = data.get("detections", [])
        if not detections:
            return False, 0.0, 0
        best_conf = max(float(d.get("confidence", 0.0)) for d in detections)
        return True, best_conf, len(detections)


@app.get("/saas", response_class=HTMLResponse)
async def saas_gui(request: Request):
    return templates.TemplateResponse("saas.html", {"request": request})


@app.get("/saas/projects")
async def saas_projects():
    conn = get_db()
    try:
        rows = conn.execute(
            """
            SELECT p.id, p.name, p.created_at,
                   COUNT(r.id) AS run_count
            FROM projects p
            LEFT JOIN survey_runs r ON r.project_id = p.id
            GROUP BY p.id
            ORDER BY p.id DESC
            """
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


@app.get("/saas/runs")
async def saas_runs(project_id: int | None = None):
    conn = get_db()
    try:
        if project_id:
            rows = conn.execute(
                """
                SELECT r.id, r.project_id, p.name AS project_name, r.name, r.status,
                       r.confidence, r.total_images, r.positive_events, r.created_at
                FROM survey_runs r
                JOIN projects p ON p.id = r.project_id
                WHERE r.project_id = ?
                ORDER BY r.id DESC
                """,
                (project_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT r.id, r.project_id, p.name AS project_name, r.name, r.status,
                       r.confidence, r.total_images, r.positive_events, r.created_at
                FROM survey_runs r
                JOIN projects p ON p.id = r.project_id
                ORDER BY r.id DESC
                """
            ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


@app.post("/saas/upload")
async def saas_upload(
    project_name: str = Query(..., min_length=2),
    run_name: str = Query("Survey Run"),
    confidence: float = Query(0.30, ge=0.0, le=1.0),
    package: UploadFile = File(...),
):
    """Upload ZIP package, run binary deterioration detection, store map-ready events."""
    if not package.filename or not package.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Upload must be a .zip package")

    zip_bytes = await package.read()
    if not zip_bytes:
        raise HTTPException(status_code=400, detail="Uploaded package is empty")

    created_at = datetime.now(timezone.utc).isoformat()
    conn = get_db()
    try:
        # Find or create project
        project = conn.execute("SELECT id FROM projects WHERE name = ?", (project_name.strip(),)).fetchone()
        if project:
            project_id = int(project["id"])
        else:
            cur = conn.execute(
                "INSERT INTO projects (name, created_at) VALUES (?, ?)",
                (project_name.strip(), created_at),
            )
            project_id = int(cur.lastrowid)

        run_cur = conn.execute(
            """
            INSERT INTO survey_runs (project_id, name, status, confidence, total_images, positive_events, created_at)
            VALUES (?, ?, 'processing', ?, 0, 0, ?)
            """,
            (project_id, run_name.strip(), confidence, created_at),
        )
        run_id = int(run_cur.lastrowid)
        conn.commit()

        run_dir = SAAS_UPLOAD_DIR / str(run_id)
        images_dir = run_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        # Parse package
        image_files: list[tuple[str, bytes]] = []
        indice_records: dict[str, dict] = {}
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
            for name in zf.namelist():
                if name.endswith("/"):
                    continue
                p = Path(name)
                if p.suffix.lower() in exts:
                    image_files.append((p.name, zf.read(name)))
                elif p.name.lower() == "indice de imagenes.txt":
                    indice_text = zf.read(name).decode("utf-8", errors="ignore")
                    indice_records = parse_indice_text(indice_text)

        if not image_files:
            conn.execute("UPDATE survey_runs SET status = 'error' WHERE id = ?", (run_id,))
            conn.commit()
            raise HTTPException(status_code=400, detail="No images found in zip package")

        total_images = 0
        positive_events = 0
        # Process images sequentially for reliability
        for image_name, image_data in image_files:
            total_images += 1
            safe_name = Path(image_name).name
            frame_path = images_dir / safe_name
            frame_path.write_bytes(image_data)

            image_id = extract_image_id_from_name(safe_name)
            gps = indice_records.get(image_id, {})

            has_damage = False
            best_conf = 0.0
            det_count = 0
            try:
                has_damage, best_conf, det_count = await detect_deterioration(safe_name, image_data, confidence)
            except Exception as e:
                logger.warning("YOLO detection failed for %s in run %s: %s", safe_name, run_id, e)

            if has_damage:
                positive_events += 1

            conn.execute(
                """
                INSERT INTO frames (
                    run_id, image_name, image_path, image_id, timestamp, lat, lon, elevation,
                    has_deterioration, confidence, detection_count, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    safe_name,
                    str(frame_path),
                    image_id,
                    gps.get("timestamp"),
                    gps.get("lat"),
                    gps.get("lon"),
                    gps.get("elevation"),
                    1 if has_damage else 0,
                    best_conf,
                    det_count,
                    created_at,
                ),
            )

        conn.execute(
            """
            UPDATE survey_runs
            SET status = 'completed', total_images = ?, positive_events = ?
            WHERE id = ?
            """,
            (total_images, positive_events, run_id),
        )
        conn.commit()

        return {
            "run_id": run_id,
            "project_id": project_id,
            "status": "completed",
            "total_images": total_images,
            "positive_events": positive_events,
            "message": "Upload processed successfully",
        }
    except HTTPException:
        raise
    except Exception as e:
        conn.execute("UPDATE survey_runs SET status = 'error' WHERE id = (SELECT MAX(id) FROM survey_runs)")
        conn.commit()
        logger.exception("SaaS upload processing failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Upload processing failed: {e}")
    finally:
        conn.close()


@app.get("/saas/runs/{run_id}/map")
async def saas_run_map(run_id: int):
    """Return map points and deterioration events for a specific run."""
    conn = get_db()
    try:
        run = conn.execute(
            "SELECT id, project_id, name, status, total_images, positive_events, created_at FROM survey_runs WHERE id = ?",
            (run_id,),
        ).fetchone()
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")

        frames = conn.execute(
            """
            SELECT id, image_name, image_path, image_id, timestamp, lat, lon, elevation,
                   has_deterioration, confidence, detection_count
            FROM frames
            WHERE run_id = ?
            ORDER BY id ASC
            """,
            (run_id,),
        ).fetchall()

        route_points = []
        events = []
        for fr in frames:
            lat = fr["lat"]
            lon = fr["lon"]
            if lat is not None and lon is not None:
                route_points.append({"lat": lat, "lon": lon, "image_id": fr["image_id"]})
                if fr["has_deterioration"] == 1:
                    events.append(
                        {
                            "frame_id": fr["id"],
                            "image_name": fr["image_name"],
                            "image_id": fr["image_id"],
                            "lat": lat,
                            "lon": lon,
                            "timestamp": fr["timestamp"],
                            "confidence": fr["confidence"],
                            "detection_count": fr["detection_count"],
                            "image_url": f"/saas/runs/{run_id}/image/{fr['image_name']}",
                        }
                    )

        return {
            "run": dict(run),
            "route_points": route_points,
            "events": events,
        }
    finally:
        conn.close()


@app.get("/saas/runs/{run_id}/image/{image_name}")
async def saas_run_image(run_id: int, image_name: str):
    safe_name = Path(image_name).name
    image_path = SAAS_UPLOAD_DIR / str(run_id) / "images" / safe_name
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path=image_path)


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
