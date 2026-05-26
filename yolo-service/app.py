"""
YOLO Detection Service — Road Damage Detection
Uses YOLOv11n-OBB model trained on RDD2022 dataset.
Classes: D00 (longitudinal/transverse cracks), D10 (reflection cracks),
         D20 (fatigue/alligator cracking), D40 (potholes/patching)
"""

import base64
import io
import logging
import os
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from PIL import Image
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="YOLO Road Damage Detection Service")

MODELS_DIR = Path("/app/models")
DEFAULT_MODEL_NAME = os.getenv("YOLO_MODEL", "road_damage.pt")
loaded_models: dict[str, YOLO] = {}
active_model_name = DEFAULT_MODEL_NAME

# Human-readable class descriptions
CLASS_DESCRIPTIONS = {
    "D00": "Longitudinal/transverse cracks",
    "D10": "Reflection cracks (concrete joints)",
    "D20": "Fatigue/alligator cracking",
    "D40": "Potholes and patching",
}


@app.on_event("startup")
def load_model():
    model = _get_model(DEFAULT_MODEL_NAME)
    if model is None:
        return
    # Warm up model on GPU with a dummy image
    import numpy as np
    dummy = Image.fromarray(np.zeros((640, 640, 3), dtype=np.uint8))
    model.predict(source=dummy, conf=0.25, imgsz=640, verbose=False)
    logger.info(f"Model warmed up on device: {next(model.model.parameters()).device}")


def _resolve_model_path(model_name: str) -> Path:
    model_file = Path(model_name).name
    return MODELS_DIR / model_file


def _get_model(model_name: str) -> YOLO | None:
    global active_model_name
    model_path = _resolve_model_path(model_name)
    model_file = model_path.name

    if model_file in loaded_models:
        active_model_name = model_file
        return loaded_models[model_file]

    logger.info(f"Loading YOLO model from {model_path}...")
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        return None

    loaded = YOLO(str(model_path))
    logger.info(f"YOLO model loaded ({model_file}). Classes: {loaded.names}")
    loaded_models[model_file] = loaded
    active_model_name = model_file
    return loaded


@app.get("/health")
def health():
    available_models = sorted([p.name for p in MODELS_DIR.glob("*.pt")])
    return {
        "status": "ok",
        "model_loaded": len(loaded_models) > 0,
        "model_type": "YOLOv11n-OBB (Road Damage)",
        "active_model": active_model_name,
        "available_models": available_models,
        "classes": loaded_models[active_model_name].names if active_model_name in loaded_models else None,
    }


@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    confidence: float = 0.25,
    model_name: str | None = Query(None, alias="model"),
):
    """
    Accept an image and return road damage detections (OBB).
    """
    selected_model_name = (model_name or DEFAULT_MODEL_NAME).strip()
    model = _get_model(selected_model_name)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model not found: {selected_model_name}")

    if confidence < 0.0 or confidence > 1.0:
        raise HTTPException(status_code=400, detail="Confidence must be between 0 and 1")

    try:
        contents = await file.read()
        # Model was trained on grayscale images — convert to grayscale then
        # back to 3-channel RGB so the tensor shape matches what YOLO expects.
        image = Image.open(io.BytesIO(contents)).convert("L").convert("RGB")
    except Exception as e:
        logger.error(f"Failed to open image: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    try:
        results = model.predict(
            source=image,
            conf=confidence,
            imgsz=640,
            verbose=False,
        )
    except Exception as e:
        logger.error(f"Model prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Model prediction error: {e}")

    detections = []
    for result in results:
        # OBB model: use result.obb for oriented bounding boxes
        obb = result.obb
        if obb is not None and len(obb):
            for i in range(len(obb)):
                cls_id = int(obb.cls[i])
                cls_name = result.names[cls_id]
                conf = round(float(obb.conf[i]), 4)
                # obb.xyxyxyxy gives 4 corner points of oriented box
                corners = obb.xyxyxyxy[i].tolist()
                # Also get axis-aligned bounding box from obb.xyxy
                xyxy = obb.xyxy[i].tolist()

                detections.append({
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "class_description": CLASS_DESCRIPTIONS.get(cls_name, cls_name),
                    "confidence": conf,
                    "bbox": {
                        "x1": round(xyxy[0], 2),
                        "y1": round(xyxy[1], 2),
                        "x2": round(xyxy[2], 2),
                        "y2": round(xyxy[3], 2),
                    },
                    "oriented_bbox": [
                        [round(c[0], 2), round(c[1], 2)] for c in corners
                    ],
                })

        # Fallback: also check result.boxes if present (non-OBB predictions)
        elif result.boxes is not None and len(result.boxes):
            for box in result.boxes:
                cls_id = int(box.cls[0])
                cls_name = result.names[cls_id]
                detections.append({
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "class_description": CLASS_DESCRIPTIONS.get(cls_name, cls_name),
                    "confidence": round(float(box.conf[0]), 4),
                    "bbox": {
                        "x1": round(float(box.xyxy[0][0]), 2),
                        "y1": round(float(box.xyxy[0][1]), 2),
                        "x2": round(float(box.xyxy[0][2]), 2),
                        "y2": round(float(box.xyxy[0][3]), 2),
                    },
                })

    # Generate annotated image with OBB boxes drawn
    annotated_b64 = None
    if results and len(detections) > 0:
        try:
            annotated_bgr = results[0].plot()  # numpy BGR array
            annotated_rgb = annotated_bgr[:, :, ::-1]  # BGR -> RGB
            ann_img = Image.fromarray(annotated_rgb)
            buf = io.BytesIO()
            ann_img.save(buf, format="JPEG", quality=85)
            annotated_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception as e:
            logger.warning(f"Failed to generate annotated image: {e}")

    return {
        "filename": file.filename,
        "model": active_model_name,
        "image_size": {"width": image.width, "height": image.height},
        "detections_count": len(detections),
        "detections": detections,
        "annotated_image": annotated_b64,
    }
