"""Image Similarity Sorter — CLIP-based visual matching against master images."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# ── Defaults ──────────────────────────────────────────────────────────────────
APP_DIR = Path(__file__).resolve().parent
MASTERS_DIR = APP_DIR / "masters"
OUTPUT_DIR = APP_DIR / "output"
DEFAULT_THRESHOLD = 0.92
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "openai/clip-vit-base-patch32"
BATCH_SIZE = 64  # encode survey images in batches for GPU speed


def load_clip(device: str = DEVICE):
    """Load CLIP model + processor once."""
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device).eval()
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    return model, processor


@torch.no_grad()
def encode_images(
    paths: list[Path],
    model: CLIPModel,
    processor: CLIPProcessor,
    device: str = DEVICE,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """Encode a list of images into L2-normalised CLIP embeddings.

    Returns an (N, 512) float32 numpy array.
    """
    all_embeds: list[np.ndarray] = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        pil_images = []
        for p in batch_paths:
            try:
                pil_images.append(Image.open(p).convert("RGB"))
            except Exception:
                # Unreadable file → zero vector (will never match)
                pil_images.append(Image.new("RGB", (224, 224)))
        inputs = processor(images=pil_images, return_tensors="pt", padding=True).to(device)
        out = model.get_image_features(**inputs)
        # Handle both tensor and BaseModelOutputWithPooling returns
        embeds = out if isinstance(out, torch.Tensor) else out.pooler_output
        embeds = embeds / embeds.norm(dim=-1, keepdim=True)
        all_embeds.append(embeds.cpu().numpy())
    return np.concatenate(all_embeds, axis=0)


def load_masters(
    masters_dir: Path,
    model: CLIPModel,
    processor: CLIPProcessor,
    device: str = DEVICE,
) -> dict[str, np.ndarray]:
    """Load master images per class.

    Returns {class_name: (M, 512) embedding array}.
    """
    class_embeds: dict[str, np.ndarray] = {}
    for class_dir in sorted(masters_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        image_files = sorted(
            [f for f in class_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
        )
        if not image_files:
            continue
        embeds = encode_images(image_files, model, processor, device)
        class_embeds[class_dir.name] = embeds
    return class_embeds


def classify_images(
    survey_embeds: np.ndarray,
    class_embeds: dict[str, np.ndarray],
    threshold: float = DEFAULT_THRESHOLD,
) -> list[tuple[str, float]]:
    """For each survey embedding, find the best-matching class.

    Returns a list of (class_name, best_similarity) per image.
    If best similarity < threshold, class_name is '_no_damage'.
    """
    results: list[tuple[str, float]] = []
    for i in range(survey_embeds.shape[0]):
        vec = survey_embeds[i]  # (512,)
        best_class = "_no_damage"
        best_sim = 0.0
        for cls_name, master_mat in class_embeds.items():
            # Cosine similarity against all masters in this class → take max
            sims = master_mat @ vec  # (M,)
            max_sim = float(sims.max())
            if max_sim > best_sim:
                best_sim = max_sim
                best_class = cls_name
        if best_sim < threshold:
            best_class = "_no_damage"
        results.append((best_class, best_sim))
    return results


def sort_images(
    survey_dir: Path,
    masters_dir: Path = MASTERS_DIR,
    output_dir: Path = OUTPUT_DIR,
    threshold: float = DEFAULT_THRESHOLD,
    copy: bool = True,
    progress_callback=None,
) -> list[dict]:
    """Main entry: scan survey_dir, classify each image, copy/move to output subfolders.

    Returns list of {filename, class_name, similarity} dicts.
    """
    model, processor = load_clip()

    # Load masters
    class_embeds = load_masters(masters_dir, model, processor)
    if not class_embeds:
        raise ValueError(f"No master classes found in {masters_dir}")

    # Discover survey images
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    survey_files = sorted([f for f in survey_dir.iterdir() if f.is_file() and f.suffix.lower() in extensions])
    if not survey_files:
        raise ValueError(f"No images found in {survey_dir}")

    # Encode all survey images
    if progress_callback:
        progress_callback(0, len(survey_files), "Encoding survey images...")
    survey_embeds = encode_images(survey_files, model, processor)

    # Classify
    assignments = classify_images(survey_embeds, class_embeds, threshold)

    # Create output folders
    output_dir.mkdir(parents=True, exist_ok=True)
    all_classes = set(class_embeds.keys()) | {"_no_damage"}
    for cls in all_classes:
        (output_dir / cls).mkdir(exist_ok=True)

    # Copy/move files
    log: list[dict] = []
    for idx, (survey_file, (cls_name, sim)) in enumerate(zip(survey_files, assignments)):
        dest = output_dir / cls_name / survey_file.name
        if copy:
            shutil.copy2(survey_file, dest)
        else:
            shutil.move(str(survey_file), str(dest))
        log.append({
            "filename": survey_file.name,
            "class_name": cls_name,
            "similarity": round(sim, 4),
            "source": str(survey_file),
            "dest": str(dest),
        })
        if progress_callback:
            progress_callback(idx + 1, len(survey_files), f"Sorted {idx + 1}/{len(survey_files)}")

    return log


# ── CLI mode ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python sorter.py <survey_folder> [threshold]")
        print(f"  Masters folder: {MASTERS_DIR}")
        print(f"  Output folder:  {OUTPUT_DIR}")
        sys.exit(1)

    survey = Path(sys.argv[1])
    thresh = float(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_THRESHOLD

    def cli_progress(current, total, msg):
        print(f"\r  {msg}", end="", flush=True)

    print(f"Survey:    {survey}")
    print(f"Masters:   {MASTERS_DIR}")
    print(f"Output:    {OUTPUT_DIR}")
    print(f"Threshold: {thresh}")
    print(f"Device:    {DEVICE}")
    print()

    results = sort_images(survey, threshold=thresh, progress_callback=cli_progress)
    print()

    # Summary
    from collections import Counter
    counts = Counter(r["class_name"] for r in results)
    print(f"\nSorted {len(results)} images:")
    for cls, n in sorted(counts.items()):
        print(f"  {cls}: {n}")
