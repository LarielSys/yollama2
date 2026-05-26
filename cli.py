"""
YOLLama CLI — Analyze images from the command line.

Usage:
    python cli.py <image_path> [--url http://localhost:8000] [--confidence 0.25] [--no-save]
"""

import argparse
import json
import sys
from pathlib import Path

import httpx


def main():
    parser = argparse.ArgumentParser(description="YOLLama CLI - Analyze images locally")
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("--url", default="http://localhost:8000", help="Orchestrator URL")
    parser.add_argument("--confidence", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--no-save", action="store_true", help="Don't save results server-side")
    parser.add_argument("--output", "-o", help="Save JSON result to this local file")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: File not found: {image_path}")
        sys.exit(1)

    print(f"Analyzing: {image_path.name}")
    print(f"Sending to: {args.url}/analyze")

    with open(image_path, "rb") as f:
        try:
            response = httpx.post(
                f"{args.url}/analyze",
                files={"file": (image_path.name, f, "image/jpeg")},
                params={"confidence": args.confidence, "save": not args.no_save},
                timeout=120.0,
            )
            response.raise_for_status()
        except httpx.ConnectError:
            print(f"Error: Could not connect to {args.url}. Is the system running?")
            sys.exit(1)
        except httpx.HTTPStatusError as e:
            print(f"Error {e.response.status_code}: {e.response.text}")
            sys.exit(1)

    result = response.json()

    # Print summary
    analysis = result.get("analysis", {})
    yolo = result.get("yolo", {})

    print(f"\n{'='*60}")
    print(f"Run ID:     {result.get('run_id')}")
    print(f"YOLO found: {yolo.get('detections_count', 0)} objects")
    print(f"{'='*60}")

    print(f"\nSummary: {analysis.get('summary', 'N/A')}")

    objects = analysis.get("objects", [])
    if objects:
        print(f"\nObjects ({len(objects)}):")
        for obj in objects:
            sev = obj.get("severity", "?")
            print(f"  - {obj['name']} (conf: {obj.get('confidence', '?')}, severity: {sev})")

    rec = analysis.get("recommendation")
    if rec:
        print(f"\nRecommendation: {rec}")

    if args.output:
        out = Path(args.output)
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nResults saved to: {out}")

    print()


if __name__ == "__main__":
    main()
