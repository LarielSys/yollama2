import httpx
import json

img_path = r"C:\Git Hub Road Detection\Road-Damage-Detection\Dataset\test\images\China_Drone_000107.jpg"

with open(img_path, "rb") as f:
    data = f.read()

print(f"Image size: {len(data)} bytes")

r = httpx.post(
    "http://localhost:8001/detect",
    files={"file": ("sample.jpg", data, "image/jpeg")},
    params={"confidence": 0.25},
    timeout=60.0,
)

print(f"Status: {r.status_code}")
d = r.json()
print(json.dumps(d, indent=2))
