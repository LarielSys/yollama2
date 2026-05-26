FROM ultralytics/ultralytics:latest

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

RUN mkdir -p /app/models /app/input

# Copy road damage detection model (YOLOv11n-OBB, RDD2022)
COPY models/road_damage.pt /app/models/road_damage.pt

EXPOSE 8001

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
