from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLOv10

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "yolov10s-fruit-best.pt"
STATIC_DIR = ROOT / "static"
CONF_THRESHOLD = float(os.getenv("FRUIT_CONF", "0.25"))

app = FastAPI(title="Fruit Freshness Detection Demo")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

model: YOLOv10 | None = None

STATUS_MAP = {
    "Fresh Apples": ("Fresh", "Apple", "可食用", "#1f9d55"),
    "Fresh Banana": ("Fresh", "Banana", "可食用", "#1f9d55"),
    "Fresh Oranges": ("Fresh", "Orange", "可食用", "#1f9d55"),
    "Rotten Apples": ("Rotten", "Apple", "不建議食用", "#d64545"),
    "Rotten Banana": ("Rotten", "Banana", "不建議食用", "#d64545"),
    "Rotten Oranges": ("Rotten", "Orange", "不建議食用", "#d64545"),
}


def get_model() -> YOLOv10:
    global model
    if model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        model = YOLOv10(str(MODEL_PATH))
    return model


def image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def draw_detection(image: Image.Image, detections: list[dict[str, Any]]) -> Image.Image:
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)

    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except OSError:
        font = ImageFont.load_default()

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        color = det["color"]
        label = f"{det['label']} {det['confidence']:.2f}"
        draw.rectangle((x1, y1, x2, y2), outline=color, width=4)

        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        label_w = right - left + 12
        label_h = bottom - top + 10
        y_label = max(0, y1 - label_h)
        draw.rectangle((x1, y_label, x1 + label_w, y_label + label_h), fill=color)
        draw.text((x1 + 6, y_label + 5), label, fill="white", font=font)

    return annotated


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": "YOLOv10s"}


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="請上傳圖片檔案。")

    content = await file.read()
    try:
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="圖片無法讀取，請換一張圖片。") from exc

    predictor = get_model()
    result = predictor.predict(image, conf=CONF_THRESHOLD, verbose=False)[0]

    detections: list[dict[str, Any]] = []
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0])
            label = predictor.names[class_id]
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
            status, fruit, advice, color = STATUS_MAP.get(label, ("Unknown", label, "請人工確認", "#444444"))
            detections.append(
                {
                    "label": label,
                    "status": status,
                    "fruit": fruit,
                    "advice": advice,
                    "confidence": confidence,
                    "box": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                    "color": color,
                }
            )

    detections.sort(key=lambda item: item["confidence"], reverse=True)
    annotated = draw_detection(image, detections)

    if detections:
        rotten_count = sum(1 for item in detections if item["status"] == "Rotten")
        fresh_count = sum(1 for item in detections if item["status"] == "Fresh")
        summary = f"偵測到 {rotten_count} 個腐爛水果，建議不要食用。" if rotten_count else f"偵測到 {fresh_count} 個新鮮水果。"
    else:
        summary = "沒有偵測到水果，請換一張更清楚的圖片。"

    return {
        "summary": summary,
        "count": len(detections),
        "detections": detections,
        "annotated_image": image_to_data_url(annotated),
        "model": "YOLOv10s",
        "confidence_threshold": CONF_THRESHOLD,
    }
