from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import streamlit as st
import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLOv10

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "yolov10s-fruit-best.pt"
CONF_THRESHOLD = 0.5
MAX_BOX_AREA_RATIO = 0.45
LARGE_BOX_CONF_THRESHOLD = 0.9
_TORCH_LOAD_PATCHED = False

STATUS_MAP = {
    "Fresh Apples": ("Fresh", "Apple", "可食用", "#1f9d55"),
    "Fresh Banana": ("Fresh", "Banana", "可食用", "#1f9d55"),
    "Fresh Oranges": ("Fresh", "Orange", "可食用", "#1f9d55"),
    "Rotten Apples": ("Rotten", "Apple", "不建議食用", "#d64545"),
    "Rotten Banana": ("Rotten", "Banana", "不建議食用", "#d64545"),
    "Rotten Oranges": ("Rotten", "Orange", "不建議食用", "#d64545"),
}
SUPPORTED_LABELS = set(STATUS_MAP)

TEXT = {
    "zh": {
        "lang_button": "Switch to English",
        "title": "水果新鮮度辨識系統",
        "subtitle": "使用 YOLOv10s 偵測蘋果、香蕉、橘子，並判斷 Fresh / Rotten。",
        "upload": "上傳水果圖片",
        "upload_info": "請上傳一張水果圖片開始辨識。",
        "spinner": "模型辨識中，第一次載入可能需要幾秒...",
        "result_image": "偵測結果圖片",
        "summary": "辨識摘要",
        "unrecognized": "無法識別。請上傳蘋果、香蕉或橘子的圖片。",
        "rotten_summary": "偵測到 {count} 個腐爛水果，建議不要食用。",
        "fresh_summary": "偵測到 {count} 個新鮮水果。",
        "status": "狀態",
        "fruit": "水果",
        "confidence": "信心分數",
        "advice": "建議",
        "caption": "Fresh / Rotten 是類別判斷；bounding box 是圖片上框出水果位置的方框。",
        "fresh": "新鮮",
        "rotten": "腐爛",
        "apple": "蘋果",
        "banana": "香蕉",
        "orange": "橘子",
        "edible": "可食用",
        "not_edible": "不建議食用",
    },
    "en": {
        "lang_button": "切換成中文",
        "title": "Fruit Freshness Detection System",
        "subtitle": "Use YOLOv10s to detect apples, bananas, and oranges, then classify them as Fresh or Rotten.",
        "upload": "Upload a fruit image",
        "upload_info": "Upload a fruit image to start detection.",
        "spinner": "Running detection. First load may take a few seconds...",
        "result_image": "Detection Result",
        "summary": "Prediction Summary",
        "unrecognized": "Unrecognized. Please upload an image of an apple, banana, or orange.",
        "rotten_summary": "Detected {count} rotten fruit(s). Eating is not recommended.",
        "fresh_summary": "Detected {count} fresh fruit(s).",
        "status": "Status",
        "fruit": "Fruit",
        "confidence": "Confidence",
        "advice": "Advice",
        "caption": "Fresh / Rotten is the class prediction; the bounding box shows where the fruit is located.",
        "fresh": "Fresh",
        "rotten": "Rotten",
        "apple": "Apple",
        "banana": "Banana",
        "orange": "Orange",
        "edible": "Edible",
        "not_edible": "Not recommended",
    },
}

FRUIT_TEXT_KEY = {
    "Apple": "apple",
    "Banana": "banana",
    "Orange": "orange",
}

st.set_page_config(
    page_title="Fruit Freshness Detector",
    page_icon="🍎",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 3rem;
        font-weight: 900;
        line-height: 1.05;
        margin-bottom: 0.25rem;
    }
    .subtle {
        color: #657064;
        font-size: 1.1rem;
    }
    .metric-card {
        border: 1px solid rgba(23, 32, 24, 0.12);
        border-radius: 18px;
        padding: 1rem;
        background: #fff8e8;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "lang" not in st.session_state:
    st.session_state.lang = "zh"

if st.button(TEXT[st.session_state.lang]["lang_button"]):
    st.session_state.lang = "en" if st.session_state.lang == "zh" else "zh"
    st.rerun()

t = TEXT[st.session_state.lang]

@st.cache_resource
def load_model() -> YOLOv10:
    patch_torch_load_for_yolov10()
    return YOLOv10(str(MODEL_PATH))


def patch_torch_load_for_yolov10() -> None:
    global _TORCH_LOAD_PATCHED
    if _TORCH_LOAD_PATCHED:
        return

    original_torch_load = torch.load

    def compatible_torch_load(*args: Any, **kwargs: Any) -> Any:
        # PyTorch 2.6+ defaults to weights_only=True, which rejects older
        # Ultralytics checkpoints. This app only loads our own bundled model.
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = compatible_torch_load  # type: ignore[assignment]
    _TORCH_LOAD_PATCHED = True


def draw_detection(image: Image.Image, detections: list[dict[str, Any]]) -> Image.Image:
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except OSError:
        font = ImageFont.load_default()

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        color = det["color"]
        label = f"{det['label']} {det['confidence']:.2f}"
        draw.rectangle((x1, y1, x2, y2), outline=color, width=5)
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        label_w = right - left + 14
        label_h = bottom - top + 12
        y_label = max(0, y1 - label_h)
        draw.rectangle((x1, y_label, x1 + label_w, y_label + label_h), fill=color)
        draw.text((x1 + 7, y_label + 6), label, fill="white", font=font)
    return annotated


def predict(image: Image.Image) -> tuple[Image.Image, list[dict[str, Any]]]:
    model = load_model()
    result = model.predict(image.convert("RGB"), conf=CONF_THRESHOLD, verbose=False)[0]
    detections: list[dict[str, Any]] = []
    image_area = image.width * image.height

    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0])
            label = model.names[class_id]
            if label not in SUPPORTED_LABELS:
                continue
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
            box_area_ratio = max(0.0, (x2 - x1) * (y2 - y1)) / image_area
            if box_area_ratio > MAX_BOX_AREA_RATIO and confidence < LARGE_BOX_CONF_THRESHOLD:
                continue
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
    return draw_detection(image, detections), detections


st.markdown(f'<div class="main-title">{t["title"]}</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="subtle">{t["subtitle"]}</div>',
    unsafe_allow_html=True,
)

col_a, col_b, col_c = st.columns(3)
col_a.metric("Model", "YOLOv10s")
col_b.metric("mAP@0.5", "0.93545")
col_c.metric("Precision", "0.92559")

uploaded = st.file_uploader(t["upload"], type=["jpg", "jpeg", "png", "webp"])

if uploaded is None:
    st.info(t["upload_info"])
    st.stop()

image = Image.open(io.BytesIO(uploaded.read())).convert("RGB")

with st.spinner(t["spinner"]):
    annotated, detections = predict(image)

left, right = st.columns([1.35, 0.65])
with left:
    st.subheader(t["result_image"])
    st.image(annotated, use_container_width=True)

with right:
    st.subheader(t["summary"])
    if not detections:
        st.warning(t["unrecognized"])
    else:
        rotten_count = sum(1 for det in detections if det["status"] == "Rotten")
        fresh_count = sum(1 for det in detections if det["status"] == "Fresh")
        if rotten_count:
            st.error(t["rotten_summary"].format(count=rotten_count))
        else:
            st.success(t["fresh_summary"].format(count=fresh_count))

        for index, det in enumerate(detections, start=1):
            confidence_percent = round(det["confidence"] * 100, 1)
            status_text = t["fresh"] if det["status"] == "Fresh" else t["rotten"]
            fruit_text = t[FRUIT_TEXT_KEY.get(det["fruit"], "fruit")] if det["fruit"] in FRUIT_TEXT_KEY else det["fruit"]
            advice_text = t["edible"] if det["status"] == "Fresh" else t["not_edible"]
            with st.container(border=True):
                st.write(f"#{index} **{det['label']}**")
                st.write(f"{t['status']}：**{status_text}**")
                st.write(f"{t['fruit']}：{fruit_text}")
                st.write(f"{t['confidence']}：{confidence_percent}%")
                st.write(f"{t['advice']}：{advice_text}")

st.caption(t["caption"])
