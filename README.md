# Fruit Freshness Detection Website

A demo web app for detecting whether apples, bananas, and oranges are fresh or rotten.

Model: YOLOv10s trained on the fruit freshness dataset.

## Local Run

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m uvicorn server:app --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

## Deploy Notes

GitHub Pages cannot run this Python backend. Use Render, Hugging Face Spaces, Railway, or another Python hosting service.

If the frontend is hosted separately on GitHub Pages, edit:

```text
static/config.js
```

Set:

```javascript
window.API_BASE_URL = "https://your-backend-url.onrender.com";
```

## API

Health check:

```text
GET /api/health
```

Prediction:

```text
POST /api/predict
Form field: file=<image>
```

## Streamlit Community Cloud Deployment

This repository also includes a Streamlit app entrypoint:

```text
streamlit_app.py
```

Deploy steps:

1. Go to https://share.streamlit.io
2. Click Create app
3. Select repository: Donalyn1008/fruit-freshness-web
4. Branch: main
5. Main file path: streamlit_app.py
6. Deploy

This is similar to the Computer-assembly Streamlit project.

## Fix Streamlit Install Error

If Streamlit shows `Error installing requirements`, open **Manage app -> Settings -> Advanced settings** and set:

```text
Python version: 3.10
```

Reason: this app uses PyTorch `2.0.1+cpu`, which is compatible with Python 3.10 but may fail on Streamlit's default newer Python version.

Then click:

```text
Reboot app
```

or delete and redeploy the app with:

```text
Repository: Donalyn1008/fruit-freshness-web
Branch: main
Main file path: streamlit_app.py
Python version: 3.10
```

## Fix OpenCV / cv2 ImportError on Streamlit Cloud

If the app fails at:

```text
import cv2
```

Streamlit Cloud needs Linux system packages for OpenCV. This repo includes:

```text
packages.txt
```

After this file is pushed, reboot or redeploy the Streamlit app so Streamlit installs those apt packages before Python requirements.

## Streamlit Cloud apt dependency note

`packages.txt` intentionally only installs:

```text
libgl1
```

Do not add `libglib2.0-0` on Streamlit Cloud because the current Debian image may resolve it from an incompatible repository and fail with missing `libffi7` / `libpcre3` dependencies.
