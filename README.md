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
