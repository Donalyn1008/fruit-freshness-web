const input = document.querySelector("#imageInput");
const button = document.querySelector("#predictBtn");
const statusText = document.querySelector("#statusText");
const resultImage = document.querySelector("#resultImage");
const emptyImage = document.querySelector("#emptyImage");
const summary = document.querySelector("#summary");
const detections = document.querySelector("#detections");
const dropzone = document.querySelector("#dropzone");

let selectedFile = null;

function setStatus(text) {
  statusText.textContent = text;
}

function setFile(file) {
  selectedFile = file;
  setStatus(file ? `已選擇：${file.name}` : "等待上傳圖片");
}

input.addEventListener("change", (event) => {
  setFile(event.target.files[0]);
});

["dragenter", "dragover"].forEach((name) => {
  dropzone.addEventListener(name, (event) => {
    event.preventDefault();
    dropzone.classList.add("dragover");
  });
});

["dragleave", "drop"].forEach((name) => {
  dropzone.addEventListener(name, (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragover");
  });
});

dropzone.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files[0];
  if (file) setFile(file);
});

button.addEventListener("click", async () => {
  if (!selectedFile) {
    setStatus("請先選擇一張圖片。");
    return;
  }

  button.disabled = true;
  setStatus("模型辨識中，第一次載入可能會等幾秒...");
  detections.innerHTML = "";

  const form = new FormData();
  form.append("file", selectedFile);

  try {
    const response = await fetch(`${window.API_BASE_URL || ""}/api/predict`, { method: "POST", body: form });
    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.detail || "辨識失敗");
    }

    const data = await response.json();
    resultImage.src = data.annotated_image;
    resultImage.style.display = "block";
    emptyImage.style.display = "none";
    summary.textContent = data.summary;

    if (!data.detections.length) {
      detections.innerHTML = `<div class="det"><div class="det-title">無法識別</div><div class="det-meta">請上傳蘋果、香蕉或橘子的圖片。</div></div>`;
    } else {
      detections.innerHTML = data.detections.map((det, index) => {
        const cls = det.status === "Rotten" ? "det rotten" : "det";
        const percent = Math.round(det.confidence * 100);
        return `
          <div class="${cls}">
            <div class="det-title">#${index + 1} ${det.label}</div>
            <div class="det-meta">狀態：${det.status} | 水果：${det.fruit} | 信心分數：${percent}%</div>
            <div class="det-meta">建議：${det.advice}</div>
          </div>`;
      }).join("");
    }

    setStatus(`完成：${data.model} 偵測到 ${data.count} 個物件。`);
  } catch (error) {
    summary.textContent = "辨識失敗";
    setStatus(error.message);
  } finally {
    button.disabled = false;
  }
});
