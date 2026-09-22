const fileInput = document.getElementById("fileInput");
const dropZone = document.getElementById("dropZone");
const detectBtn = document.getElementById("detectBtn");
const resetBtn = document.getElementById("resetBtn");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const canvasWrap = document.getElementById("canvasWrap");
const countEl = document.getElementById("count");
const confidence = document.getElementById("confidence");
const confidenceValue = document.getElementById("confidenceValue");
const tiling = document.getElementById("tiling");
const modelStatus = document.getElementById("modelStatus");
const imageInfo = document.getElementById("imageInfo");
const summary = document.getElementById("confidenceSummary");
const minusBtn = document.getElementById("minusBtn");
const plusBtn = document.getElementById("plusBtn");

let model = null;
let sourceImage = null;
let imageURL = null;
let detectedCount = 0;
let manualAdjustment = 0;
let lastDetections = [];

confidence.addEventListener("input", () => {
  confidenceValue.textContent = `${Math.round(Number(confidence.value) * 100)}%`;
});

async function loadModel() {
  try {
    await tf.ready();
    model = await cocoSsd.load({ base: "mobilenet_v2" });
    modelStatus.textContent = "AI model ready";
    modelStatus.className = "status ready";
    updateButton();
  } catch (error) {
    console.error(error);
    modelStatus.textContent = "Model failed to load";
    modelStatus.className = "status loading";
  }
}

function updateButton() {
  detectBtn.disabled = !model || !sourceImage;
}

function loadImage(file) {
  if (!file || !file.type.startsWith("image/")) return;

  if (imageURL) URL.revokeObjectURL(imageURL);
  imageURL = URL.createObjectURL(file);

  const img = new Image();
  img.onload = () => {
    sourceImage = img;
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    canvasWrap.classList.remove("empty");
    document.querySelector(".empty-state").style.display = "none";
    canvas.style.display = "block";
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0);
    imageInfo.textContent = `${img.naturalWidth} × ${img.naturalHeight}`;
    countEl.textContent = "—";
    summary.textContent = "Ready to analyze";
    detectedCount = 0;
    manualAdjustment = 0;
    lastDetections = [];
    updateButton();
  };
  img.onerror = () => alert("Could not read this image.");
  img.src = imageURL;
}

fileInput.addEventListener("change", e => loadImage(e.target.files[0]));

["dragenter", "dragover"].forEach(eventName => {
  dropZone.addEventListener(eventName, e => {
    e.preventDefault();
    dropZone.classList.add("dragging");
  });
});
["dragleave", "drop"].forEach(eventName => {
  dropZone.addEventListener(eventName, e => {
    e.preventDefault();
    dropZone.classList.remove("dragging");
  });
});
dropZone.addEventListener("drop", e => {
  const file = e.dataTransfer.files[0];
  loadImage(file);
});

function iou(a, b) {
  const ax1 = a.x, ay1 = a.y, ax2 = a.x + a.w, ay2 = a.y + a.h;
  const bx1 = b.x, by1 = b.y, bx2 = b.x + b.w, by2 = b.y + b.h;
  const ix1 = Math.max(ax1, bx1), iy1 = Math.max(ay1, by1);
  const ix2 = Math.min(ax2, bx2), iy2 = Math.min(ay2, by2);
  const iw = Math.max(0, ix2 - ix1), ih = Math.max(0, iy2 - iy1);
  const intersection = iw * ih;
  const union = a.w * a.h + b.w * b.h - intersection;
  return union <= 0 ? 0 : intersection / union;
}

// Non-Maximum Suppression removes duplicate person detections
// created when the same person appears in overlapping tiles.
function nms(detections, threshold = 0.45) {
  const sorted = [...detections].sort((a, b) => b.score - a.score);
  const kept = [];

  while (sorted.length) {
    const best = sorted.shift();
    kept.push(best);

    for (let i = sorted.length - 1; i >= 0; i--) {
      if (iou(best, sorted[i]) > threshold) sorted.splice(i, 1);
    }
  }
  return kept;
}

function makeTiles(width, height) {
  const target = 960;
  const overlap = 0.25;
  const tileW = Math.min(target, width);
  const tileH = Math.min(Math.round(target * height / width), height);

  if (width <= target) return [{ x: 0, y: 0, w: width, h: height }];

  const stepX = Math.max(1, Math.floor(tileW * (1 - overlap)));
  const stepY = Math.max(1, Math.floor(tileH * (1 - overlap)));
  const tiles = [];

  for (let y = 0; y < height; y += stepY) {
    for (let x = 0; x < width; x += stepX) {
      const w = Math.min(tileW, width - x);
      const h = Math.min(tileH, height - y);
      tiles.push({ x, y, w, h });
      if (x + w >= width) break;
    }
    if (y + tileH >= height) break;
  }
  return tiles;
}

async function detectSingle(source, minScore) {
  return await model.detect(source, 100, minScore);
}

async function analyze() {
  if (!model || !sourceImage) return;

  detectBtn.disabled = true;
  detectBtn.textContent = "Analyzing…";
  summary.textContent = "Running AI detection…";
  manualAdjustment = 0;

  const minScore = Number(confidence.value);
  let detections = [];

  try {
    if (tiling.value === "on") {
      const tiles = makeTiles(sourceImage.naturalWidth, sourceImage.naturalHeight);
      const tileCanvas = document.createElement("canvas");
      const tileCtx = tileCanvas.getContext("2d");

      for (let i = 0; i < tiles.length; i++) {
        const tile = tiles[i];
        tileCanvas.width = tile.w;
        tileCanvas.height = tile.h;
        tileCtx.drawImage(
          sourceImage,
          tile.x, tile.y, tile.w, tile.h,
          0, 0, tile.w, tile.h
        );

        const predictions = await detectSingle(tileCanvas, minScore);
        for (const p of predictions) {
          if (p.class !== "person") continue;
          detections.push({
            x: p.bbox[0] + tile.x,
            y: p.bbox[1] + tile.y,
            w: p.bbox[2],
            h: p.bbox[3],
            score: p.score
          });
        }
      }
    } else {
      const predictions = await detectSingle(sourceImage, minScore);
      detections = predictions
        .filter(p => p.class === "person")
        .map(p => ({
          x: p.bbox[0], y: p.bbox[1],
          w: p.bbox[2], h: p.bbox[3],
          score: p.score
        }));
    }

    lastDetections = nms(detections);
    detectedCount = lastDetections.length;
    render();
  } catch (error) {
    console.error(error);
    summary.textContent = "Detection failed. Try another image.";
  } finally {
    detectBtn.disabled = false;
    detectBtn.textContent = "Analyze Photo";
    updateButton();
  }
}

function render() {
  if (!sourceImage) return;

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(sourceImage, 0, 0);

  const scale = Math.max(1, sourceImage.naturalWidth / 1600);
  ctx.lineWidth = Math.max(3, 3 * scale);
  ctx.font = `700 ${Math.max(14, 16 * scale)}px system-ui`;

  lastDetections.forEach((d, index) => {
    ctx.strokeStyle = "#55a7ff";
    ctx.fillStyle = "rgba(85,167,255,.12)";
    ctx.strokeRect(d.x, d.y, d.w, d.h);
    ctx.fillRect(d.x, d.y, d.w, d.h);

    const label = `${index + 1}  ${Math.round(d.score * 100)}%`;
    const pad = 5 * scale;
    const textW = ctx.measureText(label).width;
    const labelH = 24 * scale;
    const lx = d.x;
    const ly = Math.max(labelH, d.y);

    ctx.fillStyle = "#55a7ff";
    ctx.fillRect(lx, ly - labelH, textW + pad * 2, labelH);
    ctx.fillStyle = "#04101d";
    ctx.fillText(label, lx + pad, ly - 7 * scale);
  });

  const finalCount = Math.max(0, detectedCount + manualAdjustment);
  countEl.textContent = finalCount;

  if (lastDetections.length) {
    const avg = lastDetections.reduce((s, d) => s + d.score, 0) / lastDetections.length;
    summary.textContent = `${lastDetections.length} AI detections • ${Math.round(avg * 100)}% average confidence`;
  } else {
    summary.textContent = "No people detected at this threshold";
  }
}

minusBtn.addEventListener("click", () => {
  manualAdjustment--;
  render();
});
plusBtn.addEventListener("click", () => {
  manualAdjustment++;
  render();
});

resetBtn.addEventListener("click", () => {
  if (imageURL) URL.revokeObjectURL(imageURL);
  imageURL = null;
  sourceImage = null;
  detectedCount = 0;
  manualAdjustment = 0;
  lastDetections = [];
  fileInput.value = "";
  canvas.style.display = "none";
  canvasWrap.classList.add("empty");
  document.querySelector(".empty-state").style.display = "";
  countEl.textContent = "—";
  summary.textContent = "Waiting for an image";
  imageInfo.textContent = "No image loaded";
  updateButton();
});

detectBtn.addEventListener("click", analyze);
loadModel();
