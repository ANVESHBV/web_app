#!/usr/bin/env python
"""
UI Detection Web App
Run: python app.py
Then open: http://localhost:5000
"""

from flask import Flask, request, jsonify, render_template, send_file
from pathlib import Path
import base64
import io
import os
import tempfile

app = Flask(__name__)

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_PATH = Path("best.pt")      # put best.pt in the same folder as app.py
CONFIDENCE = 0.25
MAX_FILE_MB = 20
# ────────────────────────────────────────────────────────────────────────────

model = None

def get_model():
    global model
    if model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file '{MODEL_PATH}' not found. "
                "Place best.pt in the same directory as app.py."
            )
        from ultralytics import YOLO
        model = YOLO(str(MODEL_PATH))
        print(f"✓ Model loaded: {MODEL_PATH}")
    return model


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/model-status")
def model_status():
    """Check whether model file exists"""
    return jsonify({"ready": MODEL_PATH.exists(), "path": str(MODEL_PATH)})


@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Size check
    file.seek(0, 2)
    size_mb = file.tell() / (1024 * 1024)
    file.seek(0)
    if size_mb > MAX_FILE_MB:
        return jsonify({"error": f"File too large ({size_mb:.1f} MB). Max {MAX_FILE_MB} MB."}), 400

    allowed = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed:
        return jsonify({"error": f"Unsupported format '{suffix}'. Use JPG/PNG/BMP/WEBP."}), 400

    try:
        yolo = get_model()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503

    # Save upload to temp file
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        results = yolo.predict(
            source=tmp_path,
            conf=CONFIDENCE,
            save=False,
            verbose=False,
        )
    finally:
        os.unlink(tmp_path)

    result = results[0]

    # Render annotated image to bytes
    import cv2
    import numpy as np

    annotated = result.plot()          # numpy array (BGR)
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    # Encode to PNG
    from PIL import Image
    pil_img = Image.fromarray(annotated_rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode()

    # Build detections list
    detections = []
    summary = {}
    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = result.names[class_id]
        conf_score = round(float(box.conf[0]), 3)
        x1, y1, x2, y2 = [int(c) for c in box.xyxy[0]]
        detections.append({
            "class": class_name,
            "confidence": conf_score,
            "bbox": [x1, y1, x2, y2],
        })
        summary[class_name] = summary.get(class_name, 0) + 1

    return jsonify({
        "image": f"data:image/png;base64,{img_b64}",
        "total": len(detections),
        "detections": detections,
        "summary": summary,
        "image_size": {
            "width": result.orig_shape[1],
            "height": result.orig_shape[0],
        },
    })


if __name__ == "__main__":
    print("=" * 60)
    print("  UI Detection Web App")
    print("=" * 60)
    if MODEL_PATH.exists():
        print(f"  ✓ Model found: {MODEL_PATH}")
    else:
        print(f"  ⚠  Model NOT found — place best.pt here: {MODEL_PATH.resolve()}")
    print(f"  ✓ Confidence threshold: {CONFIDENCE}")
    print(f"  → Open http://localhost:5000")
    print("=" * 60)
    app.run(debug=False, host="0.0.0.0", port=5000)
