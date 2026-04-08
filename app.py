from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import json, os, io

app = Flask(__name__)

# ── Load model & class names once at startup ──────────────────────────────
MODEL_PATH  = os.path.join("model", "model.h5")
LABELS_PATH = os.path.join("model", "class_names.json")

model = tf.keras.models.load_model(MODEL_PATH)

with open(LABELS_PATH) as f:
    class_names = json.load(f)

IMG_SIZE = (224, 224)

def preprocess(file_bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file  = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    img_bytes = file.read()
    tensor    = preprocess(img_bytes)
    preds     = model.predict(tensor)[0]

    top3_idx  = np.argsort(preds)[::-1][:3]
    results   = [
        {"label": class_names[i].replace("_", " "), "confidence": round(float(preds[i]) * 100, 2)}
        for i in top3_idx
    ]
    return jsonify({"predictions": results})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)