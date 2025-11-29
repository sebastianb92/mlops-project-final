import os
import io
import json
import time
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template
import onnxruntime
import boto3

# --- CONFIG S3 ---
S3_BUCKET = os.getenv("S3_BUCKET", "mlops-project-bucket")
S3_KEY = os.getenv("S3_KEY", "models/mobilenetv2-7.onnx")
MODEL_PATH = "app/mobilenetv2-7.onnx"

def download_model_from_s3():
    print("📥 Descargando modelo desde S3...")
    os.makedirs("app", exist_ok=True)
    s3 = boto3.client("s3")

    try:
        s3.download_file(S3_BUCKET, S3_KEY, MODEL_PATH)
        print(f"✅ Modelo descargado: s3://{S3_BUCKET}/{S3_KEY}")
    except Exception as e:
        print("❌ Error descargando modelo desde S3:", e)
        raise e

# Descargar modelo al iniciar
if not os.path.exists(MODEL_PATH):
    download_model_from_s3()

# --- Carga de etiquetas ---
LABELS_PATH = "imagenet_classes.txt"
LABELS = []
try:
    with open(LABELS_PATH, 'r') as f:
        LABELS = [line.strip() for line in f.readlines()]
except:
    print("⚠ No se encontró imagenet_classes.txt")

# --- Cargar modelo ONNX ---
ort_session = onnxruntime.InferenceSession(MODEL_PATH)
input_name = ort_session.get_inputs()[0].name

app = Flask(__name__)
INPUT_SIZE = (224, 224)

def preprocess(img_bytes):
    img = Image.open(io.BytesIO(img_bytes))
    img = img.convert("RGB").resize(INPUT_SIZE)
    data = np.asarray(img, dtype=np.float32) / 255.0
    data = data.transpose([2, 0, 1])
    return np.expand_dims(data, axis=0)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    tensor = preprocess(file.read())
    output = ort_session.run(None, {input_name: tensor})[0]

    idx = int(np.argmax(output))
    label = LABELS[idx] if LABELS else "unknown"

    return jsonify({
        "predicted_label": label,
        "index": idx
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
