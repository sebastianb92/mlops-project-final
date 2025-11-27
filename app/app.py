import os
import io
import json
import time
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template
import onnxruntime

# --- CONFIGURACIÓN Y CARGA DE MODELO ---
# Se espera que 'download_model.py' haya colocado el modelo aquí
MODEL_PATH = "app/mobilenetv2-7.onnx"
LABELS_PATH = "imagenet_classes.txt"
LABELS = []

# Determinar el entorno para el registro (dev o prod)
ENVIRONMENT = os.environ.get("ENVIRONMENT", "local") 
PREDICTION_LOG_FILE = f"predicciones_{ENVIRONMENT}.txt" 

# Cargar las etiquetas de ImageNet
try:
    with open(LABELS_PATH, 'r') as f:
        LABELS = [line.strip() for line in f.readlines()]
    print(f"Etiquetas cargadas correctamente. Total: {len(LABELS)}")
except Exception as e:
    print(f"ADVERTENCIA: No se pudo cargar el archivo de etiquetas '{LABELS_PATH}': {e}")
    
app = Flask(__name__)
ort_session = None
input_name = None

# Cargar el modelo ONNX Runtime
try:
    if not os.path.exists(MODEL_PATH):
        print(f"🚨 ERROR: Modelo no encontrado en {MODEL_PATH}. Debe ser descargado durante el build.")
        raise FileNotFoundError(f"Modelo ONNX faltante en {MODEL_PATH}")
        
    sess_options = onnxruntime.SessionOptions()
    ort_session = onnxruntime.InferenceSession(MODEL_PATH, sess_options, providers=['CPUExecutionProvider'])
    input_name = ort_session.get_inputs()[0].name
    print(f"Modelo cargado correctamente. Nombre de la entrada: {input_name}")
    print(f"Registro de predicciones a: {PREDICTION_LOG_FILE}")
    
except Exception as e:
    print(f"Error al cargar el modelo ONNX: {e}")

INPUT_SIZE = (224, 224)

# --- Funciones Auxiliares ---

def log_prediction(prediction_data):
    """Escribe la predicción en el archivo de registro (simula el bucket)."""
    # En un entorno real, esta función subiría el JSON al bucket.
    # Aquí, escribimos en un archivo local para cumplir el requisito.
    log_entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "environment": ENVIRONMENT,
        "label": prediction_data.get("predicted_label", "N/A"),
        "confidence": prediction_data.get("confidence", "N/A"),
        "index": prediction_data.get("prediction_index", -1)
    }
    
    try:
        # Abre el archivo en modo append y escribe la línea JSON
        with open(PREDICTION_LOG_FILE, 'a') as f:
            f.write(json.dumps(log_entry) + "\n")
        print(f"Registro exitoso en {PREDICTION_LOG_FILE}")
    except Exception as e:
        print(f"ADVERTENCIA: No se pudo escribir en el archivo de registro {PREDICTION_LOG_FILE}: {e}")

def preprocess_image(image_bytes):
    if ort_session is None: return None
        
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image = image.convert("RGB") 
        image = image.resize(INPUT_SIZE)

        img_data = np.asarray(image, dtype=np.float32)
        img_data = img_data / 255.0  

        img_data = img_data.transpose([2, 0, 1])
        input_tensor = np.expand_dims(img_data, axis=0)
        return input_tensor
    except Exception as e:
        print(f"Error en el preprocesamiento: {e}")
        return None

# --- Endpoints ---

@app.route('/')
def index():
    # Muestra el entorno actual en la consola (útil para el demo)
    print(f"Servicio corriendo en el entorno: {ENVIRONMENT}")
    return render_template('index.html') 

@app.route('/predict', methods=['POST'])
def predict():
    if ort_session is None:
        return jsonify({"error": "El modelo no se cargó correctamente."}), 500

    if 'file' not in request.files:
        return jsonify({"error": "No se encontró el campo 'file'."}), 400

    file = request.files['file']
    image_bytes = file.read()
    input_tensor = preprocess_image(image_bytes)

    if input_tensor is None:
        return jsonify({"error": "Error al procesar la imagen."}), 400

    try:
        ort_inputs = {input_name: input_tensor}
        ort_outputs = ort_session.run(None, ort_inputs)
        output_data = ort_outputs[0]
        
        prediction_index = np.argmax(output_data).item()
        # Se aplica softmax para obtener una probabilidad real (no solo logit)
        exp_output = np.exp(output_data - np.max(output_data))
        probabilities = exp_output / np.sum(exp_output)
        confidence = np.max(probabilities).item()
        
        predicted_label = "Etiqueta desconocida"
        if LABELS and 0 <= prediction_index < len(LABELS):
            predicted_label = LABELS[prediction_index]
        
        response_data = {
            "success": True,
            "prediction_index": prediction_index,
            "predicted_label": predicted_label,
            "confidence": f"{confidence:.4f}",
            "message": f"Predicción para el entorno {ENVIRONMENT}."
        }
        
        # LOGGING: Registrar la predicción antes de responder
        log_prediction(response_data)
        
        return jsonify(response_data)

    except Exception as e:
        return jsonify({"error": f"Error interno durante la inferencia ONNX: {e}"}), 500

if __name__ == '__main__':
    # Usar un puerto diferente para entornos locales si es necesario
    app.run(debug=False, host='0.0.0.0', port=8080)