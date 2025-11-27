import pytest
import os
import json
import numpy as np
import onnxruntime
from PIL import Image
from app.download_model import download_model

# Configuraciones de Paths
TEST_DATA_PATH = "tests/test_data.json"
MODEL_PATH = "app/mobilenetv2-7.onnx"
INPUT_SIZE = (224, 224)

# SIMULACIÓN DE DATOS DE PRUEBA: 
# En CI/CD, estos datos se "descargarían" aquí.
# Para la prueba, simulamos su contenido con un fixture.
@pytest.fixture(scope="session", autouse=True)
def setup_model_and_data():
    """1. Asegura que el modelo se 'descargue'. 2. Genera datos de prueba simulados."""
    try:
        # Asegura que el modelo esté disponible (simulando la descarga del bucket)
        download_model()
    except Exception as e:
        pytest.skip(f"No se pudo completar el setup del modelo: {e}")
        
    # Crea un archivo JSON de prueba simulado si no existe (simula la descarga del warehouse)
    if not os.path.exists(TEST_DATA_PATH):
        # Crear un array de entrada simulado (imagen 224x224x3)
        simulated_input = np.random.rand(INPUT_SIZE[0], INPUT_SIZE[1], 3).tolist()
        simulated_data = {
            "input_tensor": simulated_input,
            "expected_index": 285, # 'tabby cat' (ejemplo de predicción esperada)
            "baseline_confidence": 0.85 # Métrica de referencia para la prueba de regresión
        }
        os.makedirs(os.path.dirname(TEST_DATA_PATH), exist_ok=True)
        with open(TEST_DATA_PATH, 'w') as f:
            json.dump(simulated_data, f)
        print(f"✅ Datos de prueba simulados generados en: {TEST_DATA_PATH}")

# --- Fixtures de Pytest ---

@pytest.fixture(scope="module")
def ort_session():
    """Carga la sesión ONNX una vez para todas las pruebas."""
    return onnxruntime.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])

@pytest.fixture(scope="module")
def test_data():
    """Obtiene los datos de prueba del 'warehouse' (archivo local)."""
    with open(TEST_DATA_PATH, 'r') as f:
        return json.load(f)

# --- Pruebas Requeridas por la Rúbrica ---

def test_model_responds_with_defined_input(ort_session, test_data):
    """
    PRUEBA 1: Probar que el modelo responde con datos de entrada definidos.
    Verifica que la predicción coincida con la etiqueta esperada (ej. 'tabby cat').
    """
    input_data = np.array(test_data["input_tensor"], dtype=np.float32)
    # El modelo espera (1, 3, 224, 224), por lo que necesitamos preprocesar:
    input_tensor = input_data.transpose([2, 0, 1])
    input_tensor = np.expand_dims(input_tensor, axis=0)

    input_name = ort_session.get_inputs()[0].name
    ort_inputs = {input_name: input_tensor}
    ort_outputs = ort_session.run(None, ort_inputs)
    output_data = ort_outputs[0]

    predicted_index = np.argmax(output_data).item()
    
    # Comprobar que el índice de predicción sea el esperado o esté cerca.
    assert predicted_index == test_data["expected_index"], "La predicción no coincidió con el índice esperado."

def test_no_significant_metric_change(ort_session, test_data):
    """
    PRUEBA 2: Probar que no existe un cambio significativo en alguna métrica definida.
    Verifica que la confianza (métrica de calidad) no caiga por debajo de la línea base.
    """
    input_data = np.array(test_data["input_tensor"], dtype=np.float32)
    input_tensor = input_data.transpose([2, 0, 1])
    input_tensor = np.expand_dims(input_tensor, axis=0)

    input_name = ort_session.get_inputs()[0].name
    ort_inputs = {input_name: input_tensor}
    ort_outputs = ort_session.run(None, ort_inputs)
    output_data = ort_outputs[0]

    # Calcular la confianza para la clase esperada
    exp_output = np.exp(output_data - np.max(output_data))
    probabilities = exp_output / np.sum(exp_output)
    
    predicted_confidence = probabilities[0, test_data["expected_index"]].item()
    baseline = test_data["baseline_confidence"]
    
    # La nueva confianza NO debe ser menor al 5% de la línea base (ejemplo de umbral)
    threshold = baseline * 0.95 
    
    assert predicted_confidence >= threshold, (
        f"Regresión de métrica detectada. Confianza actual ({predicted_confidence:.4f}) "
        f"cayó por debajo del umbral ({threshold:.4f} basado en {baseline:.4f})."
    )