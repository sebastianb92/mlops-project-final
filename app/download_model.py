import os
import shutil

# --- CONFIGURACIÓN DE ALOJAMIENTO EXTERNO SIMULADO ---
# En un entorno real, este sería el URI de un bucket S3, GCS o Azure Blob.
# Para esta actividad, asumimos que el modelo existe en la raíz local para simular la descarga.
MODEL_SOURCE_PATH = "mobilenetv2-7.onnx"
MODEL_DESTINATION_PATH = "app/mobilenetv2-7.onnx"

def download_model():
    """
    Simula la descarga del modelo ONNX desde el almacenamiento externo.
    En CI/CD real, se usaría boto3, gcloud, o un cliente similar.
    """
    print("--- 🚀 SIMULACIÓN DE DESCARGA DE MODELO ---")
    if os.path.exists(MODEL_SOURCE_PATH):
        # Crear el directorio 'app' si no existe
        os.makedirs(os.path.dirname(MODEL_DESTINATION_PATH), exist_ok=True)
        # Simular la descarga/copia
        shutil.copy(MODEL_SOURCE_PATH, MODEL_DESTINATION_PATH)
        print(f"✅ Modelo '{MODEL_SOURCE_PATH}' 'descargado' correctamente a '{MODEL_DESTINATION_PATH}'.")
    else:
        print(f"❌ ERROR: El modelo '{MODEL_SOURCE_PATH}' no se encontró en la raíz del proyecto local.")
        print("Asegúrese de que el archivo .onnx exista para simular la descarga.")
        # Se detiene la ejecución si el modelo no está disponible
        raise FileNotFoundError(f"Modelo no encontrado en {MODEL_SOURCE_PATH}")

if __name__ == "__main__":
    download_model()