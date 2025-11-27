# 1. Usar una imagen base de Python oficial (ligera y adecuada)
FROM python:3.10-slim

# 2. Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# 3. Copiar solo los requisitos y el script de descarga para la etapa de setup
COPY requirements.txt .
COPY app/download_model.py app/

# 4. Instalar las dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# 5. Descargar el modelo ONNX del "bucket" (simulación)
# NOTA: El archivo 'mobilenetv2-7.onnx' DEBE estar en la raíz del contexto de Docker build
COPY mobilenetv2-7.onnx . 
RUN python app/download_model.py 

# 6. Copiar el resto de la aplicación y las etiquetas
COPY app/app.py app/
COPY templates templates
COPY imagenet_classes.txt .

# 7. Exponer el puerto
EXPOSE 8080

# 8. Comando para ejecutar la aplicación, definiendo el entorno por defecto
# La variable ENVIRONMENT será sobreescrita por GitHub Actions en el despliegue
ENV ENVIRONMENT=local
CMD ["python", "app/app.py"]