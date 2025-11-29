FROM python:3.10-slim

WORKDIR /app

# Copiar requisitos primero
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copiar archivos necesarios
COPY mobilenetv2-7.onnx .
COPY imagenet_classes.txt .

# Copiar carpetas completas
COPY app/ app/
COPY templates/ templates/

# Ejecutar setup del modelo
RUN python app/download_model.py

# Exponer puerto
EXPOSE 8080

ENV ENVIRONMENT=local

CMD ["python", "app/app.py"]