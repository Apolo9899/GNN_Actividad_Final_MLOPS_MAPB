FROM python:3.10-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements primero para aprovechar caché de capas
COPY requirements.txt .

# Instalar PyTorch CPU (más ligero para el servicio de inferencia)
RUN pip install --no-cache-dir torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir torch_geometric && \
    pip install --no-cache-dir -r requirements.txt

# Copiar código fuente y modelo
COPY src/ ./src/
COPY models/ ./models/

# Variables de entorno
ENV MODEL_PATH=/app/models/checkpoint_final.pt
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "src.inference_api:app", "--host", "0.0.0.0", "--port", "8000"]
