FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-descargar el modelo U2Net de rembg durante el build
# para que el primer request no tenga demora
RUN python -c "from rembg import new_session; new_session('u2net'); print('U2Net descargado OK')"

COPY main.py .
COPY index.html .
COPY modeloV4_aguacate.keras .

ENV MODEL_PATH=/app/modeloV4_aguacate.keras

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]