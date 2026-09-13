from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
from PIL import Image
import io
import os

app = FastAPI(
    title="🥑 Avocado Disease Classifier API",
    description="Detecta enfermedades en aguacates usando CNN con Transfer Learning (VGG16)",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Clases en orden alfabético (LabelEncoder las ordena así)
# Anthracnose=0, Healthy=1, Scab=2
CLASES = ['Anthracnose', 'Healthy', 'Scab']
CLASES_ES = {
    'Anthracnose': 'Antracnosis',
    'Healthy':     'Sano',
    'Scab':        'Roña'
}
EMOJIS = {
    'Anthracnose': '🍂',
    'Healthy':     '✅',
    'Scab':        '🟤'
}

IMG_SIZE = (224, 224)
MODEL_PATH = os.getenv("MODEL_PATH", "/app/modeloV4_aguacate.keras")

print(f"Cargando modelo desde: {MODEL_PATH}")
try:
    model = load_model(MODEL_PATH)
    print("Modelo cargado correctamente")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    model = None


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Preprocesa igual que en el notebook: resize 224x224, normalizar /255."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = img_to_array(img)
    arr = arr / 255.0
    arr = np.expand_dims(arr, axis=0)   # (1, 224, 224, 3)
    return arr


@app.get("/", response_class=FileResponse, summary="Frontend")
def root():
    return FileResponse("/app/index.html", media_type="text/html")


@app.get("/health", summary="Health check")
def health():
    return {"status": "ok", "model_loaded": model is not None, "clases": CLASES}


@app.post("/predict", summary="Detectar enfermedad en aguacate")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(status_code=400, detail="Solo se aceptan imágenes JPG, PNG o WEBP.")

    if model is None:
        raise HTTPException(status_code=503, detail="El modelo no está disponible. Verifica que modeloV4_aguacate.keras exista.")

    image_bytes = await file.read()
    try:
        img_array = preprocess_image(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"No se pudo procesar la imagen: {e}")

    predicciones = model.predict(img_array)

    # Soporte para SavedModel (salida como dict) o Keras normal
    if isinstance(predicciones, dict):
        scores = list(predicciones.values())[0][0]
    else:
        scores = predicciones[0]

    clase_idx  = int(np.argmax(scores))
    clase_en   = CLASES[clase_idx]
    confianza  = float(scores[clase_idx])

    probabilidades = {
        CLASES[i]: round(float(scores[i]) * 100, 2)
        for i in range(len(CLASES))
    }

    return JSONResponse({
        "prediccion":    clase_en,
        "prediccion_es": CLASES_ES[clase_en],
        "emoji":         EMOJIS[clase_en],
        "confianza":     round(confianza * 100, 2),
        "probabilidades": probabilidades
    })