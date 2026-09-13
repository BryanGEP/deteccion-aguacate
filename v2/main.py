from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
from PIL import Image
import io
import os
import base64
from rembg import remove, new_session

app = FastAPI(
    title="🥑 Avocado Disease Classifier API",
    description="Detecta enfermedades en aguacates usando CNN con Transfer Learning (VGG16)",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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

IMG_SIZE    = (224, 224)
PADDING_PCT = 0.05   # 5 % de margen alrededor del aguacate tras el crop
MODEL_PATH  = os.getenv("MODEL_PATH", "/app/modeloV4_aguacate.keras")

# ── Cargar modelo ──────────────────────────────────────────────────────────────
print(f"Cargando modelo desde: {MODEL_PATH}")
try:
    model = load_model(MODEL_PATH)
    print("✅ Modelo cargado correctamente")
except Exception as e:
    print(f"❌ Error al cargar el modelo: {e}")
    model = None

# ── Cargar sesión rembg ────────────────────────────────────────────────────────
print("Cargando rembg / U2Net...")
try:
    rembg_session = new_session("u2net")
    print("✅ rembg listo")
except Exception as e:
    print(f"⚠️  rembg no disponible: {e}")
    rembg_session = None


# ── Utilidades ────────────────────────────────────────────────────────────────

def remove_background(image_bytes: bytes) -> Image.Image:
    """
    Quita el fondo con rembg y devuelve imagen RGB con fondo blanco.
    Si rembg no está disponible devuelve la imagen original.
    """
    if rembg_session is None:
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")

    result_bytes = remove(image_bytes, session=rembg_session)
    img_rgba     = Image.open(io.BytesIO(result_bytes)).convert("RGBA")

    background = Image.new("RGBA", img_rgba.size, (255, 255, 255, 255))
    background.paste(img_rgba, mask=img_rgba.split()[3])
    return background.convert("RGB")


def crop_to_subject(img_rgb: Image.Image, padding: float = PADDING_PCT) -> Image.Image:
    """
    Detecta el bounding-box del aguacate (píxeles no-blancos),
    añade un pequeño margen y recorta.
    Luego lo centra en un canvas cuadrado blanco → igual que el dataset.
    """
    arr  = np.array(img_rgb)
    mask = np.any(arr < 250, axis=2)          # píxeles que NO son blancos puros

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not rows.any():                         # imagen completamente blanca (fallback)
        return img_rgb

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    h, w = arr.shape[:2]
    pad_r = int((rmax - rmin) * padding)
    pad_c = int((cmax - cmin) * padding)

    rmin = max(0, rmin - pad_r)
    rmax = min(h, rmax + pad_r)
    cmin = max(0, cmin - pad_c)
    cmax = min(w, cmax + pad_c)

    cropped = img_rgb.crop((cmin, rmin, cmax, rmax))   # PIL: (left, top, right, bottom)

    # Centrar en canvas cuadrado blanco
    side   = max(cropped.width, cropped.height)
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    offset_x = (side - cropped.width)  // 2
    offset_y = (side - cropped.height) // 2
    canvas.paste(cropped, (offset_x, offset_y))

    return canvas


def to_jpeg_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def preprocess_image(img: Image.Image) -> np.ndarray:
    """Resize 224×224 y normalizar /255 — igual que en el notebook."""
    img  = img.resize(IMG_SIZE)
    arr  = img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)


def image_to_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


# ── Rutas ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=FileResponse, summary="Frontend")
def root():
    return FileResponse("/app/index.html", media_type="text/html")


@app.get("/health", summary="Health check")
def health():
    return {
        "status":       "ok",
        "model_loaded": model is not None,
        "rembg_loaded": rembg_session is not None,
        "clases":       CLASES
    }


@app.post("/predict", summary="Detectar enfermedad en aguacate")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(status_code=400, detail="Solo se aceptan imágenes JPG, PNG o WEBP.")

    if model is None:
        raise HTTPException(status_code=503, detail="El modelo no está disponible.")

    image_bytes = await file.read()

    # ── 1. Quitar fondo → imagen RGB con fondo blanco ─────────────────────────
    try:
        img_nobg = remove_background(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Error al remover fondo: {e}")

    # ── 2. Crop inteligente → centrar aguacate en canvas cuadrado ─────────────
    try:
        img_cropped = crop_to_subject(img_nobg)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Error en crop: {e}")

    # ── 3. Convertir a bytes para devolver al frontend ────────────────────────
    processed_bytes = to_jpeg_bytes(img_cropped)

    # ── 4. Preprocesar para el modelo ─────────────────────────────────────────
    try:
        img_array = preprocess_image(img_cropped)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"No se pudo procesar la imagen: {e}")

    # ── 5. Predecir ───────────────────────────────────────────────────────────
    predicciones = model.predict(img_array)

    if isinstance(predicciones, dict):
        scores = list(predicciones.values())[0][0]
    else:
        scores = predicciones[0]

    clase_idx = int(np.argmax(scores))
    clase_en  = CLASES[clase_idx]
    confianza = float(scores[clase_idx])

    probabilidades = {
        CLASES[i]: round(float(scores[i]) * 100, 2)
        for i in range(len(CLASES))
    }

    return JSONResponse({
        "prediccion":       clase_en,
        "prediccion_es":    CLASES_ES[clase_en],
        "emoji":            EMOJIS[clase_en],
        "confianza":        round(confianza * 100, 2),
        "probabilidades":   probabilidades,
        "imagen_procesada": image_to_base64(processed_bytes),
        "fondo_removido":   rembg_session is not None
    })