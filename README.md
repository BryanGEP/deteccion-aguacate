# Detección de Enfermedades en Aguacate

Aplicación de visión computacional que detecta enfermedades en aguacates a partir de una imagen. Clasifica el fruto en **Sano (Healthy)**, **Roña (Scab)** o **Antracnosis (Anthracnose)** usando una red neuronal convolucional con Transfer Learning (VGG16).

Instituto Tecnológico de Jiquilpan (TecNM) — Ingeniería en Sistemas Computacionales.

## ¿Cómo funciona?

Subes una foto de un aguacate; el backend le quita el fondo automáticamente con **rembg** (basado en la red U²-Net), recorta y centra el fruto sobre fondo blanco para que se parezca a las imágenes con las que se entrenó, y el modelo predice la enfermedad junto con las probabilidades por categoría. La aplicación fue desplegada en Google Cloud.

## Estructura

- **v1/** — Primera versión del servicio (backend FastAPI, Dockerfile, interfaz y modelo).
- **v2/** — Versión posterior del servicio.
- **entrenamiento/** — Notebooks del análisis y entrenamiento del modelo, junto con el archivo de etiquetas del dataset.
- **documentacion/** — Guía técnica del proyecto y presentación.

Cada versión de la app (`v1`, `v2`) contiene su `main.py`, su `Dockerfile`, su `index.html`, sus dependencias y el modelo `modeloV4_aguacate.keras`.

## Dataset

El modelo fue entrenado con imágenes de aguacates obtenidas de **Kaggle**, procesadas para quitarles el fondo. Las **imágenes no se incluyen** en este repositorio por su tamaño; sí se incluye el archivo de etiquetas en la carpeta `entrenamiento`.

## Cómo ejecutar

Con Docker instalado, entra a la versión que quieras probar (`v1` o `v2`) y ejecuta:

```bash
docker build -t deteccion-aguacate .
docker run -p 8080:8080 deteccion-aguacate
```

Luego abre el navegador en `http://localhost:8080` (ajusta el puerto al que exponga el `Dockerfile` si es distinto).

> Nota: el modelo `modeloV4_aguacate.keras` se almacena con **Git LFS** por su tamaño. Para clonarlo completo necesitas tener Git LFS instalado.

## Tecnologías

- Python · FastAPI
- TensorFlow / Keras (CNN + Transfer Learning con VGG16)
- rembg / U²-Net (remoción de fondo)
- HTML, CSS, JavaScript
- Docker
- Google Cloud (despliegue)
- Git LFS (para el modelo)

## Créditos

- Dataset de imágenes de aguacate: Kaggle
- Remoción de fondo: [rembg](https://github.com/danielgatis/rembg) (red U²-Net)

## Autor

**Bryan Gael Esquivel Pérez** — [@BryanGEP](https://github.com/BryanGEP)
