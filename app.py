import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import platform

# --- Configuración de la página ---
st.set_page_config(page_title="Reconocimiento de Imágenes", page_icon="📷", layout="centered")

# --- Estilo oscuro elegante ---
st.markdown("""
    <style>
    body { background-color: #0e1117; color: white; }
    .stApp { background-color: #0e1117; }
    h1, h2, h3, h4, h5, h6 { color: #FAFAFA !important; }
    .stButton>button { background-color: #262730; color: white; border-radius: 8px; }
    .stButton>button:hover { background-color: #444654; }
    </style>
""", unsafe_allow_html=True)

# --- Información del sistema ---
st.sidebar.title("⚙️ Opciones")
st.sidebar.info("Usa un modelo exportado desde **Teachable Machine** (formato `.h5`). "
                "Toma una foto con la cámara para predecir la categoría.")
st.sidebar.markdown("---")
st.sidebar.write("🐍 Python versión:", platform.python_version())

# --- Función para cargar el modelo ---
@st.cache_resource
def cargar_modelo():
    try:
        custom_objects = {"KerasLayer": tf.keras.layers.Layer}
        model = tf.keras.models.load_model(
            "keras_model.h5",
            custom_objects=custom_objects,
            compile=False
        )
        st.sidebar.success("✅ Modelo cargado correctamente")
        return model
    except Exception as e:
        st.sidebar.error("❌ Error al cargar el modelo. Verifica el archivo `.h5`")
        st.sidebar.text(str(e))
        return None

model = cargar_modelo()

# --- Encabezado principal ---
st.title("📷 Reconocimiento de Imágenes con Teachable Machine")
st.caption("Sube una imagen o toma una foto para probar el modelo entrenado.")

# Imagen de portada
st.image("OIG5.jpg", width=350, caption="Ejemplo de reconocimiento")

# --- Captura de imagen desde cámara ---
img_file_buffer = st.camera_input("Toma una foto con tu cámara")

if img_file_buffer is not None and model is not None:
    # Convertir la imagen a formato numpy
    img = Image.open(img_file_buffer).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img)

    # Normalizar
    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # --- Predicción ---
    prediction = model.predict(data)
    st.subheader("🔍 Resultado de la predicción:")
    st.write(prediction)

    # Mostrar etiquetas más probables
    try:
        labels = [line.strip() for line in open("labels.txt", "r").readlines()]
        top_label = labels[np.argmax(prediction)]
        confidence = np.max(prediction) * 100
        st.success(f"✅ Predicción: **{top_label}** con **{confidence:.2f}%** de confianza.")
    except:
        st.warning("⚠️ No se encontró `labels.txt`, mostrando resultados numéricos.")

elif model is None:
    st.error("⚠️ No se pudo cargar el modelo. Revisa el archivo `keras_model.h5` en tu carpeta del proyecto.")
else:
    st.info("📸 Esperando una imagen...")
