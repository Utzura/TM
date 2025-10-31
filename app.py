import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Configuración de la página
st.set_page_config(page_title="📸 Clasificador Teachable Machine", page_icon="🤖", layout="centered")

st.markdown("""
    <style>
        body {background-color: #0e1117; color: white;}
        .stButton>button {
            background-color: #2b5cff;
            color: white;
            border: none;
            border-radius: 8px;
            font-weight: bold;
            padding: 8px 16px;
        }
        .stImage img {border-radius: 12px;}
    </style>
""", unsafe_allow_html=True)

# Cargar modelo (usando cache para evitar recargas)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('keras_model.h5', compile=False)
    return model

model = load_model()

# Configuración inicial
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
st.title("🔍 Clasificador de Imágenes")
st.caption("Sube o captura una imagen y el modelo la clasificará según tu entrenamiento en Teachable Machine.")

# Captura desde cámara
img_file_buffer = st.camera_input("📷 Toma una foto o súbela desde tu dispositivo")

if img_file_buffer is not None:
    # Leer imagen
    img = Image.open(img_file_buffer).convert("RGB")
    img_resized = img.resize((224, 224))

    # Normalizar
    image_array = np.asarray(img_resized)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array

    # Predicción
    prediction = model.predict(data)
    class_idx = np.argmax(prediction)
    confidence = prediction[0][class_idx]

    # Mostrar resultados
    st.image(img, caption="📸 Imagen analizada", width=320)
    st.subheader("🧠 Resultado del modelo:")
    if class_idx == 0:
        st.success(f"Izquierda 🫱 — Confianza: {confidence:.2f}")
    elif class_idx == 1:
        st.success(f"Arriba ☝️ — Confianza: {confidence:.2f}")
    else:
        st.warning("No se pudo determinar la clase con alta confianza.")
