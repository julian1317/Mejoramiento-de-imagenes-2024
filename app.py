import streamlit as st
from algoritmo import algoritmo
import numpy as np
import cv2
import os


# Ejecutar el comando Streamlit con el puerto especificado

valores_permitidos = [0, 1, 3, 5]
st.title("Sube una imagen borrosa que quieras restaurar")
test_image = st.file_uploader("Escoge una imagen")
sigma = st.slider("Mueve este slider para ajustar la limpieza de tu imagen", 0, 10000, 1000, 10)
ksize = st.select_slider("Mueve este slider para ajustar la limpieza de tu imagen", options=valores_permitidos)

if test_image is not None:
    col1, col2, col3 = st.columns(3)
    test_image = np.asarray(bytearray(test_image.read()), dtype=np.uint8)
    test_image = cv2.imdecode(test_image, 1)
    imagen_restaurada, imagen1 = algoritmo.restaurar(test_image, sigma)
    sobel = algoritmo.aplicar_sobel(imagen_restaurada, ksize)
    
    with col1:
        st.subheader("Imagen original")
        st.image(imagen1)
    with col2:
        st.subheader("Imagen restaurada")
        st.image(imagen_restaurada)
    with col3:
        st.subheader("Imagen con Sobel")
        st.image(sobel)

    # Botón para ver las imágenes en grande
    if st.button("Ver imágenes en grande"):
        # Prepara el área para mostrar las imágenes en grande
        col4, col5 = st.columns(2)
        with col4:
            st.subheader("Imagen original")
            st.image(imagen1, width=400)
        with col5:
            st.subheader("Imagen restaurada")
            st.image(imagen_restaurada, width=400)
