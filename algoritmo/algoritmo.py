import cv2
import os
import numpy as np
import math



def cargar_imagen(imagen):
    # Cargar la imagen utilizando OpenCV y escala de grises

    if imagen is not None:
        imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        return imagen_gris
    else:
        print("No se pudo cargar la imagen.")
        return None


def calcular_matriz_de_distancias(imagen):
    # Obtener las dimensiones de la imagen
    ALTO,ANCHO = imagen.shape



    # Calcular las distancias euclidianas y almacenarlas en D
    u = np.arange(ANCHO)
    v = np.arange(ALTO)

    # Utilizar la función meshgrid para crear matrices de coordenadas
    U, V = np.meshgrid(u, v)    

    # Calcular la distancia euclidiana de cada punto al origen
    D = np.sqrt((U)**2 + (V)**2)

    return D



def calcular_H_Gaussiana(D, theta):
    # Calcular H(u, v)
    E = math.e
    H =E ** (-((D ** 2) / (2 * (theta ** 2))))

    return H


def calcular_transformada_fourier(imagen):
    return np.fft.fft2(imagen)

def calcular_transformada_inversa(G, H):
   
    F_deconvolucion = np.fft.ifft2(G / H)
    return np.abs(F_deconvolucion).astype(np.uint8)

def guardar_imagen(carpeta, nombre_archivo, imagen):
    ruta_guardar = os.path.join(carpeta, nombre_archivo)
    cv2.imwrite(ruta_guardar, imagen)


    

def restaurar(imagen, sigma):
    imagen=cargar_imagen(imagen)

    if imagen is not None:
        H = calcular_H_Gaussiana(calcular_matriz_de_distancias(imagen),sigma)
        G=calcular_transformada_fourier(imagen)
        resultado=calcular_transformada_inversa(G, H)
        
        return resultado,imagen

def aplicar_sobel(imagen,Ksize):
    sobelx = cv2.Sobel(imagen, cv2.CV_64F, 1, 0, ksize=Ksize)
    sobely = cv2.Sobel(imagen, cv2.CV_64F, 0, 1, ksize=Ksize)
    sobel = np.sqrt(sobelx**2 + sobely**2)

    sobel = np.uint8(sobel)
    return sobel

def aplicar_laplaciano(imagen):
    laplaciano = cv2.Laplacian(imagen, cv2.CV_64F)
    laplaciano = np.uint8(laplaciano)
    return laplaciano
