
# =====================================================
# FUNCIONES PARA PROCESAMIENTO EN EL DOMINIO
# DE LA FRECUENCIA
# Alumno: Daniel Nieves
# Numero de lista: 45
# =====================================================

import numpy as np
import matplotlib.pyplot as plt
from skimage import data
import os

# Obtener la ruta de la carpeta de resultados
RUTA_RESULTADOS = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "resultados-img-1343"
)

# Crear carpeta si no existe
os.makedirs(RUTA_RESULTADOS, exist_ok=True)


# 1. Funcion para cargar la imagen
def cargar_imagen(nombre_imagen=None):
    imagen = data.camera()
    return imagen


# 2. Funcion para obtener el espectro de frecuencia
def obtener_espectro(imagen):
    transformada = np.fft.fft2(imagen)

    espectro = np.fft.fftshift(transformada)

    espectro = 20 * np.log(np.abs(espectro) + 1)

    return espectro


# 3. Funcion para guardar las graficas
def guardar_grafica(figura, nombre):
    ruta = os.path.join(RUTA_RESULTADOS, nombre)

    figura.savefig(ruta, dpi=300, bbox_inches="tight")

    print(f"Imagen guardada correctamente: {ruta}")



# 4. Funcion para aplicar filtros en el dominio de frecuencia
def aplicar_filtro_frecuencia(imagen, kernel):

    alto, ancho = imagen.shape
    filas, columnas = kernel.shape

    # Preparar matrices para convolucion lineal
    nuevo_alto = alto + filas - 1
    nuevo_ancho = ancho + columnas - 1

    imagen_pad = np.zeros((nuevo_alto, nuevo_ancho))
    imagen_pad[:alto, :ancho] = imagen

    # Invertir el kernel para realizar correlacion
    kernel_invertido = np.flip(kernel)

    filtro_pad = np.zeros((nuevo_alto, nuevo_ancho))
    filtro_pad[:filas, :columnas] = kernel_invertido

    # Transformadas de Fourier
    fft_imagen = np.fft.fft2(imagen_pad)
    fft_filtro = np.fft.fft2(filtro_pad)

    # Multiplicacion en frecuencia
    resultado_fft = fft_imagen * fft_filtro

    # Transformada inversa
    resultado = np.fft.ifft2(resultado_fft).real

    # Recortar al tamaño original
    resultado = resultado[
        filas // 2:filas // 2 + alto,
        columnas // 2:columnas // 2 + ancho
    ]

    return resultado
