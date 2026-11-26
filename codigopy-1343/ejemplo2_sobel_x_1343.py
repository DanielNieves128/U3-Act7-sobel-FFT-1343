
# =====================================================
# EJEMPLO 2: FILTRO SOBEL X EN EL DOMINIO DE FRECUENCIA
# Alumno: Daniel Nieves
# Numero de lista: 45
# =====================================================

import numpy as np
import matplotlib.pyplot as plt

from funciones_frecuencia_1343 import (
    cargar_imagen,
    guardar_grafica
)

# 1. Cargar imagen original
imagen = cargar_imagen('image.png')

# 2. Obtener dimensiones de la imagen
alto, ancho = imagen.shape

# 3. Transformada de Fourier de la imagen
fft_imagen = np.fft.fft2(imagen)

# 4. Crear filtro Sobel X
sobel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=float)

# 5. Ajustar el filtro al tamaño de la imagen
filtro = np.zeros((alto, ancho))
filtro[:3, :3] = sobel_x

# 6. Transformar el filtro al dominio de frecuencia
fft_filtro = np.fft.fft2(filtro)

# 7. Aplicar el filtro en frecuencia
resultado_fft = fft_imagen * fft_filtro

# 8. Regresar al dominio espacial
resultado = np.fft.ifft2(resultado_fft)

# 9. Obtener magnitud del resultado
resultado = np.abs(resultado)

# 10. Mostrar resultados
print("--- EJEMPLO 2: SOBEL X ---")
print(f"Alto: {alto}")
print(f"Ancho: {ancho}")

figura, ejes = plt.subplots(1, 2, figsize=(12, 6))

ejes[0].imshow(imagen, cmap='gray')
ejes[0].set_title("Imagen Original")
ejes[0].axis('off')

ejes[1].imshow(resultado, cmap='gray')
ejes[1].set_title("Filtro Sobel X")
ejes[1].axis('off')

plt.tight_layout()

# 11. Guardar resultado
guardar_grafica(figura, "02_sobel_x.png")

plt.show()
