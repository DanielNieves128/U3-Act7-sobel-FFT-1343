
# =====================================================
# EJEMPLO 4: MAGNITUD DEL GRADIENTE
# Alumno: Daniel Nieves
# Numero de lista: 45
# =====================================================

import numpy as np
import matplotlib.pyplot as plt

from funciones_frecuencia_1343 import (
    cargar_imagen,
    guardar_grafica,
    aplicar_filtro_frecuencia
)

# 1. Cargar imagen original
imagen = cargar_imagen('image.png')

# 2. Crear matriz Sobel X
sobel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=float)

# 3. Crear matriz Sobel Y
sobel_y = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
], dtype=float)

# 4. Aplicar filtro Sobel X en frecuencia
resultado_x = aplicar_filtro_frecuencia(imagen, sobel_x)

# 5. Aplicar filtro Sobel Y en frecuencia
resultado_y = aplicar_filtro_frecuencia(imagen, sobel_y)

# 6. Calcular magnitud del gradiente
magnitud = np.sqrt(
    resultado_x**2 + resultado_y**2
)

# 7. Normalizar resultado para visualizar
magnitud_normalizada = (
    magnitud / magnitud.max() * 255
).astype(np.uint8)

# 8. Mostrar informacion
print("--- EJEMPLO 4: MAGNITUD DEL GRADIENTE ---")
print("Imagen procesada correctamente")
print("Dimensiones:", imagen.shape)
print("Magnitud maxima:", magnitud.max())

# 9. Crear figura con cuatro imagenes
figura, ejes = plt.subplots(2, 2, figsize=(12, 10))

# Imagen original
ejes[0, 0].imshow(imagen, cmap='gray')
ejes[0, 0].set_title("Imagen Original")
ejes[0, 0].axis('off')

# Resultado Sobel X
ejes[0, 1].imshow(
    np.abs(resultado_x),
    cmap='gray',
    vmin=0,
    vmax=np.abs(resultado_x).max()
)
ejes[0, 1].set_title("Filtro Sobel X")
ejes[0, 1].axis('off')

# Resultado Sobel Y
ejes[1, 0].imshow(
    np.abs(resultado_y),
    cmap='gray',
    vmin=0,
    vmax=np.abs(resultado_y).max()
)
ejes[1, 0].set_title("Filtro Sobel Y")
ejes[1, 0].axis('off')

# Magnitud del gradiente
ejes[1, 1].imshow(magnitud_normalizada, cmap='gray')
ejes[1, 1].set_title("Magnitud del Gradiente")
ejes[1, 1].axis('off')

# 10. Ajustar espacios
plt.tight_layout()

# 11. Guardar resultado
guardar_grafica(
    figura,
    "04_magnitud_gradiente.png"
)

# 12. Mostrar resultados
plt.show()
