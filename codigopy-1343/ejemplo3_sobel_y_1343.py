
# =====================================================
# EJEMPLO 3: FILTRO SOBEL Y EN FRECUENCIA
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

# 1. Cargar imagen
imagen = cargar_imagen('image.png')

# 2. Crear matriz Sobel Y
sobel_y = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
], dtype=float)

# 3. Aplicar filtro en frecuencia
resultado = aplicar_filtro_frecuencia(imagen, sobel_y)

# 4. Calcular magnitud absoluta
resultado = np.abs(resultado)

# 5. Mostrar informacion
print("--- EJEMPLO 3: SOBEL Y ---")
print("Filtro aplicado correctamente")
print("Dimensiones:", imagen.shape)

# 6. Crear graficas
figura, ejes = plt.subplots(1, 2, figsize=(12, 6))

ejes[0].imshow(imagen, cmap='gray')
ejes[0].set_title("Imagen Original")
ejes[0].axis('off')

ejes[1].imshow(resultado, cmap='gray', vmin=0,
               vmax=resultado.max())
ejes[1].set_title("Filtro Sobel Y")
ejes[1].axis('off')

plt.tight_layout()

# 7. Guardar resultado
guardar_grafica(figura, "03_sobel_y.png")

plt.show()
