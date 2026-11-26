
# =====================================================
# EJEMPLO 1: IMAGEN ORIGINAL Y ESPECTRO DE FRECUENCIA
# Alumno: Daniel Nieves
# Numero de lista: 45
# =====================================================

import matplotlib.pyplot as plt

from funciones_frecuencia_1343 import (
    cargar_imagen,
    obtener_espectro,
    guardar_grafica
)

# 1. Cargar la imagen
imagen = cargar_imagen('image.png')

# 2. Calcular el espectro de frecuencia
espectro = obtener_espectro(imagen)

# 3. Mostrar dimensiones en consola
print("--- EJEMPLO 1 ---")
print("Imagen del camarografo cargada.")
print(f"Dimensiones: Alto={imagen.shape[0]}, Ancho={imagen.shape[1]}")

# 4. Crear figura con dos graficas
figura, ejes = plt.subplots(1, 2, figsize=(12, 6))

# 5. Mostrar imagen original
ejes[0].imshow(imagen, cmap="gray")
ejes[0].set_title("Imagen Original (Dominio Espacial)")
ejes[0].axis("off")

# 6. Mostrar espectro de frecuencia
ejes[1].imshow(espectro, cmap="gray")
ejes[1].set_title("Espectro de Frecuencia (FFT)")
ejes[1].axis("off")

plt.tight_layout()

# 7. Guardar y mostrar resultados
guardar_grafica(figura, "01_imagen_y_espectro.png")

plt.show()
