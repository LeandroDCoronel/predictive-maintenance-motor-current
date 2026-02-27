import scipy.io
import pandas as pd
import os

# -----------------------------
# Configura aquí tu archivo .mat
# -----------------------------
input_mat_file = r"C:\Oraculum_Systems\05_Products\Oraculum-Industrial-Soft-Sensor\data\raw\97.mat"
output_csv_file = r"C:\Oraculum_Systems\05_Products\Oraculum-Industrial-Soft-Sensor\data\processed\97.csv"

# -----------------------------
# Cargar archivo .mat
# -----------------------------
mat = scipy.io.loadmat(input_mat_file)

# Encontrar la clave que contiene los datos (CWRU tiene varias)
data_key = [key for key in mat.keys() if not key.startswith('__')][0]
data_array = mat[data_key]

# Convertir a DataFrame
# Si es 1D → una columna, si es 2D → varias columnas
if data_array.ndim == 1:
    df = pd.DataFrame(data_array, columns=['current'])
else:
    # Para CWRU, normalmente cada columna es un canal de sensor
    df = pd.DataFrame(data_array)

# Crear carpeta de salida si no existe
os.makedirs(os.path.dirname(output_csv_file), exist_ok=True)

# Guardar CSV
df.to_csv(output_csv_file, index=False)
print(f"Archivo convertido a CSV y guardado en: {output_csv_file}")