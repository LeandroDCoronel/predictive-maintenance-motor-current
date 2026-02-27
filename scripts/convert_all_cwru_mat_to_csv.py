import scipy.io
import pandas as pd
import os
from glob import glob

# Carpeta de entrada y salida
raw_folder = r"C:\Oraculum_Systems\05_Products\Oraculum-Industrial-Soft-Sensor\data\raw"
processed_folder = r"C:\Oraculum_Systems\05_Products\Oraculum-Industrial-Soft-Sensor\data\processed"

# Crear carpeta de salida si no existe
os.makedirs(processed_folder, exist_ok=True)

# Buscar todos los archivos .mat
mat_files = glob(os.path.join(raw_folder, "*.mat"))

if not mat_files:
    print("No se encontraron archivos .mat en:", raw_folder)
else:
    print(f"Se encontraron {len(mat_files)} archivos .mat. Convirtiendo...")

for mat_file in mat_files:
    # Cargar archivo .mat
    mat = scipy.io.loadmat(mat_file)
    
    # Encontrar la clave de datos (CWRU normalmente tiene solo 1 clave principal)
    data_key = [key for key in mat.keys() if not key.startswith('__')][0]
    data_array = mat[data_key]
    
    # Convertir a DataFrame
    if data_array.ndim == 1:
        df = pd.DataFrame(data_array, columns=['current'])
    else:
        df = pd.DataFrame(data_array)
    
    # Guardar CSV
    base_name = os.path.splitext(os.path.basename(mat_file))[0]
    output_csv_file = os.path.join(processed_folder, f"{base_name}.csv")
    df.to_csv(output_csv_file, index=False)
    print(f"{mat_file} → {output_csv_file}")

print("¡Conversión completa de todos los archivos .mat a CSV!")