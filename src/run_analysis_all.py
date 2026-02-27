import pandas as pd
import os
from glob import glob
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# -----------------------------
# Configuración
# -----------------------------
processed_folder = r"C:\Oraculum_Systems\05_Products\Oraculum-Industrial-Soft-Sensor\data\processed"
output_folder = r"C:\Oraculum_Systems\05_Products\Oraculum-Industrial-Soft-Sensor\data\results"

voltage = 380          # Voltaje de operación
cost_hour = 5000       # Costo por hora de parada por anomalía
window_size = 100      # Ventana de rolling para suavizar

# Crear carpeta de resultados si no existe
os.makedirs(output_folder, exist_ok=True)

# Buscar todos los CSV procesados
csv_files = glob(os.path.join(processed_folder, "*.csv"))
if not csv_files:
    print("No se encontraron CSVs en:", processed_folder)
    exit()

# -----------------------------
# Función de preprocesamiento
# -----------------------------
def preprocess(df, sensor_type='current', max_value=100, window_size=100):
    """
    Normaliza y aplica rolling window a la columna del sensor.
    """
    if sensor_type not in df.columns:
        print(f"Advertencia: no se encontró columna {sensor_type}")
        return df

    # Normalizar a 0-max_value
    df[sensor_type] = (df[sensor_type] - df[sensor_type].min()) / (df[sensor_type].max() - df[sensor_type].min()) * max_value

    # Rolling window
    df[f'{sensor_type}_smooth'] = df[sensor_type].rolling(window=window_size, min_periods=1).mean()
    
    return df

# -----------------------------
# Función para procesar cada CSV
# -----------------------------
def process_csv(csv_file, voltage=380, cost_hour=5000, window_size=100, sensor_type='current'):
    df = pd.read_csv(csv_file)

    # Preprocesamiento
    df = preprocess(df, sensor_type=sensor_type, max_value=100, window_size=window_size)

    # Detectar anomalías
    model = IsolationForest(contamination=0.01, random_state=42)
    df['anomaly'] = model.fit_predict(df[[f'{sensor_type}_smooth']])
    df['anomaly'] = df['anomaly'].map({1:0, -1:1})

    total_anomalies = df['anomaly'].sum()
    estimated_loss = (total_anomalies / len(df)) * cost_hour

    # Guardar CSV con anomalías
    result_csv = os.path.join(output_folder, os.path.basename(csv_file))
    df.to_csv(result_csv, index=False)

    # Guardar gráfico
    plt.figure(figsize=(12,4))
    plt.plot(df[sensor_type], label=sensor_type)
    plt.plot(df[f'{sensor_type}_smooth'], color='orange', label=f'{sensor_type}_smooth')
    for idx in df.index[df['anomaly']==1]:
        plt.axvline(idx, color='red', linestyle='--', alpha=0.5)
    plt.title(f"{os.path.basename(csv_file)} - Anomalías detectadas")
    plt.legend()
    plt.grid(True)
    plot_file = os.path.join(output_folder, os.path.basename(csv_file).replace('.csv','.png'))
    plt.savefig(plot_file)
    plt.close()

    # Generar informe tipo txt
    report_file = os.path.join(output_folder, os.path.basename(csv_file).replace('.csv','_report.txt'))
    with open(report_file,'w') as f:
        f.write(f"Archivo: {os.path.basename(csv_file)}\n")
        f.write(f"Sensor: {sensor_type}\n")
        f.write(f"Total de muestras: {len(df)}\n")
        f.write(f"Anomalías detectadas: {total_anomalies}\n")
        f.write(f"Pérdida económica estimada: ${estimated_loss:.2f}\n")
        f.write(f"Gráfico generado: {plot_file}\n")
        f.write(f"CSV con anomalías: {result_csv}\n")

    print(f"{os.path.basename(csv_file)}: {total_anomalies} anomalías, pérdida estimada: ${estimated_loss:.2f}")
    return total_anomalies, estimated_loss, plot_file, result_csv, report_file

# -----------------------------
# Procesar todos los CSV
# -----------------------------
summary = []

for csv_file in csv_files:
    df_sample = pd.read_csv(csv_file, nrows=1)
    # Detectar automáticamente tipo de sensor
    if 'current' in df_sample.columns:
        sensor = 'current'
    elif 'vibration' in df_sample.columns:
        sensor = 'vibration'
    else:
        sensor = df_sample.columns[0]  # primera columna por defecto

    anomalies, loss, plot_file, result_csv, report_file = process_csv(
        csv_file,
        voltage=voltage,
        cost_hour=cost_hour,
        window_size=window_size,
        sensor_type=sensor
    )

    summary.append({
        'file': os.path.basename(csv_file),
        'sensor': sensor,
        'anomalies_detected': anomalies,
        'estimated_loss': loss,
        'plot_file': plot_file,
        'result_csv': result_csv,
        'report_file': report_file
    })

# -----------------------------
# Guardar resumen general
# -----------------------------
summary_df = pd.DataFrame(summary)
summary_df.to_csv(os.path.join(output_folder, "summary_report.csv"), index=False)

print("\n¡Pipeline completo! Revisa la carpeta results para gráficos, CSVs e informes.")