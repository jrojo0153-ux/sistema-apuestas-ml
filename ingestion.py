
import pandas as pd
import datetime
import os
import glob

def run_ingestion():
    print(f"Iniciando ingesta de datos históricos y en vivo: {datetime.datetime.now()}")
    
    # Process all CSV files available in the root
    csv_files = glob.glob('*.csv')
    print(f"Archivos encontrados para procesar: {csv_files}")
    
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            print(f"Procesado {file}: {len(df)} filas cargadas.")
            # Aquí se puede añadir lógica de limpieza o unión de datos
        except Exception as e:
            print(f"Error procesando {file}: {e}")

    print("Datos ingeridos y procesados correctamente.")

if __name__ == '__main__':
    run_ingestion()
