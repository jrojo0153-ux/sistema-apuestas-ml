
import pandas as pd
import datetime
import os
import glob

def run_ingestion():
    print(f"Iniciando ingesta estructurada: {datetime.datetime.now()}")
    data_dir = 'data'
    
    if not os.path.exists(data_dir):
        print(f"Error: Directorio {data_dir} no encontrado.")
        return

    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    print(f"Archivos encontrados en {data_dir}: {csv_files}")
    
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            print(f"[OK] {os.path.basename(file)}: {len(df)} filas.")
        except Exception as e:
            print(f"[ERROR] {file}: {e}")

    print("Estructura de datos procesada correctamente.")

if __name__ == '__main__':
    run_ingestion()
