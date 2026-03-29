import pandas as pd
import os

RUTA = "data/historico.csv"


def guardar_resultado(partido, cuota, prob, resultado):
    fila = pd.DataFrame([{
        "partido": partido,
        "cuota": cuota,
        "prob": prob,
        "resultado": resultado
    }])

    if os.path.exists(RUTA):
        df = pd.read_csv(RUTA)
        df = pd.concat([df, fila], ignore_index=True)
    else:
        df = fila

    df.to_csv(RUTA, index=False)


def cargar_historico():
    if os.path.exists(RUTA):
        return pd.read_csv(RUTA)
    return None
