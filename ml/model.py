import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from data.features import construir_features

MODEL_PATH = "models/modelo.pkl"


def normalizar_columnas(df):
    df.columns = [col.lower().strip() for col in df.columns]

    # Mapear posibles nombres
    mapping = {
        "home_odds": ["home_odds", "odd_home", "cuota_local"],
        "away_odds": ["away_odds", "odd_away", "cuota_visitante"],
        "resultado": ["resultado", "result", "target"]
    }

    columnas = {}

    for key, posibles in mapping.items():
        for col in posibles:
            if col in df.columns:
                columnas[key] = col
                break

    return columnas


def entrenar_modelo():
    df = pd.read_csv("data/historico.csv")

    columnas = normalizar_columnas(df)

    if len(columnas) < 3:
        print("❌ CSV mal formado. Usando datos sintéticos...")

        X = np.array([
            [0.6, 0.4, 0.2],
            [0.4, 0.6, -0.2],
            [0.7, 0.3, 0.4],
            [0.5, 0.5, 0.0]
        ])
        y = [1, 0, 1, 0]

    else:
        X = []
        y = []

        for _, row in df.iterrows():
            try:
                partido = {
                    "home_odds": float(row[columnas["home_odds"]]),
                    "away_odds": float(row[columnas["away_odds"]])
                }

                X.append(construir_features(partido))
                y.append(int(row[columnas["resultado"]]))

            except:
                continue

        if len(X) == 0:
            print("❌ Sin datos válidos. Usando fallback ML")
            X = [[0.6, 0.4, 0.2], [0.4, 0.6, -0.2]]
            y = [1, 0]

    modelo = RandomForestClassifier(n_estimators=100)
    modelo.fit(X, y)

    os.makedirs("models", exist_ok=True)
    joblib.dump(modelo, MODEL_PATH)

    print("✅ Modelo entrenado y guardado")

    return modelo


def cargar_modelo():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None


def predecir(modelo, partido):
    features = construir_features(partido).reshape(1, -1)
    prob = modelo.predict_proba(features)[0][1]
    return prob
