import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from data.features import construir_features
import pandas as pd

MODEL_PATH = "models/modelo.pkl"

def entrenar_modelo():
    df = pd.read_csv("data/historico.csv")

    X = []
    y = []

    for _, row in df.iterrows():
        partido = {
            "home_odds": row["home_odds"],
            "away_odds": row["away_odds"]
        }

        X.append(construir_features(partido))
        y.append(row["resultado"])

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
