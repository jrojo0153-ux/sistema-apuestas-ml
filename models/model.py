import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

RUTA_MODELO = "models/modelo.pkl"


def entrenar_modelo():
    """
    Entrena modelo con datos históricos simples
    """

    # 🔥 DATASET EJEMPLO (puedes ampliar luego)
    data = {
        "cuota": [1.8, 2.1, 1.5, 2.5, 1.9, 2.2, 1.7, 2.8],
        "prob_estimada": [0.6, 0.55, 0.7, 0.4, 0.58, 0.52, 0.62, 0.35],
        "resultado": [1, 0, 1, 0, 1, 0, 1, 0]  # 1 = gana, 0 = pierde
    }

    df = pd.DataFrame(data)

    X = df[["cuota", "prob_estimada"]]
    y = df["resultado"]

    modelo = RandomForestClassifier(n_estimators=100)
    modelo.fit(X, y)

    guardar_modelo(modelo)

    return modelo


def guardar_modelo(modelo):
    os.makedirs("models", exist_ok=True)
    joblib.dump(modelo, RUTA_MODELO)


def cargar_modelo():
    if os.path.exists(RUTA_MODELO):
        return joblib.load(RUTA_MODELO)
    return None


def predecir_probabilidades(modelo, cuotas):
    """
    Genera probabilidades reales
    """

    import numpy as np

    probs = []

    for cuota in cuotas:
        X = np.array([[cuota, 1 / cuota]])  # feature simple
        prob = modelo.predict_proba(X)[0][1]
        probs.append(prob)

    return probs
