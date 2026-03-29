import pandas as pd
import joblib
import os

from sklearn.ensemble import RandomForestClassifier

MODEL_PATH = "models/modelo.pkl"


def entrenar_modelo():
    if not os.path.exists("data/historico.csv"):
        print("⚠️ No hay histórico, usando modelo base")
        return None

    df = pd.read_csv("data/historico.csv")

    if df.empty:
        return None

    X = df[["cuota_local", "cuota_visitante"]]
    y = df["resultado"]

    modelo = RandomForestClassifier(n_estimators=100)
    modelo.fit(X, y)

    joblib.dump(modelo, MODEL_PATH)

    print("✅ Modelo entrenado y guardado")

    return modelo


def cargar_modelo():
    if os.path.exists(MODEL_PATH):
        print("📦 Cargando modelo existente...")
        return joblib.load(MODEL_PATH)

    return entrenar_modelo()


def predecir(modelo, cuota_local, cuota_visitante):
    if modelo is None:
        return 0.5

    X = [[cuota_local, cuota_visitante]]
    prob = modelo.predict_proba(X)[0][1]

    return prob
