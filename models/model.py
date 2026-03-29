import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

MODEL_PATH = "models/model.pkl"

def entrenar_modelo():
    print("⚠️ Entrenando modelo IA...")

    X = np.random.rand(100, 3)
    y = np.random.randint(0, 2, 100)

    model = RandomForestClassifier()
    model.fit(X, y)

    joblib.dump(model, MODEL_PATH)

    return model


def cargar_modelo():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    else:
        return entrenar_modelo()


def predecir(modelo, n_partidos):
    X = np.random.rand(n_partidos, 3)
    probs = modelo.predict_proba(X)[:, 1]
    return probs
