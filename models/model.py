import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

MODEL_PATH = "models/modelo.pkl"

def entrenar_modelo():
    X = np.random.rand(100, 3)
    y = np.random.randint(0, 2, 100)

    model = RandomForestClassifier()
    model.fit(X, y)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print("✅ Modelo entrenado y guardado")
    return model


def cargar_modelo():
    if os.path.exists(MODEL_PATH):
        print("📦 Modelo cargado")
        return joblib.load(MODEL_PATH)
    else:
        return entrenar_modelo()


def predecir(model, num_partidos):
    X = np.random.rand(num_partidos, 3)
    probs = model.predict_proba(X)[:, 1]
    return probs
