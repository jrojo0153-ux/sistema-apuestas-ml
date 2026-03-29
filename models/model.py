import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier

MODEL_PATH = "models/model.pkl"

def entrenar_modelo():
    X = np.random.rand(20, 3)
    y = np.random.randint(0, 2, 20)

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)

    joblib.dump(model, MODEL_PATH)
    return model

def cargar_modelo():
    if not os.path.exists(MODEL_PATH):
        print("⚠️ Modelo no encontrado, entrenando...")
        return entrenar_modelo()

    return joblib.load(MODEL_PATH)

def predecir(modelo, X):
    return modelo.predict_proba(X)[:, 1]
