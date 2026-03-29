import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier


MODEL_PATH = "models/modelo.pkl"


def entrenar_modelo():
    """
    Entrena un modelo simple con datos dummy (puedes mejorar luego)
    """
    # Datos fake (luego puedes conectar histórico real)
    X = np.random.rand(100, 3)
    y = np.random.randint(0, 2, 100)

    modelo = RandomForestClassifier()
    modelo.fit(X, y)

    os.makedirs("models", exist_ok=True)
    joblib.dump(modelo, MODEL_PATH)

    print("✅ Modelo entrenado y guardado")

    return modelo


def cargar_modelo():
    """
    Carga el modelo si existe, si no lo entrena
    """
    if os.path.exists(MODEL_PATH):
        try:
            modelo = joblib.load(MODEL_PATH)
            print("📦 Modelo cargado")
            return modelo
        except:
            print("⚠️ Error cargando modelo, reentrenando...")
            return entrenar_modelo()
    else:
        print("⚠️ No existe modelo, entrenando...")
        return entrenar_modelo()


def predecir(modelo, partidos):
    """
    Genera probabilidades para cada partido
    """
    probabilidades = []

    for _ in partidos:
        # Features dummy (puedes mejorar luego)
        features = np.random.rand(1, 3)

        prob = modelo.predict_proba(features)[0][1]
        probabilidades.append(prob)

    return probabilidades
