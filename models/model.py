import os
import pickle
from sklearn.ensemble import RandomForestClassifier
import numpy as np

RUTA_MODELO = "models/modelo.pkl"

def entrenar_modelo():
    print("🧠 Entrenando modelo IA...")

    # Datos simulados (puedes mejorar luego)
    X = np.random.rand(200, 3)
    y = np.random.randint(0, 2, 200)

    modelo = RandomForestClassifier()
    modelo.fit(X, y)

    os.makedirs("models", exist_ok=True)

    with open(RUTA_MODELO, "wb") as f:
        pickle.dump(modelo, f)

    print("✅ Modelo guardado")

    return modelo


def cargar_modelo():
    if os.path.exists(RUTA_MODELO):
        print("📦 Cargando modelo existente...")
        with open(RUTA_MODELO, "rb") as f:
            return pickle.load(f)
    else:
        print("⚠️ Modelo no encontrado, entrenando...")
        return entrenar_modelo()
