# ml/model.py

import os
import joblib
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from data.features import construir_features
from config import MODEL_PATH


def entrenar_modelo():
    df = pd.read_csv("data/historico.csv")

    if len(df) < 10:
        raise ValueError("❌ Necesitas al menos 10 filas en historico.csv")

    X = []
    y = []

    for _, row in df.iterrows():
        try:
            partido = {
                "home_odds": float(row["home_odds"]),
                "away_odds": float(row["away_odds"])
            }

            X.append(construir_features(partido))
            y.append(int(row["resultado"]))

        except:
            continue

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    modelo = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        min_samples_split=10,
        random_state=42
    )

    modelo.fit(X_train, y_train)

    preds = modelo.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"📊 Accuracy: {round(acc, 3)}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(modelo, MODEL_PATH)

    print("✅ Modelo guardado")

    return modelo


def cargar_modelo():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None


def predecir(modelo, partido):
    try:
        features = construir_features(partido).reshape(1, -1)
        prob = modelo.predict_proba(features)[0][1]

        if not (0 <= prob <= 1):
            return None

        return prob

    except Exception as e:
        print(f"Error predicción: {e}")
        return None
