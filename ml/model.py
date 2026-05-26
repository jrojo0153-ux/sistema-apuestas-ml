import os
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

def entrenar_modelo():
    ruta_csv = Path('data/Historico.csv')
    if not ruta_csv.exists():
        print(f"❌ Error: {ruta_csv} no encontrado")
        return None
        
    try:
        df = pd.read_csv(ruta_csv).dropna(subset=['home_odds', 'away_odds', 'resultado'])
        df['diff'] = df['home_odds'] - df['away_odds']
        
        X = df[['home_odds', 'away_odds', 'diff']]
        y = df['resultado']
        
        modelo = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        modelo.fit(X, y)
        
        ruta_modelos = Path('models')
        ruta_modelos.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(modelo, ruta_modelos / 'modelo.pkl')
        print(f"✅ Modelo guardado. Accuracy de entrenamiento: {modelo.score(X, y):.4f}")
        return modelo
    except Exception as e:
        print(f"❌ Error durante el entrenamiento: {e}")
        return None

def cargar_modelo():
    ruta = Path('models/modelo.pkl')
    if ruta.exists():
        try:
            return joblib.load(ruta)
        except Exception as e:
            print(f"⚠️ Error al cargar modelo: {e}")
            return None
    return None

def predecir(modelo, partido):
    try:
        h = partido.get('home_odds')
        a = partido.get('away_odds')
        if h is None or a is None: 
            return None
        
        X_pred = pd.DataFrame([[h, a, h - a]], columns=['home_odds', 'away_odds', 'diff'])
        probabilidades = modelo.predict_proba(X_pred)[0]
        return probabilidades[1] if len(probabilidades) > 1 else probabilidades[0]
    except Exception as e:
        print(f"⚠️ Error en predicción: {e}")
        return None