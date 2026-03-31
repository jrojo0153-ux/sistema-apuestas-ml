import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def entrenar_modelo():
    # Buscamos en la subcarpeta data como indicaste
    ruta_csv = 'data/Historico.csv'
    if not os.path.exists(ruta_csv):
        print(f"❌ Error: {ruta_csv} no encontrado")
        return None
        
    df = pd.read_csv(ruta_csv)
    df['diff'] = df['home_odds'] - df['away_odds']
    
    X = df[['home_odds', 'away_odds', 'diff']]
    y = df['resultado']
    
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X, y)
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(modelo, 'models/modelo.pkl')
    
    print(f"Accuracy: {modelo.score(X, y)}")
    print("✅ Modelo guardado")
    return modelo

def cargar_modelo():
    if os.path.exists('models/modelo.pkl'):
        return joblib.load('models/modelo.pkl')
    return None

def predecir(modelo, partido):
    try:
        h = partido.get("home_odds")
        a = partido.get("away_odds")
        diff = h - a
        return modelo.predict_proba([[h, a, diff]])[0][1]
    except Exception:
        return None
