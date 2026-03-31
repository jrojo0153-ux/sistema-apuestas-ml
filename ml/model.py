import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def entrenar_modelo():
    if not os.path.exists('Historico.csv'):
        print("❌ Archivo Historico.csv no encontrado")
        return None
        
    df = pd.read_csv('Historico.csv')
    
    # Ingeniería de datos: Diferencia de cuotas
    df['diff'] = df['home_odds'] - df['away_odds']
    
    # Variables de entrada y objetivo
    X = df[['home_odds', 'away_odds', 'diff']]
    y = df['resultado']
    
    # Modelo robusto para evitar el accuracy de 0.5
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X, y)
    
    # Guardar en la ruta definida
    os.makedirs('models', exist_ok=True)
    joblib.dump(modelo, 'models/modelo.pkl')
    
    accuracy = modelo.score(X, y)
    print(f"📊 Modelo entrenado. Accuracy: {round(accuracy, 2)}")
    return modelo

def cargar_modelo():
    ruta = 'models/modelo.pkl'
    if os.path.exists(ruta):
        return joblib.load(ruta)
    return None

def predecir(modelo, partido):
    try:
        h = partido.get("home_odds")
        a = partido.get("away_odds")
        diff = h - a
        # Predice la probabilidad de victoria local (clase 1)
        prob = modelo.predict_proba([[h, a, diff]])[0][1]
        return prob
    except Exception as e:
        print(f"⚠️ Error en predicción: {e}")
        return None
