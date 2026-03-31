import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def entrenar_modelo():
    if not os.path.exists('Historico.csv'):
        print("❌ Error: Historico.csv no encontrado")
        return None
        
    df = pd.read_csv('Historico.csv')
    # Diferencia de cuotas como factor de decisión
    df['diff'] = df['home_odds'] - df['away_odds']
    
    X = df[['home_odds', 'away_odds', 'diff']]
    y = df['resultado']
    
    # Random Forest para superar el 0.5 de accuracy
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X, y)
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(modelo, 'models/modelo.pkl')
    
    print(f"Accuracy: {modelo.score(X, y)}")
    print("✅ Modelo guardado")
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
        # Probabilidad de victoria local
        prob = modelo.predict_proba([[h, a, diff]])[0][1]
        return prob
    except:
        return None
