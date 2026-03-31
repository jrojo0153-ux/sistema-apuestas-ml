import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def guardar_edge_alto(partido, prob, edge):
    ruta_csv = 'data/Historico.csv'
    nuevo_dato = pd.DataFrame([{
        "home_team": partido['home_team'],
        "away_team": partido['away_team'],
        "home_odds": partido['home_odds'],
        "away_odds": partido['away_odds'],
        "prob_ia": prob,
        "edge": edge,
        "resultado": None # Se llenará cuando termine el partido
    }])
    
    # Guardar sin borrar lo anterior (append)
    nuevo_dato.to_csv(ruta_csv, mode='a', header=not os.path.exists(ruta_csv), index=False)

def entrenar_modelo():
    ruta_csv = 'data/Historico.csv'
    if not os.path.exists(ruta_csv): return None
        
    df = pd.read_csv(ruta_csv)
    df['diff'] = df['home_odds'] - df['away_odds']
    
    X = df[['home_odds', 'away_odds', 'diff']]
    y = df['resultado'].dropna() # Solo entrenar con partidos terminados
    
    if len(y) < 10: return None # Necesitamos datos mínimos
    
    X = X.loc[y.index]
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X, y)
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(modelo, 'models/modelo.pkl')
    return modelo

def cargar_modelo():
    if os.path.exists('models/modelo.pkl'):
        return joblib.load('models/modelo.pkl')
    return None

def predecir(modelo, partido):
    try:
        h, a = partido['home_odds'], partido['away_odds']
        return modelo.predict_proba([[h, a, h-a]])[0][1]
    except:
        return None
