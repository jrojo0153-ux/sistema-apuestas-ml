import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def entrenar_modelo():
    # Cargar tu Historico.csv sin moverlo de sitio
    df = pd.read_csv('Historico.csv')
    
    # Usamos las cuotas como base, pero el modelo necesita ver la diferencia (Edge)
    df['odds_diff'] = df['home_odds'] - df['away_odds']
    
    X = df[['home_odds', 'away_odds', 'odds_diff']]
    y = df['resultado'] # 1 si gana local, 0 si no
    
    # Usamos RandomForest con 100 estimadores para mayor estabilidad
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X, y)
    
    # Guardar en la ruta que ya definiste en tu YAML
    os.makedirs('models', exist_ok=True)
    joblib.dump(modelo, 'models/modelo.pkl')
    
    accuracy = modelo.score(X, y)
    print(f"📊 Nuevo Accuracy de entrenamiento: {accuracy}")
    return modelo

def predecir(modelo, partido):
    # Preparar los datos del partido actual igual que el entrenamiento
    home_odds = partido.get("home_odds")
    away_odds = partido.get("away_odds")
    odds_diff = home_odds - away_odds
    
    # El modelo devuelve la probabilidad del resultado "1" (Gana local)
    prob = modelo.predict_proba([[home_odds, away_odds, odds_diff]])[0][1]
    return prob
