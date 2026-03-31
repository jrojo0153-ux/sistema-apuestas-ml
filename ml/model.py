import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def entrenar_modelo():
    ruta_csv = 'data/Historico.csv'
    if not os.path.exists(ruta_csv):
        print("❌ Historico.csv no encontrado")
        return None
        
    df = pd.read_csv(ruta_csv)
    # Aseguramos que existan las columnas para el Edge
    if 'home_odds' not in df.columns:
        return None

    df['diff'] = df['home_odds'] - df['away_odds']
    X = df[['home_odds', 'away_odds', 'diff']]
    y = df['resultado']
    
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X, y)
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(modelo, 'models/modelo.pkl')
    print(f"📊 Accuracy: {modelo.score(X, y)}")
    return modelo

def cargar_modelo():
    ruta = 'models/modelo.pkl'
    if os.path.exists(ruta):
        try:
            return joblib.load(ruta)
        except Exception as e:
            print(f"⚠️ Modelo incompatible o corrupto, eliminando... ({e})")
            os.remove(ruta) # Borramos el archivo malo para que el bot cree uno nuevo
    return None

def predecir(modelo, partido):
    try:
        h, a = partido['home_odds'], partido['away_odds']
        return modelo.predict_proba([[h, a, h-a]])[0][1]
    except:
        return None
