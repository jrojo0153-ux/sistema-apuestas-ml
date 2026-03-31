import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def entrenar_modelo():
    ruta_csv = 'data/Historico.csv'
    if not os.path.exists(ruta_csv):
        print(f"❌ Error: {ruta_csv} no encontrado")
        return None
        
    try:
        df = pd.read_csv(ruta_csv)
        df['diff'] = df['home_odds'] - df['away_odds']
        
        X = df[['home_odds', 'away_odds', 'diff']]
        y = df['resultado']
        
        modelo = RandomForestClassifier(n_estimators=100, random_state=42)
        modelo.fit(X, y)
        
        # Crear carpeta models si no existe
        os.makedirs('models', exist_ok=True)
        
        joblib.dump(modelo, 'models/modelo.pkl')
        print(f"✅ Modelo guardado. Accuracy: {modelo.score(X, y)}")
        return modelo
    except Exception as e:
        print(f"❌ Error durante el entrenamiento: {e}")
        return None

def cargar_modelo():
    ruta = 'models/modelo.pkl'
    if os.path.exists(ruta):
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
        if h is None or a is None: return None
        
        return modelo.predict_proba([[h, a, h-a]])[0][1]
    except Exception as e:
        print(f"⚠️ Error en predicción: {e}")
        return None
