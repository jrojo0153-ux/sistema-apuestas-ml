import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def entrenar_modelo():
    # 1. Verificar datos
    ruta_csv = 'data/Historico.csv'
    if not os.path.exists(ruta_csv):
        print("❌ Error: data/Historico.csv no encontrado")
        return None
        
    df = pd.read_csv(ruta_csv)
    df['diff'] = df['home_odds'] - df['away_odds']
    
    X = df[['home_odds', 'away_odds', 'diff']]
    y = df['resultado']
    
    # 2. Entrenar
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X, y)
    
    # 3. CREAR CARPETA SI NO EXISTE
    if not os.path.exists('models'):
        print("📁 Carpeta 'models' no detectada. Creándola...")
        os.makedirs('models')
    
    # 4. Guardar
    joblib.dump(modelo, 'models/modelo.pkl')
    print(f"✅ Modelo guardado exitosamente. Accuracy: {modelo.score(X, y)}")
    return modelo

def cargar_modelo():
    ruta = 'models/modelo.pkl'
    if os.path.exists(ruta):
        try:
            return joblib.load(ruta)
        except:
            print("⚠️ Modelo corrupto detectado. Eliminando para re-entrenar...")
            os.remove(ruta)
    return None

def predecir(modelo, partido):
    try:
        h, a = partido['home_odds'], partido['away_odds']
        return modelo.predict_proba([[h, a, h-a]])[0][1]
    except:
        return None
