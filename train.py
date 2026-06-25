
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

def train_models():
    print("Iniciando entrenamiento avanzado...")
    
    # Mapeo de archivos y sus respectivas columnas de goles
    config = [
        {'file': 'data/wc_all_matches.csv', 'h': 'score1', 'a': 'score2'},
        {'file': 'data/Fifa_world_cup_matches.csv', 'h': 'number of goals team1', 'a': 'number of goals team2'}
    ]
    
    df = None
    for item in config:
        if os.path.exists(item['file']):
            temp = pd.read_csv(item['file'])
            if item['h'] in temp.columns:
                temp = temp.rename(columns={item['h']: 'home_score', item['a']: 'away_score'})
                df = temp
                print(f"Usando {item['file']} para el entrenamiento.")
                break

    if df is None:
        print("No se encontró un dataset válido para entrenar.")
        return

    # Limpieza: eliminar filas sin score y calcular total
    df = df.dropna(subset=['home_score', 'away_score'])
    df['total_goals'] = df['home_score'].astype(float) + df['away_score'].astype(float)
    
    # Features: Equipos
    X = pd.get_dummies(df[['team1', 'team2']], drop_first=True)
    y = df['total_goals']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    print(f"Modelo entrenado. Score R2: {model.score(X_test, y_test)}")

    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/modelo_rf.pkl')
    print("Modelo guardado en models/modelo_rf.pkl")

if __name__ == '__main__':
    train_models()
