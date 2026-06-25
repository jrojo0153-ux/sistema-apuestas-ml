
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

def train_models():
    print("Iniciando entrenamiento avanzado...")
    
    # Usar el archivo que sí tiene datos de goles (basado en la inspección)
    # Nota: wc_all_matches.csv o Fifa_world_cup_matches.csv suelen tener scores
    possible_files = ['data/Fifa_world_cup_matches.csv', 'data/wc_all_matches.csv', 'data/matches.csv']
    df = None
    
    for f in possible_files:
        if os.path.exists(f):
            temp = pd.read_csv(f)
            if 'home_score' in temp.columns or 'goals_home' in temp.columns:
                df = temp
                print(f"Usando {f} para entrenamiento.")
                break

    if df is None:
        print("No se encontró un dataset con columnas de puntuación.")
        return

    # Estandarizar nombres de columnas si es necesario
    if 'goals_home' in df.columns: df.rename(columns={'goals_home': 'home_score', 'goals_away': 'away_score'}, inplace=True)
    
    df['total_goals'] = df['home_score'] + df['away_score']
    
    # Features: Equipos
    X = pd.get_dummies(df[['home_team', 'away_team']], drop_first=True)
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
