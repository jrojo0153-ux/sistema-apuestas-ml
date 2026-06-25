
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
import joblib
import os

def train_models():
    print("Iniciando entrenamiento de modelos...")
    
    # Cargar datos principales
    try:
        matches = pd.read_csv('data/matches.csv')
        # Intentar cargar datos adicionales si existen para enriquecer
        if os.path.exists('data/Fifa_world_cup_matches.csv'):
            extra_matches = pd.read_csv('data/Fifa_world_cup_matches.csv')
            # Lógica simple de unión (ejemplo)
            # matches = pd.concat([matches, extra_matches], ignore_index=True)
    except Exception as e:
        print(f"Error cargando datos: {e}")
        return

    # Preprocesamiento básico (Ejemplo: Predecir goles totales)
    # Nota: Aquí se ajustaría a la lógica específica del usuario
    if 'home_score' in matches.columns and 'away_score' in matches.columns:
        matches['total_goals'] = matches['home_score'] + matches['away_score']
        
        # Features simples: Codificación de equipos
        X = pd.get_dummies(matches[['home_team', 'away_team']], drop_first=True)
        y = matches['total_goals']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 1. Random Forest
        rf = RandomForestRegressor(n_estimators=100)
        rf.fit(X_train, y_train)
        print(f"Random Forest Score: {rf.score(X_test, y_test)}")

        # 2. LightGBM
        train_data = lgb.Dataset(X_train, label=y_train)
        params = {'objective': 'regression', 'metric': 'rmse'}
        bst = lgb.train(params, train_data, num_boost_round=50)

        # Guardar el modelo principal
        os.makedirs('models', exist_ok=True)
        joblib.dump(rf, 'models/modelo_rf.pkl')
        bst.save_model('models/modelo_lgb.txt')
        
        print("Modelos entrenados y guardados en la carpeta models/")
    else:
        print("No se encontraron las columnas necesarias para el entrenamiento en matches.csv")

if __name__ == '__main__':
    train_models()
