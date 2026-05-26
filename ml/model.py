import os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, roc_auc_score
from xgboost import XGBClassifier

class FeatureEngineer:
    @staticmethod
    def transform(df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        df = df.copy()
        
        required_cols = ['home_odds', 'away_odds']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Falta la columna requerida: {col}")
                
        df['home_odds'] = pd.to_numeric(df['home_odds'], errors='coerce')
        df['away_odds'] = pd.to_numeric(df['away_odds'], errors='coerce')
        
        df['implied_prob_home'] = 1.0 / df['home_odds']
        df['implied_prob_away'] = 1.0 / df['away_odds']
        
        df['bookmaker_margin'] = (df['implied_prob_home'] + df['implied_prob_away']) - 1.0
        
        df['clean_prob_home'] = df['implied_prob_home'] / (df['implied_prob_home'] + df['implied_prob_away'])
        df['clean_prob_away'] = df['implied_prob_away'] / (df['implied_prob_home'] + df['implied_prob_away'])
        
        df['odds_ratio'] = df['home_odds'] / df['away_odds']
        df['log_odds_ratio'] = np.log(df['odds_ratio'] + 1e-9)
        df['odds_diff'] = df['home_odds'] - df['away_odds']
        df['odds_sum'] = df['home_odds'] + df['away_odds']
        df['entropy'] = - (df['clean_prob_home'] * np.log(df['clean_prob_home'] + 1e-9) + 
                           df['clean_prob_away'] * np.log(df['clean_prob_away'] + 1e-9))
        
        features = [
            'home_odds', 'away_odds', 'implied_prob_home', 'implied_prob_away',
            'bookmaker_margin', 'clean_prob_home', 'clean_prob_away',
            'odds_ratio', 'log_odds_ratio', 'odds_diff', 'odds_sum', 'entropy'
        ]
        
        if is_training:
            if 'resultado' not in df.columns:
                raise ValueError("La columna 'resultado' es obligatoria para el entrenamiento.")
            df['target'] = df['resultado'].astype(int)
            return df[features], df['target']
            
        return df[features]

class KellyPortfolioManager:
    def __init__(self, fraction: float = 0.25, max_bet_fraction: float = 0.05, min_edge: float = 0.02):
        self.fraction = fraction
        self.max_bet_fraction = max_bet_fraction
        self.min_edge = min_edge

    def calculate_bet_size(self, decimal_odds: float, predicted_prob: float) -> float:
        if decimal_odds <= 1.0 or predicted_prob <= 0.0:
            return 0.0
        
        implied_prob = 1.0 / decimal_odds
        edge = predicted_prob - implied_prob
        
        if edge < self.min_edge:
            return 0.0
            
        b = decimal_odds - 1.0
        q = 1.0 - predicted_prob
        
        kelly_f = (predicted_prob * b - q) / b
        
        suggested_bet = kelly_f * self.fraction
        
        return float(np.clip(suggested_bet, 0.0, self.max_bet_fraction))

class PredictiveModel:
    def __init__(self, model_dir: str = 'models'):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.model_dir / 'calibrated_xgb.pkl'
        self.model = None

    def train(self, X: pd.DataFrame, y: pd.Series) -> XGBClassifier:
        base_estimator = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            scale_pos_weight=1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            n_jobs=-1
        )
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        self.model = CalibratedClassifierCV(
            estimator=base_estimator,
            method='sigmoid',
            cv=cv
        )
        
        self.model.fit(X, y)
        
        try:
            train_preds = self.model.predict_proba(X)[:, 1]
            loss = log_loss(y, train_preds)
            auc = roc_auc_score(y, train_preds)
            print(f"📊 Métricas de Entrenamiento | LogLoss: {loss:.4f} | AUC-ROC: {auc:.4f}")
        except Exception as e:
            print(f"⚠️ Error al calcular métricas: {e}")
            
        joblib.dump(self.model, self.model_path)
        print(f"✅ Modelo calibrado y guardado con éxito en: {self.model_path}")
        return self.model

    def load(self) -> bool:
        if self.model_path.exists():
            try:
                self.model = joblib.load(self.model_path)
                return True
            except Exception as e:
                print(f"❌ Error al cargar el modelo: {e}")
                return False
        return False

    def predict_probability(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            if not self.load():
                raise RuntimeError("Modelo no cargado. Entrene el modelo primero.")
        return self.model.predict_proba(X)[:, 1]

def entrenar_modelo():
    ruta_csv = Path('data/Historico.csv')
    if not ruta_csv.exists():
        print(f"❌ Error: {ruta_csv} no encontrado")
        return None
        
    try:
        df = pd.read_csv(ruta_csv).dropna(subset=['home_odds', 'away_odds', 'resultado'])
        
        X, y = FeatureEngineer.transform(df, is_training=True)
        
        trainer = PredictiveModel()
        modelo = trainer.train(X, y)
        return modelo
    except Exception as e:
        print(f"❌ Error crítico durante el entrenamiento: {e}")
        return None

def cargar_modelo():
    trainer = PredictiveModel()
    if trainer.load():
        return trainer
    return None

def predecir(modelo_wrapper, partido: dict, bankroll: float = 1000.0) -> dict:
    try:
        h = partido.get('home_odds')
        a = partido.get('away_odds')
        if h is None or a is None:
            return {"error": "Faltan cuotas de partido"}
            
        df_partido = pd.DataFrame([partido])
        X_pred = FeatureEngineer.transform(df_partido, is_training=False)
        
        prob_home = modelo_wrapper.predict_probability(X_pred)[0]
        prob_away = 1.0 - prob_home
        
        pm = KellyPortfolioManager(fraction=0.25, max_bet_fraction=0.05, min_edge=0.02)
        
        bet_size_home_pct = pm.calculate_bet_size(float(h), float(prob_home))
        bet_size_away_pct = pm.calculate_bet_size(float(a), float(prob_away))
        
        decision = "NO BET"
        bet_amount = 0.0
        selected_odds = 1.0
        predicted_prob = 0.0
        
        if bet_size_home_pct > 0.0 and bet_size_home_pct >= bet_size_away_pct:
            decision = "HOME WIN"
            bet_amount = bet_size_home_pct * bankroll
            selected_odds = h
            predicted_prob = prob_home
        elif bet_size_away_pct > 0.0 and bet_size_away_pct > bet_size_home_pct:
            decision = "AWAY WIN"
            bet_amount = bet_size_away_pct * bankroll
            selected_odds = a
            predicted_prob = prob_away
            
        return {
            "decision": decision,
            "bet_amount": round(bet_amount, 2),
            "suggested_fraction": round(max(bet_size_home_pct, bet_size_away_pct), 4),
            "home_probability": round(float(prob_home), 4),
            "away_probability": round(float(prob_away), 4),
            "expected_value": round((predicted_prob * selected_odds) - 1.0, 4) if decision != "NO BET" else 0.0
        }
    except Exception as e:
        print(f"❌ Error en la predicción/gestión de banca: {e}")
        return {"error": str(e)}