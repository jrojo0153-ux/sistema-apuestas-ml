import os
import requests
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

class TelegramNotifier:
    def __init__(self):
        self.token = os.getenv("TELEGRAM_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")

    def send_alert(self, message: str) -> bool:
        if not self.token or not self.chat_id:
            print("⚠️ Telegram no configurado")
            return False
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {"chat_id": self.chat_id, "text": message, "parse_mode": "HTML"}
        try:
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            print(f"❌ Error al enviar alerta de Telegram: {e}")
            return False

class FeatureEngineer:
    @staticmethod
    def create_features(df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        df = df.copy()
        df = df.sort_values(by="timestamp").reset_index(drop=True)
        
        # Métricas de rendimiento histórico rodante (Rolling Windows)
        for window in [3, 5, 10]:
            df[f"rolling_mean_home_score_{window}"] = df.groupby("home_team")["home_score"].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
            df[f"rolling_mean_away_score_{window}"] = df.groupby("away_team")["away_score"].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
            df[f"rolling_std_home_score_{window}"] = df.groupby("home_team")["home_score"].transform(lambda x: x.shift(1).rolling(window, min_periods=1).std())
            df[f"rolling_std_away_score_{window}"] = df.groupby("away_team")["away_score"].transform(lambda x: x.shift(1).rolling(window, min_periods=1).std())
            
        # Factores de Momentum y Media Móvil Exponencial (EMA)
        df["home_ema_5"] = df.groupby("home_team")["home_score"].transform(lambda x: x.shift(1).ewm(span=5, adjust=False).mean())
        df["away_ema_5"] = df.groupby("away_team")["away_score"].transform(lambda x: x.shift(1).ewm(span=5, adjust=False).mean())
        
        # Ingeniería de cuotas e implícitas
        df["implied_prob_home"] = 1 / df["odds_home"]
        df["implied_prob_away"] = 1 / df["odds_away"]
        df["bookmaker_margin"] = (1 / df["odds_home"]) + (1 / df["odds_away"]) - 1
        df["market_discrepancy"] = df["implied_prob_home"] - df["implied_prob_away"]
        
        # Creación de Variable Objetivo para Clasificación Binaria (Home Win = 1, Else = 0)
        if is_training and "home_score" in df.columns and "away_score" in df.columns:
            df["target"] = (df["home_score"] > df["away_score"]).astype(int)
            
        return df.fillna(0)

class MLPredictor:
    def __init__(self):
        # Hiperparámetros optimizados para mitigar sobreajuste
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.02,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss"
        )
        self.lgb_model = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.02,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=-1
        )
        self.scaler = StandardScaler()
        self.calibrated_xgb = None
        self.calibrated_lgb = None
        self._is_trained = False

    def train(self, X: pd.DataFrame, y: pd.Series):
        X_scaled = self.scaler.fit_transform(X)
        
        # Calibración de probabilidad (Crucial para una correcta aplicación de Criterio de Kelly)
        self.calibrated_xgb = CalibratedClassifierCV(estimator=self.xgb_model, method="sigmoid", cv=5)
        self.calibrated_lgb = CalibratedClassifierCV(estimator=self.lgb_model, method="sigmoid", cv=5)
        
        self.calibrated_xgb.fit(X_scaled, y)
        self.calibrated_lgb.fit(X_scaled, y)
        self._is_trained = True

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self._is_trained:
            raise ValueError("El modelo debe ser entrenado previamente.")
        X_scaled = self.scaler.transform(X)
        xgb_probs = self.calibrated_xgb.predict_proba(X_scaled)[:, 1]
        lgb_probs = self.calibrated_lgb.predict_proba(X_scaled)[:, 1]
        
        # Modelo Ensamble (Promedio ponderado para maximizar robustez)
        return 0.5 * xgb_probs + 0.5 * lgb_probs

class KellyCalculator:
    @staticmethod
    def calculate_fractional_kelly(
        prob: float, 
        odds: float, 
        fraction: float = 0.15, 
        min_probability_diff: float = 0.03
    ) -> float:
        """
        Calcula el tamaño óptimo de la apuesta usando Criterio de Kelly Fraccional Avanzado.
        - fraction: Fracción de Kelly (Seguridad frente a varianza extrema, p.ej. 0.15 para Kelly de un sexto).
        - min_probability_diff: Margen mínimo de ventaja requerida frente al mercado (Edge).
        """
        implied_prob = 1 / odds
        edge = prob - implied_prob
        
        if edge <= min_probability_diff:
            return 0.0
            
        b = odds - 1
        q = 1.0 - prob
        kelly_f = (prob * b - q) / b
        
        # Aplicación de Kelly Fraccional
        fractional_f = kelly_f * fraction
        
        # Límite máximo de riesgo por evento individual (Risk Management de nivel institucional)
        return float(np.clip(fractional_f, 0.0, 0.05))

class ProductionPipeline:
    def __init__(self, bankroll: float = 10000.0):
        self.bankroll = bankroll
        self.notifier = TelegramNotifier()
        self.predictor = MLPredictor()
        self.feature_cols = []
        
    def run_training_pipeline(self, historical_data: pd.DataFrame):
        df = FeatureEngineer.create_features(historical_data, is_training=True)
        self.feature_cols = [
            c for c in df.columns if c not in ["target", "timestamp", "home_team", "away_team", "home_score", "away_score"]
        ]
        
        X = df[self.feature_cols]
        y = df["target"]
        
        self.predictor.train(X, y)
        print("🚀 [SUCCESS] Sistema entrenado y calibrado con éxito.")

    def process_live_events(self, live_data: pd.DataFrame):
        processed_data = FeatureEngineer.create_features(live_data, is_training=False)
        X_live = processed_data[self.feature_cols]
        
        probabilities = self.predictor.predict_proba(X_live)
        
        for idx, row in processed_data.iterrows():
            prob = probabilities[idx]
            odds = row["odds_home"]
            stake_pct = KellyCalculator.calculate_fractional_kelly(prob, odds)
            
            if stake_pct > 0:
                monetary_stake = self.bankroll * stake_pct
                edge = prob - (1 / odds)
                
                alert_msg = (
                    f"🤖 <b>SEÑAL DE ALTA PROBABILIDAD (ML)</b> 🤖\n\n"
                    f"⚽ <b>Evento:</b> {row['home_team']} vs {row['away_team']}\n"
                    f"📊 <b>Probabilidad ML Calibrada:</b> {prob:.2%}\n"
                    f"📈 <b>Cuota del Mercado:</b> {odds:.2f} (Implícita: {1/odds:.2%})\n"
                    f"🔥 <b>Ventaja Detectada (Edge):</b> {edge:.2%}\n"
                    f"💎 <b>Criterio de Kelly (Fraccional):</b> {stake_pct:.2%}\n"
                    f"💰 <b>Asignación de Capital:</b> {monetary_stake:.2f} USD (De {self.bankroll:.2f} USD)"
                )
                self.notifier.send_alert(alert_msg)
                print(f"✅ Señal enviada a Telegram con {stake_pct:.2%} de asignación para {row['home_team']}.")
            else:
                print(f"ℹ️ Evento analizado {row['home_team']} vs {row['away_team']} - Sin valor suficiente detectado.")

if __name__ == "__main__":
    # Simulación e inicialización sintética
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=1000, freq="D")
    teams = ["Real Madrid", "Barcelona", "Man City", "Bayern", "PSG", "Liverpool"]
    
    historical_mock_db = pd.DataFrame({
        "timestamp": dates,
        "home_team": np.random.choice(teams, size=1000),
        "away_team": np.random.choice(teams, size=1000),
        "home_score": np.random.randint(0, 5, size=1000),
        "away_score": np.random.randint(0, 5, size=1000),
        "odds_home": np.random.uniform(1.4, 3.8, size=1000),
        "odds_away": np.random.uniform(1.4, 3.8, size=1000)
    })
    
    # Arrancar Pipeline de Producción
    pipeline = ProductionPipeline(bankroll=25000.0)
    pipeline.run_training_pipeline(historical_mock_db)
    
    # Evento Entrante en Tiempo Real
    live_event = pd.DataFrame({
        "timestamp": [pd.Timestamp.now()],
        "home_team": ["Real Madrid"],
        "away_team": ["Barcelona"],
        "home_score": [0],
        "away_score": [0],
        "odds_home": [2.40],
        "odds_away": [1.60]
    })
    
    pipeline.process_live_events(live_event)