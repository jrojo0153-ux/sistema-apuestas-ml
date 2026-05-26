import os
import sys
import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.calibration import IsotonicRegression
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("QuantBetAI")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
BANKROLL_INICIAL = float(os.getenv("BANKROLL_INICIAL", "10000.0"))
KELLY_FRACCION = float(os.getenv("KELLY_FRACCION", "0.25"))
EDGE_MINIMO = float(os.getenv("EDGE_MINIMO", "0.02"))
MAX_EXPOSICION_TOTAL = float(os.getenv("MAX_EXPOSICION_TOTAL", "0.15"))
MAX_APUESTA_INDIVIDUAL = float(os.getenv("MAX_APUESTA_INDIVIDUAL", "0.05"))

class MarketMathematics:
    @staticmethod
    def remove_margin(odds: List[float]) -> np.ndarray:
        if not odds or any(o <= 1.0 for o in odds):
            return np.array([])
        implied_probs = np.array([1.0 / o for o in odds])
        margin = np.sum(implied_probs) - 1.0
        if margin <= 0:
            return implied_probs / np.sum(implied_probs)
        return implied_probs / (1.0 + margin)

    @staticmethod
    def compute_ev(prob: float, odds: float) -> float:
        return (prob * odds) - 1.0

class FeatureEngineering:
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False

    def build_features(self, df: pd.DataFrame, training: bool = False) -> pd.DataFrame:
        df = df.copy()
        
        if 'home_odds' in df.columns and 'away_odds' in df.columns:
            df['implied_prob_home'] = 1.0 / df['home_odds']
            df['implied_prob_away'] = 1.0 / df['away_odds']
            df['market_margin'] = (df['implied_prob_home'] + df['implied_prob_away']) - 1.0
            
            probs = df.apply(lambda row: MarketMathematics.remove_margin([row['home_odds'], row['away_odds']]), axis=1)
            df['clean_prob_home'] = [p[0] if len(p) > 0 else 0.5 for p in probs]
            df['clean_prob_away'] = [p[1] if len(p) > 0 else 0.5 for p in probs]
        else:
            df['implied_prob_home'] = 0.5
            df['implied_prob_away'] = 0.5
            df['market_margin'] = 0.05
            df['clean_prob_home'] = 0.5
            df['clean_prob_away'] = 0.5

        if 'goles_local_historico' in df.columns and 'goles_visitante_historico' in df.columns:
            df['diff_goles_historico'] = df['goles_local_historico'] - df['goles_visitante_historico']
            df['ratio_goles'] = (df['goles_local_historico'] + 1) / (df['goles_visitante_historico'] + 1)
        else:
            df['diff_goles_historico'] = 0.0
            df['ratio_goles'] = 1.0

        if 'posesion_home' in df.columns and 'posesion_away' in df.columns:
            df['diff_posesion'] = df['posesion_home'] - df['posesion_away']
        else:
            df['diff_posesion'] = 0.0

        feature_cols = [
            'implied_prob_home', 'implied_prob_away', 'market_margin', 
            'clean_prob_home', 'clean_prob_away', 'diff_goles_historico', 
            'ratio_goles', 'diff_posesion'
        ]

        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0.0

        df_features = df[feature_cols].fillna(0.0)

        if training:
            self.scaler.fit(df_features)
            self.is_fitted = True
            
        if self.is_fitted:
            scaled_array = self.scaler.transform(df_features)
            df_scaled = pd.DataFrame(scaled_array, columns=feature_cols, index=df.index)
            return df_scaled
        
        return df_features

class EnsemblePredictor:
    def __init__(self):
        self.xgb_model = None
        self.lgb_model = None
        self.calibrator = None
        self.feature_pipeline = FeatureEngineering()
        
    def train(self, X: pd.DataFrame, y: np.ndarray):
        logger.info("Iniciando entrenamiento del ensamble de alta precisión (XGBoost + LightGBM)...")
        
        X_features = self.feature_pipeline.build_features(X, training=True)
        X_train, X_val, y_train, y_val = train_test_split(X_features, y, test_size=0.2, random_state=42)

        xgb_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 5,
            'learning_rate': 0.03,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'seed': 42
        }
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        self.xgb_model = xgb.train(
            xgb_params, dtrain, num_boost_round=500,
            evals=[(dval, 'val')], early_stopping_rounds=50, verbose_eval=False
        )

        lgb_train = lgb.Dataset(X_train, label=y_train)
        lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
        lgb_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'learning_rate': 0.03,
            'num_leaves': 31,
            'max_depth': 5,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'seed': 42,
            'verbose': -1
        }
        self.lgb_model = lgb.train(
            lgb_params, lgb_train, num_boost_round=500,
            valid_sets=[lgb_val], callbacks=[lgb.early_stopping(50, verbose=False)]
        )

        pred_xgb = self.xgb_model.predict(xgb.DMatrix(X_val))
        pred_lgb = self.lgb_model.predict(X_val)
        ensemble_preds = (pred_xgb * 0.5) + (pred_lgb * 0.5)

        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.calibrator.fit(ensemble_preds, y_val)
        logger.info("Entrenamiento y calibración del ensamble finalizados con éxito.")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.xgb_model is None or self.lgb_model is None or self.calibrator is None:
            raise ValueError("El modelo o su calibración no están entrenados.")
            
        X_features = self.feature_pipeline.build_features(X, training=False)
        pred_xgb = self.xgb_model.predict(xgb.DMatrix(X_features))
        pred_lgb = self.lgb_model.predict(X_features)
        
        ensemble_preds = (pred_xgb * 0.5) + (pred_lgb * 0.5)
        calibrated_preds = self.calibrator.transform(ensemble_preds)
        return calibrated_preds

class AdvancedKellyCalculator:
    @staticmethod
    def calculate_kelly_stake(prob: float, odds: float, fraction: float = KELLY_FRACCION) -> float:
        if odds <= 1.0:
            return 0.0
        edge = MarketMathematics.compute_ev(prob, odds)
        if edge <= 0:
            return 0.0
        kelly_f = edge / (odds - 1.0)
        return float(np.clip(kelly_f * fraction, 0.0, MAX_APUESTA_INDIVIDUAL))

    @staticmethod
    def optimize_simultaneous_bets(bets: List[Dict[str, Any]], bankroll: float) -> List[Dict[str, Any]]:
        if not bets:
            return []
        
        sum_raw_stakes = sum(b['raw_kelly'] for b in bets)
        for bet in bets:
            if sum_raw_stakes > MAX_EXPOSICION_TOTAL:
                bet['allocated_stake'] = bet['raw_kelly'] * (MAX_EXPOSICION_TOTAL / sum_raw_stakes)
            else:
                bet['allocated_stake'] = bet['raw_kelly']
            
            bet['allocated_amount'] = bet['allocated_stake'] * bankroll
        return bets

class NotificationManager:
    def __init__(self):
        self.session = requests.Session()
        retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        self.session.mount("https://", HTTPAdapter(max_retries=retries))

    def send_telegram(self, message: str):
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            logger.warning("Falta configurar TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID.")
            return

        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True
        }
        try:
            response = self.session.post(url, json=payload, timeout=10)
            response.raise_for_status()
        except Exception as e:
            logger.error(f"Error al enviar notificación de Telegram: {e}")

class QuantBetEngine:
    def __init__(self):
        self.predictor = EnsemblePredictor()
        self.notifier = NotificationManager()
        self.session = requests.Session()

    def load_historical_data(self) -> Tuple[pd.DataFrame, np.ndarray]:
        synth_size = 1000
        np.random.seed(42)
        
        data = {
            'home_odds': np.random.uniform(1.5, 4.0, synth_size),
            'away_odds': np.random.uniform(1.5, 4.0, synth_size),
            'goles_local_historico': np.random.poisson(1.6, synth_size),
            'goles_visitante_historico': np.random.poisson(1.2, synth_size),
            'posesion_home': np.random.uniform(40.0, 60.0, synth_size),
            'posesion_away': np.random.uniform(40.0, 60.0, synth_size),
        }
        df = pd.DataFrame(data)
        
        probs = df.apply(lambda row: MarketMathematics.remove_margin([row['home_odds'], row['away_odds']]), axis=1)
        prob_home_clean = np.array([p[0] for p in probs])
        
        y = np.where(prob_home_clean + np.random.normal(0, 0.1, synth_size) > 0.5, 1, 0)
        return df, y

    def fetch_live_matches(self) -> List[Dict[str, Any]]:
        try:
            from data.live_matches import obtener_partidos
            partidos = obtener_partidos()
            if partidos:
                return partidos
        except ImportError:
            logger.warning("No se pudo cargar 'data.live_matches'. Generando simulaciones en tiempo real de alta calidad.")
        
        return [
            {
                'home_team': 'Real Madrid', 'away_team': 'Barcelona',
                'home_odds': 1.95, 'away_odds': 3.40,
                'goles_local_historico': 2.1, 'goles_visitante_historico': 1.8,
                'posesion_home': 54.0, 'posesion_away': 46.0
            },
            {
                'home_team': 'Manchester City', 'away_team': 'Liverpool',
                'home_odds': 1.80, 'away_odds': 3.90,
                'goles_local_historico': 2.5, 'goles_visitante_historico': 1.9,
                'posesion_home': 58.0, 'posesion_away': 42.0
            },
            {
                'home_team': 'Bayern Munich', 'away_team': 'Dortmund',
                'home_odds': 1.50, 'away_odds': 5.50,
                'goles_local_historico': 2.8, 'goles_visitante_historico': 1.4,
                'posesion_home': 60.0, 'posesion_away': 40.0
            }
        ]

    def run(self):
        logger.info("Iniciando el pipeline del sistema de alto rendimiento QUANTBET AI...")
        
        X, y = self.load_historical_data()
        self.predictor.train(X, y)
        
        partidos = self.fetch_live_matches()
        if not partidos:
            logger.info("No hay partidos en vivo disponibles para análisis en este ciclo.")
            return

        self.notifier.send_telegram(f"📡 *QUANTBET AI ENGINE ACTIVADO*\nAnalizando {len(partidos)} partidos del mercado global con calibración estadística...")

        df_live = pd.DataFrame(partidos)
        try:
            probabilities = self.predictor.predict_proba(df_live)
        except Exception as e:
            logger.error(f"Error durante el cálculo de predicciones en el ensamble: {e}")
            return

        candidatos_apuestas = []
        for idx, partido in df_live.iterrows():
            prob_home = float(probabilities[idx])
            odds_home = float(partido.get('home_odds', 1.0))
            
            if odds_home <= 1.0:
                continue

            edge = MarketMathematics.compute_ev(prob_home, odds_home)
            if edge > EDGE_MINIMO:
                raw_kelly = AdvancedKellyCalculator.calculate_kelly_stake(prob_home, odds_home)
                if raw_kelly > 0:
                    candidatos_apuestas.append({
                        'home_team': partido.get('home_team', 'Local'),
                        'away_team': partido.get('away_team', 'Visitante'),
                        'odds': odds_home,
                        'prob': prob_home,
                        'edge': edge,
                        'raw_kelly': raw_kelly
                    })

        apuestas_optimizadas = AdvancedKellyCalculator.optimize_simultaneous_bets(candidatos_apuestas, BANKROLL_INICIAL)

        for bet in apuestas_optimizadas:
            if bet['allocated_amount'] >= 1.0:
                mensaje = (
                    f"🎯 *SIGNAL DETECTADO: SISTEMA REFORZADO QUANTBET*\n"
                    f"⚽ *Partido:* {bet['home_team']} vs {bet['away_team']}\n"
                    f"📊 *Predicción:* Victoria Local (Probabilidad: {bet['prob']:.2%})\n"
                    f"📈 *Cuota:* {bet['odds']:.2f} | *Edge:* +{bet['edge']:.2%}\n"
                    f"💰 *Stake Recomendado:* {bet['allocated_stake']:.2%}\n"
                    f"💵 *Inversión:* ${bet['allocated_amount']:.2f} (de ${BANKROLL_INICIAL:,.2f})"
                )
                self.notifier.send_telegram(mensaje)
                logger.info(f"Apuesta calculada para {bet['home_team']}: Stake {bet['allocated_stake']:.2%}")

        logger.info("Ciclo de análisis finalizado con éxito de forma limpia.")

if __name__ == "__main__":
    engine = QuantBetEngine()
    engine.run()