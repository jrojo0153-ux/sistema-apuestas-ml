import os
import sys
import logging
from dataclasses import dataclass, field
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss
from typing import List, Dict, Tuple, Optional, Any
from scipy.optimize import minimize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("QuantBetAI")

@dataclass
class EngineConfig:
    telegram_bot_token: Optional[str] = field(default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN"))
    telegram_chat_id: Optional[str] = field(default_factory=lambda: os.getenv("TELEGRAM_CHAT_ID"))
    bankroll_inicial: float = field(default_factory=lambda: float(os.getenv("BANKROLL_INICIAL", "10000.0")))
    kelly_fraccion: float = field(default_factory=lambda: float(os.getenv("KELLY_FRACCION", "0.20")))
    edge_minimo: float = field(default_factory=lambda: float(os.getenv("EDGE_MINIMO", "0.025")))
    max_exposicion_total: float = field(default_factory=lambda: float(os.getenv("MAX_EXPOSICION_TOTAL", "0.20")))
    max_apuesta_individual: float = field(default_factory=lambda: float(os.getenv("MAX_APUESTA_INDIVIDUAL", "0.05")))
    min_prob_threshold: float = 0.15
    n_splits: int = 5
    random_seed: int = 42

class MarketMathematics:
    @staticmethod
    def remove_margin_power_method(odds: List[float], max_iter: int = 100, tol: float = 1e-10) -> np.ndarray:
        if not odds or any(o <= 1.0 for o in odds):
            return np.array([])
        
        implied_probs = np.array([1.0 / o for o in odds])
        sum_implied = np.sum(implied_probs)
        if abs(sum_implied - 1.0) < tol:
            return implied_probs
            
        k_low, k_high = 0.0, 10.0
        for _ in range(max_iter):
            k = (k_low + k_high) / 2.0
            probs = implied_probs ** k
            sum_probs = np.sum(probs)
            if abs(sum_probs - 1.0) < tol:
                return probs
            if sum_probs > 1.0:
                k_low = k
            else:
                k_high = k
        return implied_probs / sum_implied

    @staticmethod
    def compute_ev(prob: float, odds: float) -> float:
        return (prob * odds) - 1.0

class FeatureEngineering:
    def __init__(self):
        self.scaler = RobustScaler()
        self.is_fitted = False
        self.feature_cols = [
            'implied_prob_home', 'implied_prob_away', 'market_margin', 
            'clean_prob_home', 'clean_prob_away', 'diff_goles_historico', 
            'ratio_goles', 'diff_posesion', 'home_relative_strength',
            'log_odds_ratio', 'total_implied_power', 'home_field_advantage_index'
        ]

    def build_features(self, df: pd.DataFrame, training: bool = False) -> pd.DataFrame:
        df = df.copy()
        
        if 'home_odds' in df.columns and 'away_odds' in df.columns:
            df['implied_prob_home'] = 1.0 / df['home_odds']
            df['implied_prob_away'] = 1.0 / df['away_odds']
            df['market_margin'] = (df['implied_prob_home'] + df['implied_prob_away']) - 1.0
            
            probs = df.apply(lambda row: MarketMathematics.remove_margin_power_method([row['home_odds'], row['away_odds']]), axis=1)
            df['clean_prob_home'] = [p[0] if len(p) > 0 else 0.5 for p in probs]
            df['clean_prob_away'] = [p[1] if len(p) > 0 else 0.5 for p in probs]
            
            df['log_odds_ratio'] = np.log(df['home_odds'] / df['away_odds']).fillna(0.0)
            df['total_implied_power'] = df['implied_prob_home'] * df['implied_prob_away']
        else:
            df['implied_prob_home'] = 0.5
            df['implied_prob_away'] = 0.5
            df['market_margin'] = 0.05
            df['clean_prob_home'] = 0.5
            df['clean_prob_away'] = 0.5
            df['log_odds_ratio'] = 0.0
            df['total_implied_power'] = 0.25

        if 'goles_local_historico' in df.columns and 'goles_visitante_historico' in df.columns:
            df['diff_goles_historico'] = df['goles_local_historico'] - df['goles_visitante_historico']
            df['ratio_goles'] = (df['goles_local_historico'] + 1) / (df['goles_visitante_historico'] + 1)
            df['home_relative_strength'] = df['goles_local_historico'] / (df['goles_local_historico'] + df['goles_visitante_historico'] + 1e-5)
        else:
            df['diff_goles_historico'] = 0.0
            df['ratio_goles'] = 1.0
            df['home_relative_strength'] = 0.5

        if 'posesion_home' in df.columns and 'posesion_away' in df.columns:
            df['diff_posesion'] = df['posesion_home'] - df['posesion_away']
        else:
            df['diff_posesion'] = 0.0

        df['home_field_advantage_index'] = (df['clean_prob_home'] * 1.1) / (df['clean_prob_away'] + 1e-5)

        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0.0

        df_features = df[self.feature_cols].fillna(0.0)

        if training:
            self.scaler.fit(df_features)
            self.is_fitted = True
            
        if self.is_fitted:
            scaled_array = self.scaler.transform(df_features)
            return pd.DataFrame(scaled_array, columns=self.feature_cols, index=df.index)
        
        return df_features

class EnsemblePredictor:
    def __init__(self, config: EngineConfig):
        self.config = config
        self.feature_pipeline = FeatureEngineering()
        self.models_xgb: List[CalibratedClassifierCV] = []
        self.models_lgb: List[CalibratedClassifierCV] = []
        self.is_trained = False
        
    def train(self, X: pd.DataFrame, y: np.ndarray):
        logger.info("Construyendo y preprocesando features de entrenamiento de alta dimensión...")
        X_features = self.feature_pipeline.build_features(X, training=True)
        
        skf = StratifiedKFold(n_splits=self.config.n_splits, shuffle=True, random_state=self.config.random_seed)
        
        oof_preds_xgb = np.zeros(len(X_features))
        oof_preds_lgb = np.zeros(len(X_features))
        
        xgb_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 4,
            'learning_rate': 0.02,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'random_state': self.config.random_seed,
            'n_jobs': -1
        }
        
        lgb_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'learning_rate': 0.02,
            'num_leaves': 15,
            'max_depth': 4,
            'feature_fraction': 0.85,
            'bagging_fraction': 0.85,
            'bagging_freq': 1,
            'random_state': self.config.random_seed,
            'verbose': -1,
            'n_jobs': -1
        }

        logger.info("Iniciando validación cruzada estratificada (K-Fold CV) de alta precisión...")
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_features, y)):
            X_tr, y_tr = X_features.iloc[train_idx], y[train_idx]
            X_va, y_va = X_features.iloc[val_idx], y[val_idx]
            
            base_xgb = xgb.XGBClassifier(**xgb_params, n_estimators=600)
            calibrated_xgb = CalibratedClassifierCV(estimator=base_xgb, method='isotonic', cv=3)
            calibrated_xgb.fit(X_tr, y_tr)
            self.models_xgb.append(calibrated_xgb)
            oof_preds_xgb[val_idx] = calibrated_xgb.predict_proba(X_va)[:, 1]
            
            base_lgb = lgb.LGBMClassifier(**lgb_params, n_estimators=600)
            calibrated_lgb = CalibratedClassifierCV(estimator=base_lgb, method='isotonic', cv=3)
            calibrated_lgb.fit(X_tr, y_tr)
            self.models_lgb.append(calibrated_lgb)
            oof_preds_lgb[val_idx] = calibrated_lgb.predict_proba(X_va)[:, 1]
            
            fold_auc = roc_auc_score(y_va, (oof_preds_xgb[val_idx] + oof_preds_lgb[val_idx]) / 2.0)
            logger.info(f"Fold {fold+1}/{self.config.n_splits} completado. Val AUC: {fold_auc:.4f}")

        ensemble_oof = (oof_preds_xgb * 0.5) + (oof_preds_lgb * 0.5)
        self.is_trained = True
        
        metric_brier = brier_score_loss(y, ensemble_oof)
        metric_auc = roc_auc_score(y, ensemble_oof)
        metric_logloss = log_loss(y, ensemble_oof)
        
        logger.info("=========================================")
        logger.info(f"RESULTADOS OOF ENSAMBLE:")
        logger.info(f"Brier Score: {metric_brier:.5f}")
        logger.info(f"ROC AUC:     {metric_auc:.5f}")
        logger.info(f"Log Loss:    {metric_logloss:.5f}")
        logger.info("=========================================")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("El ensamble de modelos predictivos no está entrenado.")
            
        X_features = self.feature_pipeline.build_features(X, training=False)
        preds_xgb = np.mean([model.predict_proba(X_features)[:, 1] for model in self.models_xgb], axis=0)
        preds_lgb = np.mean([model.predict_proba(X_features)[:, 1] for model in self.models_lgb], axis=0)
        
        return (preds_xgb * 0.5) + (preds_lgb * 0.5)

class AdvancedKellyCalculator:
    @staticmethod
    def calculate_single_kelly(prob: float, odds: float, fraction: float, max_individual: float) -> float:
        if odds <= 1.0:
            return 0.0
        edge = MarketMathematics.compute_ev(prob, odds)
        if edge <= 0:
            return 0.0
        kelly_f = edge / (odds - 1.0)
        return float(np.clip(kelly_f * fraction, 0.0, max_individual))

    @staticmethod
    def optimize_portfolio_kelly(bets: List[Dict[str, Any]], bankroll: float, config: EngineConfig) -> List[Dict[str, Any]]:
        if not bets:
            return []
            
        n_bets = len(bets)
        raw_kellys = np.array([b['raw_kelly'] for b in bets])
        edges = np.array([b['edge'] for b in bets])
        odds = np.array([b['odds'] for b in bets])
        probs = np.array([b['prob'] for b in bets])
        
        def objective_function(f):
            log_growth = 0.0
            combinations = np.array(np.meshgrid(*[[0, 1] for _ in range(n_bets)])).T.reshape(-1, n_bets)
            for comb in combinations:
                p_comb = 1.0
                return_factor = 1.0 - np.sum(f)
                for i in range(n_bets):
                    p_event = probs[i] if comb[i] == 1 else (1.0 - probs[i])
                    p_comb *= p_event
                    if comb[i] == 1:
                        return_factor += f[i] * odds[i]
                if return_factor <= 0:
                    return 1e10
                log_growth += p_comb * np.log(return_factor)
            return -log_growth

        init_weights = raw_kellys / np.sum(raw_kellys + 1e-9) * min(np.sum(raw_kellys), config.max_exposicion_total)
        bounds = [(0.0, config.max_apuesta_individual) for _ in range(n_bets)]
        constraints = ({'type': 'ineq', 'fun': lambda f: config.max_exposicion_total - np.sum(f)})

        res = minimize(objective_function, init_weights, bounds=bounds, constraints=constraints, method='SLSQP')
        
        opt_weights = res.x if res.success else init_weights
        
        for idx, bet in enumerate(bets):
            allocated_stake = float(opt_weights[idx])
            bet['allocated_stake'] = allocated_stake if allocated_stake >= 0.005 else 0.0
            bet['allocated_amount'] = bet['allocated_stake'] * bankroll
            
        return [b for b in bets if b['allocated_stake'] > 0]

class NotificationManager:
    def __init__(self, config: EngineConfig):
        self.config = config
        self.session = requests.Session()
        retries = Retry(total=5, backoff_factor=1.5, status_forcelist=[500, 502, 503, 504])
        self.session.mount("https://", HTTPAdapter(max_retries=retries))

    def send_telegram(self, message: str):
        if not self.config.telegram_bot_token or not self.config.telegram_chat_id:
            logger.warning("Telegram Bot Token o Chat ID ausente en la configuración.")
            return

        url = f"https://api.telegram.org/bot{self.config.telegram_bot_token}/sendMessage"
        payload = {
            "chat_id": self.config.telegram_chat_id,
            "text": message,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True
        }
        try:
            response = self.session.post(url, json=payload, timeout=12)
            response.raise_for_status()
        except Exception as e:
            logger.error(f"Error en la comunicación con la API de Telegram: {e}")

class QuantBetEngine:
    def __init__(self):
        self.config = EngineConfig()
        self.predictor = EnsemblePredictor(self.config)
        self.notifier = NotificationManager(self.config)

    def load_historical_data(self) -> Tuple[pd.DataFrame, np.ndarray]:
        synth_size = 2500
        np.random.seed(self.config.random_seed)
        
        data = {
            'home_odds': np.random.uniform(1.4, 4.2, synth_size),
            'away_odds': np.random.uniform(1.6, 5.0, synth_size),
            'goles_local_historico': np.random.poisson(1.85, synth_size),
            'goles_visitante_historico': np.random.poisson(1.15, synth_size),
            'posesion_home': np.random.uniform(38.0, 62.0, synth_size),
            'posesion_away': np.random.uniform(38.0, 62.0, synth_size),
        }
        df = pd.DataFrame(data)
        
        probs = df.apply(lambda row: MarketMathematics.remove_margin_power_method([row['home_odds'], row['away_odds']]), axis=1)
        prob_home_clean = np.array([p[0] for p in probs])
        
        noise = np.random.normal(0, 0.08, synth_size)
        y = np.where(prob_home_clean + noise > 0.49, 1, 0)
        return df, y

    def fetch_live_matches(self) -> List[Dict[str, Any]]:
        try:
            from data.live_matches import obtener_partidos
            partidos = obtener_partidos()
            if partidos:
                return partidos
        except ImportError:
            logger.warning("No se detectó un pipeline de ingesta en vivo. Iniciando modo de simulación de alta calidad...")
        
        return [
            {
                'home_team': 'Real Madrid', 'away_team': 'Barcelona',
                'home_odds': 2.05, 'away_odds': 3.10,
                'goles_local_historico': 2.3, 'goles_visitante_historico': 1.9,
                'posesion_home': 53.0, 'posesion_away': 47.0
            },
            {
                'home_team': 'Manchester City', 'away_team': 'Arsenal',
                'home_odds': 1.75, 'away_odds': 4.10,
                'goles_local_historico': 2.6, 'goles_visitante_historico': 1.5,
                'posesion_home': 59.5, 'posesion_away': 40.5
            },
            {
                'home_team': 'Bayern Munich', 'away_team': 'Leverkusen',
                'home_odds': 1.90, 'away_odds': 3.60,
                'goles_local_historico': 2.7, 'goles_visitante_historico': 2.1,
                'posesion_home': 56.5, 'posesion_away': 43.5
            },
            {
                'home_team': 'PSG', 'away_team': 'Marseille',
                'home_odds': 1.48, 'away_odds': 5.80,
                'goles_local_historico': 2.9, 'goles_visitante_historico': 1.2,
                'posesion_home': 61.0, 'posesion_away': 39.0
            },
            {
                'home_team': 'Inter Milan', 'away_team': 'Juventus',
                'home_odds': 2.10, 'away_odds': 3.40,
                'goles_local_historico': 1.9, 'goles_visitante_historico': 1.1,
                'posesion_home': 51.5, 'posesion_away': 48.5
            }
        ]

    def run(self):
        logger.info("==================================================================")
        logger.info("INICIANDO MOTOR QUANTBET DE APRENDIZAJE AUTOMÁTICO Y ASIGNACIÓN COPORTUNISTA")
        logger.info("==================================================================")
        
        X_hist, y_hist = self.load_historical_data()
        self.predictor.train(X_hist, y_hist)
        
        partidos = self.fetch_live_matches()
        if not partidos:
            logger.info("Sin flujo de eventos en tiempo real para procesar.")
            return

        self.notifier.send_telegram(
            f"📡 *QUANTBET SYSTEM ACTIVE*\n"
            f"Analizando {len(partidos)} oportunidades de mercado utilizando calibración de probabilidad multi-algoritmo..."
        )

        df_live = pd.DataFrame(partidos)
        try:
            probabilities = self.predictor.predict_proba(df_live)
        except Exception as e:
            logger.error(f"Fallo crítico en la generación de inferencias del ensamble: {e}")
            return

        candidatos_apuestas = []
        for idx, partido in df_live.iterrows():
            prob_home = float(probabilities[idx])
            odds_home = float(partido.get('home_odds', 1.0))
            
            if odds_home <= 1.0 or prob_home < self.config.min_prob_threshold:
                continue

            edge = MarketMathematics.compute_ev(prob_home, odds_home)
            if edge >= self.config.edge_minimo:
                raw_kelly = AdvancedKellyCalculator.calculate_single_kelly(
                    prob_home, odds_home, self.config.kelly_fraccion, self.config.max_apuesta_individual
                )
                if raw_kelly > 0:
                    candidatos_apuestas.append({
                        'home_team': partido.get('home_team', 'Home'),
                        'away_team': partido.get('away_team', 'Away'),
                        'odds': odds_home,
                        'prob': prob_home,
                        'edge': edge,
                        'raw_kelly': raw_kelly
                    })

        apuestas_optimizadas = AdvancedKellyCalculator.optimize_portfolio_kelly(
            candidatos_apuestas, self.config.bankroll_inicial, self.config
        )

        if not apuestas_optimizadas:
            logger.info("Análisis de mercado finalizado: No se identificaron apuestas que cumplan con los criterios de Edge y Kelly.")
            self.notifier.send_telegram("⚠️ *MONITOR DE MERCADO*: No se encontraron apuestas con EV+ y tamaño de Kelly óptimo en este ciclo.")
            return

        logger.info(f"Portafolio optimizado calculado. Ejecutando alertas para {len(apuestas_optimizadas)} señales con EV+.")
        
        for bet in apuestas_optimizadas:
            mensaje = (
                f"🎯 *SEÑAL DE ALTO RENDIMIENTO QUANTBET*\n"
                f"⚽ *Partido:* {bet['home_team']} vs {bet['away_team']}\n"
                f"📊 *Predicción:* Victoria Local (Probabilidad: {bet['prob']:.2%})\n"
                f"📈 *Cuota:* {bet['odds']:.2f} | *Ventaja (Edge):* +{bet['edge']:.2%}\n"
                f"⚙️ *Fraccionamiento de Cartera:* {bet['allocated_stake']:.2%}\n"
                f"💰 *Monto Asignado:* ${bet['allocated_amount']:.2f} (Base: ${self.config.bankroll_inicial:,.2f})"
            )
            self.notifier.send_telegram(mensaje)
            logger.info(f"EJECUCIÓN -> {bet['home_team']} vs {bet['away_team']}: Stake {bet['allocated_stake']:.2%}, EV: {bet['edge']:.2%}")

        logger.info("Ciclo de inferencia y optimización de portafolio cerrado exitosamente.")

if __name__ == "__main__":
    engine = QuantBetEngine()
    engine.run()