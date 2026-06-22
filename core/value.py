import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Any, List, Optional
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("MLBettingSystem")

class FeatureEngineer:
    """
    Ingeniería de variables avanzada para maximizar el Alpha y explotar ineficiencias del mercado.
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_feat = self._create_features(df)
        numerical_cols = df_feat.select_dtypes(include=[np.number]).columns.tolist()
        if "target" in numerical_cols:
            numerical_cols.remove("target")
        
        df_feat[numerical_cols] = self.scaler.fit_transform(df_feat[numerical_cols].fillna(0))
        self.is_fitted = True
        return df_feat

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("FeatureEngineer debe ser ajustado (fit) antes de transformar.")
        df_feat = self._create_features(df)
        numerical_cols = df_feat.select_dtypes(include=[np.number]).columns.tolist()
        if "target" in numerical_cols:
            numerical_cols.remove("target")
            
        df_feat[numerical_cols] = self.scaler.transform(df_feat[numerical_cols].fillna(0))
        return df_feat

    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Implied Probabilities & Market Margin
        df['implied_prob_home'] = 1 / df['odds_home']
        df['implied_prob_away'] = 1 / df['odds_away']
        df['market_margin'] = (df['implied_prob_home'] + df['implied_prob_away']) - 1.0
        
        # Fair Odds (Sin Margen de la casa de apuestas)
        df['fair_implied_home'] = df['implied_prob_home'] / (df['implied_prob_home'] + df['implied_prob_away'])
        df['fair_odds_home'] = 1 / df['fair_implied_home']
        
        # Variables de Momentum / Diferenciales Históricos (asumiendo orden cronológico)
        if 'team_home_score' in df.columns and 'team_away_score' in df.columns:
            df['goal_diff'] = df['team_home_score'] - df['team_away_score']
            df['roll_goal_diff_home'] = df.groupby('team_home')['goal_diff'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
            df['roll_goal_diff_away'] = df.groupby('team_away')['goal_diff'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
            df['momentum_diff'] = df['roll_goal_diff_home'] - df['roll_goal_diff_away']
        
        # Discrepancias de valor percibido vs real mercado
        df['odds_ratio'] = df['odds_home'] / (df['odds_away'] + 1e-5)
        df['log_odds_ratio'] = np.log1p(df['odds_ratio'])
        
        # Rellenado de nulos críticos
        df.fillna(0, inplace=True)
        return df


class PredictiveModel:
    """
    Pipeline predictivo de ML con calibración de probabilidad para estimaciones de EV precisas.
    """
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = params or {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 5,
            'learning_rate': 0.03,
            'n_estimators': 350,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'use_label_encoder': False
        }
        self.base_model = xgb.XGBClassifier(**self.params)
        self.calibrated_model = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        logger.info("Entrenando clasificador XGBoost y calibrando probabilidades...")
        # Calibración de probabilidad (Isotonic/Sigmoid) crucial para el Criterio de Kelly
        self.calibrated_model = CalibratedClassifierCV(estimator=self.base_model, method='sigmoid', cv=5)
        self.calibrated_model.fit(X, y)
        logger.info("Entrenamiento completado exitosamente.")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.calibrated_model is None:
            raise ValueError("El modelo debe entrenarse antes de realizar predicciones.")
        return self.calibrated_model.predict_proba(X)[:, 1]


class AdvancedKellyManager:
    """
    Gestión de Riesgo y Asignación de Capital Dinámica con Criterio de Kelly Avanzado.
    Incluye Kelly Fraccional, protección contra Drawdown y penalización de volatilidad.
    """
    def __init__(self, 
                 fraction: float = 0.25, 
                 max_bet_fraction: float = 0.05, 
                 min_ev_threshold: float = 0.02,
                 drawdown_protection_threshold: float = 0.20):
        self.fraction = fraction  # Kelly Fraccional para reducir varianza (ej. 0.25 = Quarter Kelly)
        self.max_bet_fraction = max_bet_fraction  # Exposición máxima permitida por apuesta (ej. 5%)
        self.min_ev_threshold = min_ev_threshold  # Valor Esperado mínimo para ejecutar orden
        self.drawdown_protection_threshold = drawdown_protection_threshold

    def calculate_allocation(self, prob: float, odds: float, current_bankroll: float, drawdown: float = 0.0) -> Tuple[float, float, float]:
        """
        Calcula el EV y el tamaño óptimo de la apuesta con restricciones prudenciales.
        """
        if odds <= 1.0 or not (0.0 <= prob <= 1.0):
            return 0.0, 0.0, 0.0

        # Cálculo de Valor Esperado (EV)
        ev = (prob * odds) - 1.0

        # Filtro de EV mínimo
        if ev < self.min_ev_threshold:
            return ev, 0.0, 0.0

        # Fórmula estándar de Kelly: f* = (p * b - q) / b  donde b = odds - 1
        net_odds = odds - 1.0
        raw_kelly = ev / net_odds

        if raw_kelly <= 0:
            return ev, 0.0, 0.0

        # Aplicación de Kelly Fraccional
        suggested_fraction = raw_kelly * self.fraction

        # Protección dinámica frente a pérdidas acumuladas (Drawdown Multiplier)
        if drawdown > self.drawdown_protection_threshold:
            multiplier = max(0.0, 1.0 - (drawdown * 2))
            suggested_fraction *= multiplier

        # Límite máximo de riesgo por operación
        final_fraction = min(suggested_fraction, self.max_bet_fraction)
        wager_amount = final_fraction * current_bankroll

        return ev, final_fraction, wager_amount


class BettingSystemOrchestrator:
    """
    Orquestador principal que unifica ML Pipeline, Procesamiento de Datos y Execution Engine.
    """
    def __init__(self, bankroll_inicial: float = 10000.0, fraction_kelly: float = 0.25):
        self.bankroll = bankroll_inicial
        self.peak_bankroll = bankroll_inicial
        self.fe = FeatureEngineer()
        self.model = PredictiveModel()
        self.kelly_manager = AdvancedKellyManager(fraction=fraction_kelly)

    def get_drawdown(self) -> float:
        if self.bankroll > self.peak_bankroll:
            self.peak_bankroll = self.bankroll
        return (self.peak_bankroll - self.bankroll) / self.peak_bankroll

    def train_pipeline(self, historical_data: pd.DataFrame, target: pd.Series):
        logger.info("Iniciando pipeline de entrenamiento...")
        X_engineered = self.fe.fit_transform(historical_data)
        self.model.fit(X_engineered, target)

    def process_and_evaluate(self, live_data: pd.DataFrame) -> pd.DataFrame:
        """
        Procesa nuevos eventos en tiempo real, infiere probabilidades, calcula EV y asigna capital óptimo.
        """
        X_live = self.fe.transform(live_data)
        predicted_probs = self.model.predict_proba(X_live)
        
        results = []
        drawdown = self.get_drawdown()

        for idx, row in live_data.iterrows():
            prob = predicted_probs[idx]
            # Usamos odds_home para el ejemplo de evaluación de apuesta a victoria local
            odds = row['odds_home']
            
            ev, kelly_frac, wager = self.kelly_manager.calculate_allocation(
                prob=prob, 
                odds=odds, 
                current_bankroll=self.bankroll,
                drawdown=drawdown
            )
            
            results.append({
                'predicted_prob': prob,
                'ev': ev,
                'kelly_fraction': kelly_frac,
                'wager_amount': wager,
                'odds': odds
            })
            
        return pd.DataFrame(results)

    def update_bankroll(self, result: float):
        """
        Actualiza el estado de la banca de forma controlada según pnl neto de operaciones resueltas.
        """
        self.bankroll += result
        if self.bankroll > self.peak_bankroll:
            self.peak_bankroll = self.bankroll
        logger.info(f"Banca actualizada. Nuevo Saldo: {self.bankroll:.2f} | Drawdown: {self.get_drawdown()*100:.2f}%")