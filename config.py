import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, roc_auc_score
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional

class SystemConfig:
    BANKROLL_INICIAL: float = 1000.0
    KELLY_FRACCION: float = 0.05  # Más conservador para proteger contra varianza extrema
    APUESTA_MINIMA: float = 1.0
    APUESTA_MAXIMA_PCT: float = 0.03
    STOP_LOSS_PCT: float = 0.30
    MAX_APUESTAS_DIA: int = 5

    EV_MINIMO: float = 0.03
    PROB_MINIMA: float = 0.35
    PROB_MAXIMA: float = 0.85
    CUOTA_MINIMA: float = 1.35
    CUOTA_MAXIMA: float = 6.00
    EDGE_MINIMO: float = 0.015

    MODEL_PATH: Path = Path("models")
    MODEL_PATH.mkdir(parents=True, exist_ok=True)


class FeatureEngineer:
    @staticmethod
    def create_features(df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        df = df.copy()
        
        # 1. Probabilidades Implícitas por Mercado
        df['implied_prob'] = 1.0 / df['cuota']
        
        # 2. Features de Momentum y Rendimiento Histórico (Rolling Metrics)
        if 'team_id' in df.columns:
            df = df.sort_values(by=['team_id', 'timestamp'])
            for window in [3, 5, 10]:
                df[f'team_win_rate_last_{window}'] = df.groupby('team_id')['result'].transform(
                    lambda x: x.shift(1).rolling(window, min_periods=1).mean()
                )
                df[f'team_avg_goals_last_{window}'] = df.groupby('team_id')['goals_scored'].transform(
                    lambda x: x.shift(1).rolling(window, min_periods=1).mean()
                )
                df[f'team_conceded_avg_last_{window}'] = df.groupby('team_id')['goals_conceded'].transform(
                    lambda x: x.shift(1).rolling(window, min_periods=1).mean()
                )
        
        # 3. Diferencias de Calidad (H2H & Ratings de Mercado)
        if 'opponent_rating' in df.columns and 'team_rating' in df.columns:
            df['rating_diff'] = df['team_rating'] - df['opponent_rating']
            df['expected_margin'] = df['rating_diff'] * 0.1
            
        # 4. Features basadas en Cuotas (Presión de Mercado)
        if 'cuota_apertura' in df.columns:
            df['odds_drift'] = df['cuota'] - df['cuota_apertura']
            df['odds_drift_pct'] = (df['cuota'] - df['cuota_apertura']) / df['cuota_apertura']
            
        # 5. Volatilidad de mercado
        if 'cuota_std' in df.columns:
            df['market_instability'] = df['cuota_std'] / df['cuota']
            
        # Rellenar valores nulos resultantes de operaciones temporales
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        return df


class PredictiveEnsemble:
    def __init__(self):
        self.xgb_model = None
        self.lgb_model = None
        self.scaler = StandardScaler()
        self.is_trained = False

    def train(self, X: pd.DataFrame, y: pd.Series):
        X_scaled = self.scaler.fit_transform(X)
        
        # Configuración robusta para evitar Overfitting
        xgb_base = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        )
        
        lgb_base = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            verbose=-1,
            random_state=42,
            n_jobs=-1
        )

        # Calibración de probabilidad esencial para Criterio de Kelly preciso
        self.xgb_model = CalibratedClassifierCV(estimator=xgb_base, method='isotonic', cv=3)
        self.lgb_model = CalibratedClassifierCV(estimator=lgb_base, method='isotonic', cv=3)

        self.xgb_model.fit(X_scaled, y)
        self.lgb_model.fit(X_scaled, y)
        self.is_trained = True

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado antes de realizar predicciones.")
        X_scaled = self.scaler.transform(X)
        xgb_preds = self.xgb_model.predict_proba(X_scaled)[:, 1]
        lgb_preds = self.lgb_model.predict_proba(X_scaled)[:, 1]
        
        # Ensamble ponderado por media geométrica o media simple robusta
        return (xgb_preds + lgb_preds) / 2.0

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        preds = self.predict_proba(X)
        return {
            "brier_score": brier_score_loss(y, preds),
            "auc_roc": roc_auc_score(y, preds)
        }

    def save_model(self, path: Path):
        joblib.dump({"xgb": self.xgb_model, "lgb": self.lgb_model, "scaler": self.scaler}, path / "ensemble_model.pkl")

    def load_model(self, path: Path):
        checkpoint = joblib.load(path / "ensemble_model.pkl")
        self.xgb_model = checkpoint["xgb"]
        self.lgb_model = checkpoint["lgb"]
        self.scaler = checkpoint["scaler"]
        self.is_trained = True


class AdvancedKellyOptimizer:
    @staticmethod
    def calculate_kelly_bet(prob_pred: float, cuota: float, current_bankroll: float) -> float:
        """
        Criterio de Kelly Avanzado fraccionado con límites dinámicos y control de drawdown.
        """
        if cuota <= 1.0:
            return 0.0
        
        # Cálculo de ventaja (Edge)
        b = cuota - 1.0
        p = prob_pred
        q = 1.0 - p
        
        kelly_f = (p * (b + 1) - 1) / b
        
        # Si no hay ventaja matemática, no se apuesta
        if kelly_f <= 0:
            return 0.0
            
        # Aplicación de Fracción Kelly (atenuación de riesgo)
        adjusted_bet_pct = kelly_f * SystemConfig.KELLY_FRACCION
        
        # Control estricto de exposición por apuesta única
        max_bet_allowed = current_bankroll * SystemConfig.APUESTA_MAXIMA_PCT
        final_bet = min(adjusted_bet_pct * current_bankroll, max_bet_allowed)
        
        # Comprobar límite mínimo absoluto
        if final_bet < SystemConfig.APUESTA_MINIMA:
            return 0.0
            
        return round(final_bet, 2)

    @staticmethod
    def optimize_simultaneous_bets(bets: List[Dict], current_bankroll: float) -> List[Dict]:
        """
        Algoritmo de optimización de cartera para eventos concurrentes.
        Aplica contracción cuando la exposición agregada supera los umbrales lógicos del sistema.
        """
        if not bets:
            return []
            
        total_risk_pct = sum([b['kelly_bet'] / current_bankroll for b in bets])
        max_total_exposure = SystemConfig.APUESTA_MAXIMA_PCT * SystemConfig.MAX_APUESTAS_DIA
        
        # Si la suma de apuestas excede el límite agregado de riesgo, escalamos linealmente
        if total_risk_pct > max_total_exposure:
            scale_factor = max_total_exposure / total_risk_pct
            for b in bets:
                b['final_bet'] = max(round(b['kelly_bet'] * scale_factor, 2), SystemConfig.APUESTA_MINIMA)
        else:
            for b in bets:
                b['final_bet'] = b['kelly_bet']
                
        return bets


class MLBettingEngine:
    def __init__(self, current_bankroll: float = SystemConfig.BANKROLL_INICIAL):
        self.bankroll = current_bankroll
        self.stop_loss_limit = SystemConfig.BANKROLL_INICIAL * (1.0 - SystemConfig.STOP_LOSS_PCT)
        self.ensemble = PredictiveEnsemble()
        self.fe = FeatureEngineer()

    def check_stop_loss(self) -> bool:
        return self.bankroll <= self.stop_loss_limit

    def execute_betting_cycle(self, market_data: pd.DataFrame) -> List[Dict]:
        if self.check_stop_loss():
            return []  # Circuito de parada de emergencia activado
            
        # 1. Feature Engineering
        processed_data = self.fe.create_features(market_data, is_training=False)
        
        features_cols = [col for col in processed_data.columns if col not in ['target', 'result', 'timestamp', 'team_id']]
        
        # 2. Inferencia de Probabilidades Predictivas Calibradas
        probabilities = self.ensemble.predict_proba(processed_data[features_cols])
        
        candidates = []
        for idx, row in processed_data.iterrows():
            prob = probabilities[idx]
            cuota = row['cuota']
            implied = 1.0 / cuota
            edge = prob - implied
            ev = (prob * cuota) - 1.0
            
            # 3. Filtrado de Valor por Edge y Probabilidad Mínima
            if (prob >= SystemConfig.PROB_MINIMA and 
                prob <= SystemConfig.PROB_MAXIMA and 
                cuota >= SystemConfig.CUOTA_MINIMA and 
                cuota <= SystemConfig.CUOTA_MAXIMA and 
                edge >= SystemConfig.EDGE_MINIMO and 
                ev >= SystemConfig.EV_MINIMO):
                
                kelly_bet = AdvancedKellyOptimizer.calculate_kelly_bet(prob, cuota, self.bankroll)
                
                if kelly_bet > 0:
                    candidates.append({
                        'event_id': row.get('event_id', idx),
                        'cuota': cuota,
                        'pred_prob': prob,
                        'edge': edge,
                        'expected_value': ev,
                        'kelly_bet': kelly_bet
                    })
        
        # 4. Optimización de cartera con restricciones agregadas
        candidates = sorted(candidates, key=lambda x: x['expected_value'], reverse=True)[:SystemConfig.MAX_APUESTAS_DIA]
        final_bets = AdvancedKellyOptimizer.optimize_simultaneous_bets(candidates, self.bankroll)
        
        return final_bets

    def update_bankroll(self, settled_bets: List[Dict]):
        """
        Actualiza el estado de la banca basándose en los resultados reales de las apuestas asentadas.
        """
        for bet in settled_bets:
            if bet['won']:
                self.bankroll += bet['amount'] * (bet['cuota'] - 1)
            else:
                self.bankroll -= bet['amount']
        
        # Ajuste dinámico de parámetros según la fluctuación de capital
        if self.bankroll < SystemConfig.BANKROLL_INICIAL * 0.8:
            # Estrategia defensiva temporal: Reducimos fracción de Kelly
            SystemConfig.KELLY_FRACCION = 0.03
        else:
            SystemConfig.KELLY_FRACCION = 0.05