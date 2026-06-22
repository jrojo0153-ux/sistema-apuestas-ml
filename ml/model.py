import os
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Union, Optional
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
from xgboost import XGBClassifier

# Configure professional logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ML_Portfolio_System")

class FeatureEngineer:
    @staticmethod
    def transform(df: pd.DataFrame, is_training: bool = True) -> Union[Tuple[pd.DataFrame, pd.Series], pd.DataFrame]:
        df = df.copy()
        
        required_cols = ['home_odds', 'away_odds']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
                
        df['home_odds'] = pd.to_numeric(df['home_odds'], errors='coerce')
        df['away_odds'] = pd.to_numeric(df['away_odds'], errors='coerce')
        
        # Core Implied Probabilities
        df['implied_prob_home'] = 1.0 / df['home_odds']
        df['implied_prob_away'] = 1.0 / df['away_odds']
        
        # Overround / Bookmaker Margin
        df['bookmaker_margin'] = (df['implied_prob_home'] + df['implied_prob_away']) - 1.0
        
        # Margin-adjusted Fair (Clean) Probabilities (Power & Shin calibration approximations)
        df['clean_prob_home'] = df['implied_prob_home'] / (df['implied_prob_home'] + df['implied_prob_away'])
        df['clean_prob_away'] = df['implied_prob_away'] / (df['implied_prob_home'] + df['implied_prob_away'])
        
        # Advanced Market Features
        df['odds_ratio'] = df['home_odds'] / df['away_odds']
        df['log_odds_ratio'] = np.log(df['odds_ratio'] + 1e-9)
        df['odds_diff'] = df['home_odds'] - df['away_odds']
        df['odds_sum'] = df['home_odds'] + df['away_odds']
        
        # Market Shannon Entropy
        df['entropy'] = - (df['clean_prob_home'] * np.log(df['clean_prob_home'] + 1e-9) + 
                           df['clean_prob_away'] * np.log(df['clean_prob_away'] + 1e-9))
        
        # Nonlinear Interactions & Market Sentiment Indicators
        df['market_polarization'] = (df['clean_prob_home'] - df['clean_prob_away']).abs()
        df['margin_to_odds_ratio_home'] = df['bookmaker_margin'] / (df['home_odds'] + 1e-9)
        df['margin_to_odds_ratio_away'] = df['bookmaker_margin'] / (df['away_odds'] + 1e-9)
        df['log_implied_ratio'] = np.log(df['implied_prob_home'] / (df['implied_prob_away'] + 1e-9) + 1e-9)
        df['skewness_proxy'] = (df['home_odds'] - df['away_odds']) / (df['odds_sum'] + 1e-9)
        
        # Logit transforms of probabilities
        df['logit_clean_home'] = np.log(df['clean_prob_home'] / (1.0 - df['clean_prob_home'] + 1e-9) + 1e-9)
        
        features = [
            'home_odds', 'away_odds', 'implied_prob_home', 'implied_prob_away',
            'bookmaker_margin', 'clean_prob_home', 'clean_prob_away',
            'odds_ratio', 'log_odds_ratio', 'odds_diff', 'odds_sum', 'entropy',
            'market_polarization', 'margin_to_odds_ratio_home', 'margin_to_odds_ratio_away',
            'log_implied_ratio', 'skewness_proxy', 'logit_clean_home'
        ]
        
        if is_training:
            if 'resultado' not in df.columns:
                raise ValueError("Target column 'resultado' is required for training.")
            df['target'] = df['resultado'].astype(int)
            return df[features], df['target']
            
        return df[features]

class KellyPortfolioManager:
    def __init__(self, fraction: float = 0.15, max_bet_fraction: float = 0.04, min_edge: float = 0.015, risk_free_rate: float = 0.0):
        # Using a conservative fractional Kelly (e.g. 0.15 - 0.25) to protect bankroll against model uncertainty
        self.fraction = fraction
        self.max_bet_fraction = max_bet_fraction
        self.min_edge = min_edge
        self.risk_free_rate = risk_free_rate

    def calculate_bet_size(self, decimal_odds: float, predicted_prob: float, uncertainty_scale: float = 1.0) -> float:
        """
        Calculates the optimal bet size using an advanced adjusted Fractional Kelly Criterion.
        Incorporates model confidence/uncertainty scaling.
        """
        if decimal_odds <= 1.0 or predicted_prob <= 0.0 or predicted_prob >= 1.0:
            return 0.0
        
        implied_prob = 1.0 / decimal_odds
        edge = predicted_prob - implied_prob
        
        # Zero allocation if edge is below minimum threshold (margin of safety)
        if edge < self.min_edge:
            return 0.0
            
        b = decimal_odds - 1.0
        q = 1.0 - predicted_prob
        
        # Standard Kelly Formula: f* = (bp - q) / b
        kelly_f = (predicted_prob * b - q) / b
        
        if kelly_f <= 0.0:
            return 0.0
            
        # Apply fractional Kelly scaling with uncertainty modifier (higher model uncertainty -> smaller bet)
        suggested_bet = kelly_f * self.fraction * uncertainty_scale
        
        # Strict protection caps
        return float(np.clip(suggested_bet, 0.0, self.max_bet_fraction))

class PredictiveModel:
    def __init__(self, model_dir: str = 'models'):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.model_dir / 'calibrated_xgb_v2.pkl'
        self.model = None

    def train(self, X: pd.DataFrame, y: pd.Series) -> CalibratedClassifierCV:
        # Advanced hyperparameters optimized to handle extreme betting market noise and prevent overfitting
        base_estimator = XGBClassifier(
            n_estimators=450,
            max_depth=3,                  # Shallower trees are highly effective for noisy tabular data
            learning_rate=0.015,          # Lower learning rate for maximum generalization
            subsample=0.7,
            colsample_bytree=0.7,
            min_child_weight=5,
            gamma=0.2,
            reg_alpha=1.5,                # L1 regularization to sparsify feature impact
            reg_lambda=3.0,               # L2 regularization to smooth predictions
            scale_pos_weight=1.0,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            n_jobs=-1
        )
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Calibrate probabilities explicitly to output clean expected values critical for Kelly
        self.model = CalibratedClassifierCV(
            estimator=base_estimator,
            method='isotonic',            # Isotonic calibration fits well with ample data
            cv=cv
        )
        
        logger.info("Fitting and calibrating the ensemble classifier...")
        self.model.fit(X, y)
        
        try:
            train_preds = self.model.predict_proba(X)[:, 1]
            loss = log_loss(y, train_preds)
            auc = roc_auc_score(y, train_preds)
            brier = brier_score_loss(y, train_preds)
            logger.info(f"📊 Model Diagnostics | LogLoss: {loss:.5f} | AUC-ROC: {auc:.5f} | Brier Score: {brier:.5f}")
        except Exception as e:
            logger.warning(f"Failed to calculate detailed metrics: {e}")
            
        joblib.dump(self.model, self.model_path)
        logger.info(f"✅ Production-ready calibrated model saved at: {self.model_path}")
        return self.model

    def load(self) -> bool:
        if self.model_path.exists():
            try:
                self.model = joblib.load(self.model_path)
                logger.info("✅ Model loaded successfully from disk.")
                return True
            except Exception as e:
                logger.error(f"❌ Failed to load model: {e}")
                return False
        return False

    def predict_probability(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            if not self.load():
                raise RuntimeError("Model is uninitialized. Run training pipeline or supply a valid path.")
        return self.model.predict_proba(X)[:, 1]

def entrenar_modelo() -> Optional[CalibratedClassifierCV]:
    ruta_csv = Path('data/Historico.csv')
    if not ruta_csv.exists():
        logger.error(f"❌ Traindata file not found: {ruta_csv}")
        return None
        
    try:
        df = pd.read_csv(ruta_csv).dropna(subset=['home_odds', 'away_odds', 'resultado'])
        X, y = FeatureEngineer.transform(df, is_training=True)
        
        trainer = PredictiveModel()
        modelo = trainer.train(X, y)
        return modelo
    except Exception as e:
        logger.critical(f"❌ Critical error during operational model training: {e}", exc_info=True)
        return None

def cargar_modelo() -> Optional[PredictiveModel]:
    trainer = PredictiveModel()
    if trainer.load():
        return trainer
    return None

def predecir(modelo_wrapper, partido: dict, bankroll: float = 1000.0) -> dict:
    try:
        h = partido.get('home_odds')
        a = partido.get('away_odds')
        if h is None or a is None:
            return {"error": "Missing essential odds data."}
            
        df_partido = pd.DataFrame([partido])
        X_pred = FeatureEngineer.transform(df_partido, is_training=False)
        
        # Predict calibrated probability of Home win
        prob_home = modelo_wrapper.predict_probability(X_pred)[0]
        prob_away = 1.0 - prob_home
        
        # Instantiate Kelly Portfolio with optimized, risk-averse settings
        pm = KellyPortfolioManager(fraction=0.15, max_bet_fraction=0.04, min_edge=0.015)
        
        # Calculate optimal bet sizes scaling down if entropy/uncertainty is high
        entropy = - (prob_home * np.log(prob_home + 1e-9) + prob_away * np.log(prob_away + 1e-9))
        uncertainty_scale = float(np.clip(1.0 - (entropy / 0.6931), 0.5, 1.0)) # penalize high uncertainty matches
        
        bet_size_home_pct = pm.calculate_bet_size(float(h), float(prob_home), uncertainty_scale)
        bet_size_away_pct = pm.calculate_bet_size(float(a), float(prob_away), uncertainty_scale)
        
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
        logger.error(f"❌ Error during runtime prediction or bankroll management logic: {e}")
        return {"error": str(e)}