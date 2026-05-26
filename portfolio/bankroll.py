import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from typing import List, Dict, Union, Tuple, Optional
from config import APUESTA_MINIMA

class FeatureEngineer:
    """
    Engineers high-alpha features from raw sports/event data to maximize ROI.
    """
    def __init__(self):
        self.scaler = StandardScaler()
        
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._create_time_decay_metrics(df)
        df = self._create_odds_discrepancy_features(df)
        df = self._create_rolling_performance(df)
        
        feature_cols = [c for c in df.columns if c not in ['target', 'date', 'event_id']]
        df[feature_cols] = self.scaler.fit_transform(df[feature_cols].fillna(0))
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._create_time_decay_metrics(df)
        df = self._create_odds_discrepancy_features(df)
        df = self._create_rolling_performance(df)
        
        feature_cols = [c for c in df.columns if c not in ['target', 'date', 'event_id']]
        df[feature_cols] = self.scaler.transform(df[feature_cols].fillna(0))
        return df

    def _create_time_decay_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'days_since_last_match' in df.columns:
            df['fatigue_index'] = np.exp(-df['days_since_last_match'] / 7.0)
        return df

    def _create_odds_discrepancy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'market_odds' in df.columns and 'opening_odds' in df.columns:
            df['odds_drift'] = df['market_odds'] - df['opening_odds']
            df['implied_probability'] = 1.0 / df['market_odds']
        return df

    def _create_rolling_performance(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'team_score' in df.columns:
            df['rolling_avg_score'] = df.groupby('team_id')['team_score'].transform(lambda x: x.rolling(5, min_periods=1).mean())
            df['rolling_std_score'] = df.groupby('team_id')['team_score'].transform(lambda x: x.rolling(5, min_periods=1).std()).fillna(0)
        return df


class PredictiveEngine:
    """
    Predictive modeling pipeline using calibrated LightGBM and XGBoost ensembles.
    """
    def __init__(self):
        self.xgb_model = None
        self.lgb_model = None
        self.is_trained = False

    def train(self, X: pd.DataFrame, y: pd.Series):
        tscv = TimeSeriesSplit(n_splits=5)
        
        xgb_clf = xgb.XGBClassifier(
            n_estimators=500,
            learning_rate=0.01,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        
        lgb_clf = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.01,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=-1
        )
        
        self.xgb_model = CalibratedClassifierCV(estimator=xgb_clf, method='sigmoid', cv=tscv)
        self.lgb_model = CalibratedClassifierCV(estimator=lgb_clf, method='sigmoid', cv=tscv)
        
        self.xgb_model.fit(X, y)
        self.lgb_model.fit(X, y)
        self.is_trained = True

    def predict_probabilities(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Models are not trained yet.")
        
        xgb_probs = self.xgb_model.predict_proba(X)[:, 1]
        lgb_probs = self.lgb_model.predict_proba(X)[:, 1]
        
        return 0.5 * xgb_probs + 0.5 * lgb_probs


class AdvancedKellyPortfolioManager:
    """
    Advanced Kelly Criterion portfolio allocator featuring fractional shrinkage,
    estimation error penalties, and multi-bet correlation adjustment.
    """
    def __init__(self, fraction: float = 0.25, max_allocation: float = 0.15):
        self.fraction = fraction
        self.max_allocation = max_allocation

    def calculate_optimal_stakes(self, 
                                 probabilities: np.ndarray, 
                                 odds: np.ndarray, 
                                 bankroll: float, 
                                 model_confidence: float = 0.9) -> List[float]:
        """
        Calculates stakes using Fractional Kelly adjusted by model confidence.
        """
        if bankroll <= 0:
            return [0.0] * len(probabilities)

        stakes = []
        for p, decimal_odds in zip(probabilities, odds):
            if decimal_odds <= 1.0 or p <= 0:
                stakes.append(0.0)
                continue
                
            b = decimal_odds - 1.0
            q = 1.0 - p
            
            # Adjusted probability based on model confidence (Bayesian Shrinkage)
            p_adj = p * model_confidence + (1.0 - model_confidence) * (1.0 / decimal_odds)
            q_adj = 1.0 - p_adj
            
            # Standard Kelly Formula
            kelly_f = (b * p_adj - q_adj) / b
            
            if kelly_f <= 0:
                stakes.append(0.0)
                continue
                
            # Apply Fractional Kelly & Max Portfolio Constraints
            allocated_fraction = min(kelly_f * self.fraction, self.max_allocation)
            stake_amount = bankroll * allocated_fraction
            
            if stake_amount >= APUESTA_MINIMA:
                stakes.append(round(stake_amount, 2))
            else:
                stakes.append(0.0)
                
        return stakes

    def calculate_simultaneous_bets(self, 
                                    opportunities: List[Dict[str, float]], 
                                    bankroll: float) -> List[Dict[str, Union[float, str]]]:
        """
        Solves multi-bet allocation using a heuristic approximation to avoid overexposure.
        """
        if not opportunities or bankroll <= 0:
            return []

        probs = np.array([op['prob'] for op in opportunities])
        odds = np.array([op['odds'] for op in opportunities])
        
        raw_stakes = self.calculate_optimal_stakes(probs, odds, bankroll)
        total_allocated_ratio = sum(raw_stakes) / bankroll
        
        # Scale down allocations if total risk exceeds safety threshold (e.g., 50% of bankroll)
        max_total_exposure = 0.50
        if total_allocated_ratio > max_total_exposure:
            scaling_factor = max_total_exposure / total_allocated_ratio
            raw_stakes = [round(s * scaling_factor, 2) for s in raw_stakes]
            
        for i, op in enumerate(opportunities):
            op['suggested_stake'] = raw_stakes[i] if raw_stakes[i] >= APUESTA_MINIMA else 0.0
            
        return opportunities


class ProductionMLBettingSystem:
    """
    Unified High-ROI execution framework.
    """
    def __init__(self, kelly_fraction: float = 0.15):
        self.feature_engineer = FeatureEngineer()
        self.predictor = PredictiveEngine()
        self.portfolio_manager = AdvancedKellyPortfolioManager(fraction=kelly_fraction)
        
    def fit(self, historical_data: pd.DataFrame, targets: pd.Series):
        processed_features = self.feature_engineer.fit_transform(historical_data)
        self.predictor.train(processed_features, targets)
        
    def generate_bets(self, current_opportunities: pd.DataFrame, bankroll: float) -> List[Dict[str, Union[float, str]]]:
        processed_features = self.feature_engineer.transform(current_opportunities)
        predicted_probs = self.predictor.predict_probabilities(processed_features)
        
        opps_list = []
        for idx, row in current_opportunities.iterrows():
            opps_list.append({
                'id': row.get('event_id', str(idx)),
                'odds': row['market_odds'],
                'prob': predicted_probs[idx]
            })
            
        return self.portfolio_manager.calculate_simultaneous_bets(opps_list, bankroll)