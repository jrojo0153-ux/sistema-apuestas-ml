import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from scipy.optimize import minimize
from typing import List, Dict, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.numeric_cols = []

    def fit_transform(self, df: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
        df = df.copy()
        df = self._generate_features(df)
        
        exclude_cols = ['odds', 'event_id', 'date']
        if target_col:
            exclude_cols.append(target_col)
            
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.difference(exclude_cols).tolist()
        
        if self.numeric_cols:
            df[self.numeric_cols] = self.scaler.fit_transform(df[self.numeric_cols].fillna(0.0))
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._generate_features(df)
        if self.numeric_cols:
            df[self.numeric_cols] = self.scaler.transform(df[self.numeric_cols].fillna(0.0))
        return df

    def _generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['implied_prob'] = 1.0 / df['odds']
        
        if 'historical_prob' in df.columns:
            df['prob_discrepancy'] = df['implied_prob'] - df['historical_prob']
        
        if 'team_score_avg' in df.columns and 'opponent_score_avg' in df.columns:
            df['score_diff_avg'] = df['team_score_avg'] - df['opponent_score_avg']
            
        if 'volume' in df.columns:
            df['market_liquidity_score'] = np.log1p(df['volume'])
            
        if 'odds_trend' in df.columns:
            df['odds_momentum'] = df['odds_trend'] * df['implied_prob']
            
        if 'rolling_wins_team' in df.columns:
            df['form_index'] = df['rolling_wins_team'] - df.get('rolling_wins_opp', 0.0)
            
        return df


class BettingPredictor:
    def __init__(self):
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=500,
            learning_rate=0.015,
            max_depth=6,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            eval_metric='logloss',
            tree_method='hist'
        )
        self.lgb_model = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.015,
            max_depth=6,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            verbosity=-1
        )
        self.calibrated_ensemble = []

    def train(self, X: pd.DataFrame, y: pd.Series):
        tscv = TimeSeriesSplit(n_splits=5)
        
        cal_xgb = CalibratedClassifierCV(estimator=self.xgb_model, method='isotonic', cv=tscv)
        cal_lgb = CalibratedClassifierCV(estimator=self.lgb_model, method='isotonic', cv=tscv)
        
        cal_xgb.fit(X, y)
        cal_lgb.fit(X, y)
        
        self.calibrated_ensemble = [cal_xgb, cal_lgb]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.calibrated_ensemble:
            raise ValueError("Predictor models have not been trained yet.")
        
        predictions = [model.predict_proba(X)[:, 1] for model in self.calibrated_ensemble]
        return np.mean(predictions, axis=0)


class KellyOptimizer:
    @staticmethod
    def calculate_ev(prob: float, odds: float) -> float:
        return (prob * odds) - 1.0

    @staticmethod
    def single_kelly(prob: float, odds: float, fraction: float = 0.20) -> float:
        if odds <= 1.0 or prob <= 0.0:
            return 0.0
        ev = KellyOptimizer.calculate_ev(prob, odds)
        if ev <= 0.0:
            return 0.0
        raw_kelly = ev / (odds - 1.0)
        return float(np.clip(raw_kelly * fraction, 0.0, 1.0))

    @staticmethod
    def optimize_portfolio(probs: np.ndarray, odds: np.ndarray, fraction: float = 0.20, max_total_risk: float = 0.40) -> np.ndarray:
        n = len(probs)
        if n == 0:
            return np.array([])
            
        def objective(weights):
            log_growth = 0.0
            for i in range(n):
                win_outcome = 1.0 + weights[i] * (odds[i] - 1.0)
                lose_outcome = 1.0 - weights[i]
                win_outcome = max(1e-12, win_outcome)
                lose_outcome = max(1e-12, lose_outcome)
                log_growth += probs[i] * np.log(win_outcome) + (1.0 - probs[i]) * np.log(lose_outcome)
            return -log_growth

        bounds = []
        for i in range(n):
            max_indiv_kelly = KellyOptimizer.single_kelly(probs[i], odds[i], fraction)
            bounds.append((0.0, max_indiv_kelly))

        constraints = ({'type': 'ineq', 'fun': lambda w: max_total_risk - np.sum(w)})
        
        initial_weights = np.zeros(n)
        res = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if res.success:
            return res.x
        return np.array([KellyOptimizer.single_kelly(p, o, fraction) for p, o in zip(probs, odds)])


class MLBettingSystem:
    def __init__(self, kelly_fraction: float = 0.20, max_portfolio_risk: float = 0.40):
        self.feature_engineer = FeatureEngineer()
        self.predictor = BettingPredictor()
        self.kelly_fraction = kelly_fraction
        self.max_portfolio_risk = max_portfolio_risk
        self.is_trained = False

    def train_system(self, historical_df: pd.DataFrame, target_col: str):
        X = historical_df.copy()
        y = X.pop(target_col)
        
        X_processed = self.feature_engineer.fit_transform(X, target_col)
        self.predictor.train(X_processed, y)
        self.is_trained = True

    def process_and_allocate(self, current_events_df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_trained:
            raise ValueError("The system has not been trained. Run train_system first.")
            
        df = current_events_df.copy()
        X_processed = self.feature_engineer.transform(df)
        
        df['predicted_prob'] = self.predictor.predict_proba(X_processed)
        df['expected_value'] = (df['predicted_prob'] * df['odds']) - 1.0
        
        df['optimal_stake_fraction'] = 0.0
        
        positive_ev_mask = df['expected_value'] > 0.0
        if positive_ev_mask.any():
            pos_ev_df = df[positive_ev_mask]
            indices = pos_ev_df.index
            
            probs = pos_ev_df['predicted_prob'].values
            odds = pos_ev_df['odds'].values
            
            optimized_allocations = KellyOptimizer.optimize_portfolio(
                probs=probs,
                odds=odds,
                fraction=self.kelly_fraction,
                max_total_risk=self.max_portfolio_risk
            )
            
            df.loc[indices, 'optimal_stake_fraction'] = optimized_allocations
            
        return df