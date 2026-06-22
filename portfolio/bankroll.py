import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import RobustScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from scipy.optimize import minimize
from typing import List, Dict, Union, Tuple, Optional
from config import APUESTA_MINIMA

class EloRatingSystem:
    """
    State-of-the-art dynamic Elo Rating System for tracking true team latent strength.
    """
    def __init__(self, k_factor: float = 25.0, default_rating: float = 1500.0):
        self.k_factor = k_factor
        self.default_rating = default_rating
        self.ratings = {}

    def get_rating(self, team_id: Union[str, int]) -> float:
        if team_id not in self.ratings:
            self.ratings[team_id] = self.default_rating
        return self.ratings[team_id]

    def update_ratings(self, home_id: Union[str, int], away_id: Union[str, int], home_score: float, away_score: float):
        r_home = self.get_rating(home_id)
        r_away = self.get_rating(away_id)
        
        e_home = 1.0 / (1.0 + 10.0 ** ((r_away - r_home) / 400.0))
        e_away = 1.0 - e_home
        
        if home_score > away_score:
            s_home, s_away = 1.0, 0.0
        elif home_score < away_score:
            s_home, s_away = 0.0, 1.0
        else:
            s_home, s_away = 0.5, 0.5
            
        self.ratings[home_id] = r_home + self.k_factor * (s_home - e_home)
        self.ratings[away_id] = r_away + self.k_factor * (s_away - e_away)


class FeatureEngineer:
    """
    High-alpha predictive feature engineering pipeline.
    Robust scaling, Elo rating propagation, and advanced market efficiency metrics.
    """
    def __init__(self):
        self.scaler = RobustScaler()
        self.elo_system = EloRatingSystem()
        self.feature_cols = []

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._sort_chronologically(df)
        df = self._compute_elo_features(df)
        df = self._create_time_decay_metrics(df)
        df = self._create_odds_discrepancy_features(df)
        df = self._create_rolling_performance(df)
        
        self.feature_cols = [c for c in df.columns if c not in ['target', 'date', 'event_id', 'home_team_id', 'away_team_id', 'home_score', 'away_score']]
        df[self.feature_cols] = self.scaler.fit_transform(df[self.feature_cols].fillna(0))
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._sort_chronologically(df)
        df = self._compute_elo_features(df)
        df = self._create_time_decay_metrics(df)
        df = self._create_odds_discrepancy_features(df)
        df = self._create_rolling_performance(df)
        
        df[self.feature_cols] = self.scaler.transform(df[self.feature_cols].fillna(0))
        return df

    def _sort_chronologically(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
        return df

    def _compute_elo_features(self, df: pd.DataFrame) -> pd.DataFrame:
        home_elos = []
        away_elos = []
        elo_diffs = []
        
        for _, row in df.iterrows():
            h_id = row.get('home_team_id', 'default_home')
            a_id = row.get('away_team_id', 'default_away')
            
            h_elo = self.elo_system.get_rating(h_id)
            a_elo = self.elo_system.get_rating(a_id)
            
            home_elos.append(h_elo)
            away_elos.append(a_elo)
            elo_diffs.append(h_elo - a_elo)
            
            h_score = row.get('home_score', None)
            a_score = row.get('away_score', None)
            if h_score is not None and a_score is not None and not (pd.isna(h_score) or pd.isna(a_score)):
                self.elo_system.update_ratings(h_id, a_id, float(h_score), float(a_score))
                
        df['home_elo'] = home_elos
        df['away_elo'] = away_elos
        df['elo_diff'] = elo_diffs
        return df

    def _create_time_decay_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'days_since_last_match' in df.columns:
            df['fatigue_index'] = np.exp(-df['days_since_last_match'] / 7.0)
            df['rest_advantage'] = df.get('home_days_since_last', 0.0) - df.get('away_days_since_last', 0.0)
        return df

    def _create_odds_discrepancy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'market_odds' in df.columns:
            df['implied_probability'] = 1.0 / df['market_odds']
            if 'opening_odds' in df.columns:
                df['odds_drift'] = df['market_odds'] - df['opening_odds']
                df['log_odds_drift'] = np.log(df['market_odds'] / (df['opening_odds'] + 1e-9) + 1e-9)
            if 'draw_odds' in df.columns and 'away_odds' in df.columns:
                df['market_overround'] = (1.0 / df['market_odds']) + (1.0 / df['draw_odds']) + (1.0 / df['away_odds']) - 1.0
        return df

    def _create_rolling_performance(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'home_score' in df.columns and 'home_team_id' in df.columns:
            df['home_rolling_avg'] = df.groupby('home_team_id')['home_score'].transform(lambda x: x.rolling(5, min_periods=1).mean())
            df['away_rolling_avg'] = df.groupby('away_team_id')['away_score'].transform(lambda x: x.rolling(5, min_periods=1).mean())
            df['rolling_diff'] = df['home_rolling_avg'] - df['away_rolling_avg']
        return df


class PredictiveEngine:
    """
    Ensemble of XGBoost and LightGBM models with progressive time-series split calibration.
    """
    def __init__(self):
        self.xgb_model = None
        self.lgb_model = None
        self.is_trained = False
        self.xgb_weight = 0.5
        self.lgb_weight = 0.5

    def train(self, X: pd.DataFrame, y: pd.Series):
        tscv = TimeSeriesSplit(n_splits=5)
        
        xgb_clf = xgb.XGBClassifier(
            n_estimators=1000,
            learning_rate=0.015,
            max_depth=6,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.75,
            gamma=0.1,
            scale_pos_weight=1,
            random_state=42,
            eval_metric='logloss'
        )
        
        lgb_clf = lgb.LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.015,
            max_depth=6,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.75,
            min_child_samples=15,
            random_state=42,
            verbosity=-1
        )
        
        self.xgb_model = CalibratedClassifierCV(estimator=xgb_clf, method='isotonic', cv=tscv)
        self.lgb_model = CalibratedClassifierCV(estimator=lgb_clf, method='isotonic', cv=tscv)
        
        self.xgb_model.fit(X, y)
        self.lgb_model.fit(X, y)
        
        self.is_trained = True

    def predict_probabilities(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Predictive Engine must be trained prior to inference.")
        
        xgb_probs = self.xgb_model.predict_proba(X)[:, 1]
        lgb_probs = self.lgb_model.predict_proba(X)[:, 1]
        
        return self.xgb_weight * xgb_probs + self.lgb_weight * lgb_probs


class AdvancedKellyPortfolioManager:
    """
    True Multi-Bet Kelly Optimization.
    Computes exact fractional allocations under joint exposure using convex optimization.
    """
    def __init__(self, fraction: float = 0.10, max_total_exposure: float = 0.40):
        self.fraction = fraction
        self.max_total_exposure = max_total_exposure

    def calculate_optimal_stakes(self, 
                                 probabilities: np.ndarray, 
                                 odds: np.ndarray, 
                                 bankroll: float, 
                                 model_confidence: float = 0.95) -> List[float]:
        if bankroll <= 0 or len(probabilities) == 0:
            return [0.0] * len(probabilities)

        adjusted_probs = []
        for p, d_odds in zip(probabilities, odds):
            p_adj = p * model_confidence + (1.0 - model_confidence) * (1.0 / d_odds)
            adjusted_probs.append(p_adj)
            
        adjusted_probs = np.array(adjusted_probs)
        net_odds = odds - 1.0
        n = len(probabilities)

        # Objective: Maximize expected logarithmic growth rate
        def objective(f):
            # Safe log calculation for non-betting scenario
            expected_log = 0.0
            for i in range(n):
                expected_log += adjusted_probs[i] * np.log(1.0 + f[i] * net_odds[i]) + (1.0 - adjusted_probs[i]) * np.log(1.0 - f[i])
            return -expected_log

        # Constraints: Allocations must be positive, sum of fractions must respect maximum exposure limits
        bounds = [(0.0, self.fraction) for _ in range(n)]
        constraints = {'type': 'ineq', 'fun': lambda f: self.max_total_exposure - np.sum(f)}
        
        initial_guess = [0.01] * n
        
        res = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if not res.success:
            # Fallback to decoupled fractional Kelly
            stakes = []
            for p, o in zip(adjusted_probs, odds):
                b = o - 1.0
                q = 1.0 - p
                f = (b * p - q) / b
                if f > 0:
                    allocated = min(f * self.fraction, self.fraction)
                    stakes.append(allocated * bankroll)
                else:
                    stakes.append(0.0)
            return stakes

        optimized_fractions = res.x
        stakes = []
        for f in optimized_fractions:
            stake_amount = bankroll * f
            if stake_amount >= APUESTA_MINIMA:
                stakes.append(round(stake_amount, 2))
            else:
                stakes.append(0.0)
                
        return stakes

    def calculate_simultaneous_bets(self, 
                                    opportunities: List[Dict[str, float]], 
                                    bankroll: float) -> List[Dict[str, Union[float, str]]]:
        if not opportunities or bankroll <= 0:
            return []

        probs = np.array([op['prob'] for op in opportunities])
        odds = np.array([op['odds'] for op in opportunities])
        
        stakes = self.calculate_optimal_stakes(probs, odds, bankroll)
        
        for i, op in enumerate(opportunities):
            op['suggested_stake'] = stakes[i]
            
        return opportunities


class ProductionMLBettingSystem:
    """
    Unified end-to-end professional sports prediction & dynamic asset allocation pipeline.
    """
    def __init__(self, kelly_fraction: float = 0.10, max_total_exposure: float = 0.40):
        self.feature_engineer = FeatureEngineer()
        self.predictor = PredictiveEngine()
        self.portfolio_manager = AdvancedKellyPortfolioManager(fraction=kelly_fraction, max_total_exposure=max_total_exposure)
        
    def fit(self, historical_data: pd.DataFrame, targets: pd.Series):
        processed_features = self.feature_engineer.fit_transform(historical_data)
        # Drop non-feature columns that feature engineer used but aren't models inputs
        train_cols = [c for c in processed_features.columns if c in self.feature_engineer.feature_cols]
        self.predictor.train(processed_features[train_cols], targets)
        
    def generate_bets(self, current_opportunities: pd.DataFrame, bankroll: float) -> List[Dict[str, Union[float, str]]]:
        processed_features = self.feature_engineer.transform(current_opportunities)
        train_cols = [c for c in processed_features.columns if c in self.feature_engineer.feature_cols]
        predicted_probs = self.predictor.predict_probabilities(processed_features[train_cols])
        
        opps_list = []
        for idx, row in current_opportunities.iterrows():
            opps_list.append({
                'id': row.get('event_id', str(idx)),
                'odds': float(row['market_odds']),
                'prob': float(predicted_probs[idx])
            })
            
        return self.portfolio_manager.calculate_simultaneous_bets(opps_list, bankroll)