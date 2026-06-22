import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from typing import List, Dict, Tuple, Any, Optional

class FeatureEngineer:
    @staticmethod
    def extract_features(partido: Dict[str, Any]) -> np.ndarray:
        try:
            home_odds = float(partido.get("home_odds") or 0.0)
            away_odds = float(partido.get("away_odds") or 0.0)
            draw_odds = float(partido.get("draw_odds") or 0.0)
        except (ValueError, TypeError):
            home_odds, away_odds, draw_odds = 0.0, 0.0, 0.0

        p_home = 1.0 / home_odds if home_odds > 1.0 else 0.0
        p_away = 1.0 / away_odds if away_odds > 1.0 else 0.0
        p_draw = 1.0 / draw_odds if draw_odds > 1.0 else 0.0

        sum_implied = p_home + p_away + p_draw
        overround = sum_implied - 1.0 if sum_implied > 1.0 else 0.0

        if sum_implied > 0:
            p_home_fair = p_home / sum_implied
            p_away_fair = p_away / sum_implied
            p_draw_fair = p_draw / sum_implied if p_draw > 0 else 0.0
        else:
            p_home_fair, p_away_fair, p_draw_fair = 0.0, 0.0, 0.0

        diff_home_away = p_home - p_away
        ratio_home_away = home_odds / away_odds if away_odds > 0 else 0.0
        log_ratio = np.log(ratio_home_away) if ratio_home_away > 0 else 0.0

        probs = np.array([p_home_fair, p_away_fair, p_draw_fair])
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log(probs)) if len(probs) > 0 else 0.0

        return np.array([
            home_odds,
            away_odds,
            draw_odds,
            p_home,
            p_away,
            p_draw,
            p_home_fair,
            p_away_fair,
            p_draw_fair,
            overround,
            diff_home_away,
            ratio_home_away,
            log_ratio,
            entropy
        ], dtype=np.float64)

    @classmethod
    def transform_dataset(cls, partidos: List[Dict[str, Any]]) -> pd.DataFrame:
        features_list = [cls.extract_features(p) for p in partidos]
        columns = [
            "home_odds", "away_odds", "draw_odds",
            "p_home", "p_away", "p_draw",
            "p_home_fair", "p_away_fair", "p_draw_fair",
            "overround", "diff_home_away", "ratio_home_away",
            "log_ratio", "entropy"
        ]
        return pd.DataFrame(features_list, columns=columns)

class SportsPredictor:
    def __init__(self):
        self.model = None
        self.is_trained = False

    def train(self, X: pd.DataFrame, y: np.ndarray):
        base_model = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss"
        )
        self.model = CalibratedClassifierCV(estimator=base_model, method="sigmoid", cv=5)
        self.model.fit(X, y)
        self.is_trained = True

    def predict_probabilities(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained or self.model is None:
            raise ValueError("Modelo no entrenado. Llame a 'train' primero.")
        return self.model.predict_proba(X)

class KellyCriterionManager:
    def __init__(self, fraction: float = 0.10, max_bet_fraction: float = 0.05, min_edge: float = 0.02):
        self.fraction = fraction
        self.max_bet_fraction = max_bet_fraction
        self.min_edge = min_edge

    def calculate_stake(self, bankroll: float, predicted_prob: float, odds: float) -> Tuple[float, float]:
        if odds <= 1.0 or predicted_prob <= 0.0:
            return 0.0, 0.0

        b = odds - 1.0
        q = 1.0 - predicted_prob
        edge = (predicted_prob * b) - q

        if edge < self.min_edge:
            return 0.0, edge

        kelly_f = edge / b
        safe_kelly_f = kelly_f * self.fraction
        final_fraction = min(safe_kelly_f, self.max_bet_fraction)
        stake = max(0.0, final_fraction * bankroll)

        return stake, edge

class SportsBettingEngine:
    def __init__(self, kelly_fraction: float = 0.10, max_bet_fraction: float = 0.05, min_edge: float = 0.02):
        self.feature_engineer = FeatureEngineer()
        self.predictor = SportsPredictor()
        self.kelly_manager = KellyCriterionManager(kelly_fraction, max_bet_fraction, min_edge)

    def train_system(self, historical_data: List[Dict[str, Any]], labels: List[int]):
        X = self.feature_engineer.transform_dataset(historical_data)
        y = np.array(labels)
        self.predictor.train(X, y)

    def evaluate_bet(self, partido: Dict[str, Any], bankroll: float) -> Dict[str, Any]:
        features = self.feature_engineer.extract_features(partido).reshape(1, -1)
        columns = [
            "home_odds", "away_odds", "draw_odds",
            "p_home", "p_away", "p_draw",
            "p_home_fair", "p_away_fair", "p_draw_fair",
            "overround", "diff_home_away", "ratio_home_away",
            "log_ratio", "entropy"
        ]
        df_features = pd.DataFrame(features, columns=columns)
        probs = self.predictor.predict_probabilities(df_features)[0]
        
        home_odds = float(partido.get("home_odds") or 0.0)
        away_odds = float(partido.get("away_odds") or 0.0)
        draw_odds = float(partido.get("draw_odds") or 0.0)

        outcomes = [
            {"name": "HOME", "prob": probs[0], "odds": home_odds},
            {"name": "AWAY", "prob": probs[1] if len(probs) > 1 else 0.0, "odds": away_odds}
        ]
        if len(probs) > 2 and draw_odds > 0:
            outcomes.append({"name": "DRAW", "prob": probs[2], "odds": draw_odds})

        best_bet = None
        max_stake = 0.0

        for outcome in outcomes:
            if outcome["odds"] > 1.0:
                stake, edge = self.kelly_manager.calculate_stake(bankroll, outcome["prob"], outcome["odds"])
                if stake > max_stake:
                    max_stake = stake
                    best_bet = {
                        "selection": outcome["name"],
                        "odds": outcome["odds"],
                        "probability": outcome["prob"],
                        "edge": edge,
                        "stake": stake
                    }

        return best_bet or {"selection": "NO_BET", "odds": 0.0, "probability": 0.0, "edge": 0.0, "stake": 0.0}