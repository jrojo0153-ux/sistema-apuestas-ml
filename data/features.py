import numpy as np

def construir_features(partido: dict) -> np.ndarray:
    try:
        home_odds = float(partido.get("home_odds") or 0)
        away_odds = float(partido.get("away_odds") or 0)
    except (ValueError, TypeError):
        home_odds, away_odds = 0.0, 0.0

    prob_home = 1.0 / home_odds if home_odds > 0 else 0.0
    prob_away = 1.0 / away_odds if away_odds > 0 else 0.0
    diff_prob = prob_home - prob_away

    return np.array([
        home_odds,
        away_odds,
        prob_home,
        prob_away,
        diff_prob
    ], dtype=np.float64)