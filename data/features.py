# data/features.py

import numpy as np

def construir_features(partido):
    home_odds = float(partido["home_odds"])
    away_odds = float(partido["away_odds"])

    prob_home = 1 / home_odds
    prob_away = 1 / away_odds

    diff_prob = prob_home - prob_away

    return np.array([
        home_odds,
        away_odds,
        prob_home,
        prob_away,
        diff_prob
    ])
