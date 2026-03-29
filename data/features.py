import numpy as np

def construir_features(partido):
    try:
        home_odds = partido["home_odds"]
        away_odds = partido["away_odds"]

        prob_home = 1 / home_odds
        prob_away = 1 / away_odds

        diff_prob = prob_home - prob_away

        return np.array([
            prob_home,
            prob_away,
            diff_prob
        ])

    except:
        return np.array([0.5, 0.5, 0.0])
