def calculate_edge(prob, odds):
    implied = 1 / odds
    return prob - implied
