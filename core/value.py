from ml.model import get_winrate

def calculate_edge(odds):
    prob_model = get_winrate()  # 🔥 aprendizaje real
    implied = 1 / odds

    edge = prob_model - implied
    return edge
