# core/value.py

def evaluar_apuesta(prob, odds):
    """
    Calcula:
    - EV (Expected Value)
    - Kelly Criterion
    """

    if odds <= 1:
        return -1, -1

    ev = (prob * odds) - 1

    kelly = ((prob * (odds - 1)) - (1 - prob)) / (odds - 1)

    return ev, kelly
