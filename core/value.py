def evaluar_apuesta(prob: float, odds: float) -> tuple[float, float]:
    """
    Calcula de manera optimizada:
    - EV (Expected Value)
    - Criterio de Kelly (simplificado matemáticamente a EV / net_odds)
    """
    if odds <= 1.0 or not (0.0 <= prob <= 1.0):
        return -1.0, -1.0

    ev = (prob * odds) - 1.0
    kelly = ev / (odds - 1.0)

    return ev, kelly