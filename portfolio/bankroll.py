from config import APUESTA_MINIMA

def calcular_apuesta(kelly: float, bankroll: float) -> float:
    if kelly <= 0 or bankroll <= 0:
        return 0.0
    
    stake = bankroll * kelly
    return round(stake, 2) if stake >= APUESTA_MINIMA else 0.0