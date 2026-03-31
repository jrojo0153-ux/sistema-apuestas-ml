# portfolio/bankroll.py

from config import APUESTA_MINIMA

def calcular_apuesta(kelly, bankroll):
    stake = bankroll * kelly

    if stake < APUESTA_MINIMA:
        return 0

    return round(stake, 2)
