BANKROLL_INICIAL = 1000

def calcular_apuesta(kelly, bankroll):
    stake = bankroll * kelly
    return round(max(10, stake), 2)
