def calcular_apuesta(bankroll, kelly, riesgo_max=0.05):
    if kelly <= 0:
        return 0

    apuesta = bankroll * kelly

    # limitar riesgo máximo
    max_apuesta = bankroll * riesgo_max

    return round(min(apuesta, max_apuesta), 2)
