import numpy as np

def calcular_ev_y_kelly(probabilidades, cuotas):
    resultados = []

    for prob, cuota in zip(probabilidades, cuotas):

        if prob <= 0 or cuota <= 1:
            continue

        ev = (prob * cuota) - 1

        kelly = ((prob * (cuota - 1)) - (1 - prob)) / (cuota - 1)

        kelly = max(0, kelly)

        resultados.append({
            "ev": round(ev, 4),
            "kelly": round(kelly, 4)
        })

    return resultados
