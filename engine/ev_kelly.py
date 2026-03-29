import numpy as np

def calcular_ev_y_kelly(probabilidades, cuotas):
    resultados = []

    if len(probabilidades) == 0 or len(cuotas) == 0:
        return []

    for prob, cuota in zip(probabilidades, cuotas):

        p = float(prob)
        q = 1 - p
        b = float(cuota) - 1

        ev = (p * cuota) - 1

        if b == 0:
            kelly = 0
        else:
            kelly = (b * p - q) / b

        kelly = max(0, kelly)

        resultados.append({
            "ev": round(ev, 4),
            "kelly": round(kelly, 4)
        })

    return resultados
