import numpy as np

def calcular_ev_y_kelly(probabilidades, cuotas):
    probabilidades = np.array(probabilidades)
    cuotas = np.array(cuotas)

    if probabilidades.size == 0 or cuotas.size == 0:
        return []

    resultados = []

    for p, cuota in zip(probabilidades, cuotas):
        if cuota <= 1:
            continue

        b = cuota - 1
        q = 1 - p

        ev = (p * cuota) - 1
        kelly = (p * (cuota - 1) - q) / b if b != 0 else 0

        resultados.append({
            "ev": round(ev, 4),
            "kelly": max(0, round(kelly, 4))
        })

    return resultados
