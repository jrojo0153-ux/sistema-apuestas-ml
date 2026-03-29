import numpy as np


def calcular_ev(prob, cuota):
    """
    Calcula el valor esperado (EV)
    """
    return (prob * cuota) - 1


def calcular_kelly(prob, cuota):
    """
    Fórmula de Kelly
    """
    b = cuota - 1
    return (prob * (b + 1) - 1) / b if b > 0 else 0


def calcular_ev_y_kelly(probabilidades, cuotas):
    """
    Calcula EV y Kelly para listas completas
    """

    # 🔒 VALIDACIÓN CORRECTA (FIX ERROR)
    if probabilidades is None or cuotas is None:
        return []

    if len(probabilidades) == 0 or len(cuotas) == 0:
        return []

    if len(probabilidades) != len(cuotas):
        raise ValueError("Probabilidades y cuotas no tienen el mismo tamaño")

    resultados = []

    for prob, cuota in zip(probabilidades, cuotas):

        # Validaciones individuales
        if cuota <= 1:
            continue

        ev = calcular_ev(prob, cuota)
        kelly = calcular_kelly(prob, cuota)

        resultados.append({
            "probabilidad": prob,
            "cuota": cuota,
            "ev": round(ev, 4),
            "kelly": round(kelly, 4)
        })

    return resultados
