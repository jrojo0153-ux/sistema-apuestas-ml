# engine/ev_kelly.py

import numpy as np


def calcular_ev(prob, cuota):
    """
    Calcula el Valor Esperado (EV)
    EV = (probabilidad * cuota) - 1
    """
    try:
        if prob is None or cuota is None:
            return 0

        prob = float(prob)
        cuota = float(cuota)

        if prob <= 0 or cuota <= 0:
            return 0

        return (prob * cuota) - 1

    except Exception:
        return 0


def calcular_kelly(prob, cuota):
    """
    Fórmula de Kelly:
    kelly = ((prob * cuota) - 1) / (cuota - 1)
    """
    try:
        if prob is None or cuota is None:
            return 0

        prob = float(prob)
        cuota = float(cuota)

        if prob <= 0 or cuota <= 1:
            return 0

        kelly = ((prob * cuota) - 1) / (cuota - 1)

        # Limitar valores extremos
        if kelly < 0:
            return 0
        if kelly > 1:
            return 1

        return kelly

    except Exception:
        return 0


def evaluar_partido(probs, cuotas):
    """
    Evalúa un partido completo

    probs = [prob_local, prob_empate, prob_visitante]
    cuotas = [cuota_local, cuota_empate, cuota_visitante]
    """

    try:
        # VALIDACIÓN FUERTE
        if probs is None or cuotas is None:
            return None

        if len(probs) != 3 or len(cuotas) != 3:
            return None

        resultados = []

        etiquetas = ["local", "empate", "visitante"]

        for i in range(3):
            prob = probs[i]
            cuota = cuotas[i]

            ev = calcular_ev(prob, cuota)
            kelly = calcular_kelly(prob, cuota)

            resultados.append({
                "tipo": etiquetas[i],
                "probabilidad": round(prob, 4),
                "cuota": cuota,
                "ev": round(ev, 4),
                "kelly": round(kelly, 4)
            })

        # Filtrar solo apuestas con valor positivo
        apuestas_valor = [r for r in resultados if r["ev"] > 0]

        if len(apuestas_valor) == 0:
            return None

        # Ordenar por EV
        mejor = sorted(apuestas_valor, key=lambda x: x["ev"], reverse=True)[0]

        return mejor

    except Exception as e:
        print(f"❌ Error en EV/Kelly: {e}")
        return None
