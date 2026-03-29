def calcular_ev(prob: float, cuota: float) -> float:
    """
    Calcula valor esperado (EV)
    """
    return (prob * cuota) - 1


def calcular_kelly(prob: float, cuota: float) -> float:
    """
    Fórmula de Kelly
    """
    if cuota <= 1:
        return 0

    kelly = ((prob * (cuota - 1)) - (1 - prob)) / (cuota - 1)

    return max(0, kelly)


def calcular_ev_y_kelly(probabilidades: list, cuotas: list) -> list:
    """
    Calcula EV y Kelly para múltiples eventos
    """

    resultados = []

    if len(probabilidades) != len(cuotas):
        print("❌ Error: listas de diferente tamaño")
        return []

    for prob, cuota in zip(probabilidades, cuotas):

        # Validaciones seguras (evita error numpy)
        if prob is None or cuota is None:
            continue

        if prob <= 0 or prob >= 1:
            continue

        if cuota <= 1:
            continue

        ev = calcular_ev(prob, cuota)
        kelly = calcular_kelly(prob, cuota)

        resultados.append({
            "ev": round(ev, 4),
            "kelly": round(kelly, 4)
        })

    return resultados
