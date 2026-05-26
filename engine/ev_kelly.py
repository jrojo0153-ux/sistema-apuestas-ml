def calcular_ev(probabilidad: float, cuota: float) -> float:
    return (probabilidad * cuota) - 1


def calcular_kelly(probabilidad: float, cuota: float) -> float:
    if cuota <= 1 or probabilidad <= 0:
        return 0.0
    
    ev = calcular_ev(probabilidad, cuota)
    return max(0.0, ev / (cuota - 1))