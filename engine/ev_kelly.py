def calcular_ev(probabilidad, cuota):
    return (probabilidad * cuota) - 1


def calcular_kelly(probabilidad, cuota):
    b = cuota - 1
    p = probabilidad
    q = 1 - p

    kelly = (b * p - q) / b

    return max(0, kelly)
