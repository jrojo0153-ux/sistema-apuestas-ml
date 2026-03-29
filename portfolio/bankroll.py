def calcular_apuesta(bankroll: float, kelly: float, riesgo_max: float = 0.05) -> float:
    """
    Calcula el tamaño de apuesta usando criterio de Kelly con control de riesgo.

    Args:
        bankroll (float): dinero total disponible
        kelly (float): fracción de Kelly (0 a 1)
        riesgo_max (float): límite máximo por apuesta (default 5%)

    Returns:
        float: monto a apostar
    """

    # Validaciones
    if bankroll <= 0:
        return 0

    if kelly <= 0:
        return 0

    # Limitar riesgo (anti-ruina)
    kelly_ajustado = min(kelly, riesgo_max)

    apuesta = bankroll * kelly_ajustado

    return round(apuesta, 2)


def actualizar_bankroll(bankroll: float, apuesta: float, gano: bool, cuota: float) -> float:
    """
    Actualiza bankroll después de apuesta

    Args:
        bankroll (float)
        apuesta (float)
        gano (bool)
        cuota (float)

    Returns:
        float
    """

    if apuesta <= 0:
        return bankroll

    if gano:
        ganancia = apuesta * (cuota - 1)
        bankroll += ganancia
    else:
        bankroll -= apuesta

    return round(bankroll, 2)
