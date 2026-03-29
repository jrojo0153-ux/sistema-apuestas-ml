bankroll = 1000  # bankroll inicial

# límites de seguridad
MAX_KELLY = 0.25      # nunca apostar más del 25%
MIN_KELLY = 0.01      # mínimo para considerar apuesta
MAX_APUESTA = 0.10    # máximo 10% del bankroll por apuesta


def obtener_bankroll():
    return bankroll


def actualizar_bankroll(resultado):
    """
    resultado: float (ganancia o pérdida)
    """
    global bankroll
    bankroll += resultado
    bankroll = max(bankroll, 0)  # nunca negativo
    return bankroll


def calcular_apuesta(bankroll, kelly):
    if kelly <= 0:
        return 0

    # 🔥 Kelly conservador (50%)
    stake = bankroll * (kelly * 0.5)

    return round(stake, 2)

    # Limitar Kelly (muy importante)
    kelly_ajustado = min(kelly, MAX_KELLY)

    # Calcular apuesta base
    apuesta = bankroll * kelly_ajustado

    # Limitar por porcentaje máximo
    apuesta_max = bankroll * MAX_APUESTA
    apuesta_final = min(apuesta, apuesta_max)

    return round(apuesta_final, 2)


def registrar_resultado(apuesta, cuota, gano):
    """
    Actualiza bankroll según resultado real
    """
    if apuesta <= 0:
        return bankroll

    if gano:
        ganancia = apuesta * (cuota - 1)
    else:
        ganancia = -apuesta

    return actualizar_bankroll(ganancia)
