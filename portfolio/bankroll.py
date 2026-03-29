import math

class BankrollManager:
    def __init__(self, bankroll_inicial=1000, riesgo_max=0.05):
        """
        bankroll_inicial: dinero total inicial
        riesgo_max: % máximo por apuesta (protección)
        """
        self.bankroll = bankroll_inicial
        self.bankroll_inicial = bankroll_inicial
        self.riesgo_max = riesgo_max
        self.historial = []

    
  def calcular_apuesta(bankroll, kelly, riesgo_max=0.05):
    if kelly <= 0:
        return 0

    kelly = min(kelly, riesgo_max)

    apuesta = bankroll * kelly

    return round(apuesta, 2)
