"""
portfolio/bankroll.py — Gestión de bankroll
"""

class GestorBankroll:
    """
    Controla el dinero y apuestas realizadas
    """

    def __init__(self, bankroll_inicial=1000):
        self.bankroll = bankroll_inicial
        self.historial = []
        self.apuestas = 0

    def calcular_apuesta(self, señal):
        """
        Calcula cuánto apostar basado en Kelly
        """
        kelly = señal.get("kelly", 0)

        # usamos fracción de Kelly (más conservador)
        apuesta = self.bankroll * (kelly * 0.25)

        # mínimo
        if apuesta < 1:
            apuesta = 1

        return apuesta

    def registrar_apuesta(self, señal):
        """
        Registra una apuesta (simulada)
        """
        apuesta = self.calcular_apuesta(señal)

        # simulación simple (no sabemos resultado aquí)
        self.bankroll -= apuesta

        self.historial.append(self.bankroll)
        self.apuestas += 1

    def resumen(self):
        """
        Resumen del estado del bankroll
        """
        return {
            "bankroll_actual": self.bankroll,
            "apuestas_realizadas": self.apuestas
        }
