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

    def calcular_apuesta(self, kelly):
        """
        Calcula apuesta usando Kelly fraccionado + límite de riesgo
        """

        if kelly <= 0:
            return 0

        # Kelly conservador (25%)
        kelly_ajustado = kelly * 0.25

        # apuesta base
        apuesta = self.bankroll * kelly_ajustado

        # límite máximo por riesgo
        max_apuesta = self.bankroll * self.riesgo_max

        apuesta_final = min(apuesta, max_apuesta)

        return round(apuesta_final, 2)

    def actualizar_bankroll(self, apuesta, cuota, gano):
        """
        Actualiza bankroll después de cada apuesta
        """

        if apuesta <= 0:
            return self.bankroll

        if gano:
            ganancia = apuesta * (cuota - 1)
            self.bankroll += ganancia
        else:
            self.bankroll -= apuesta

        self.historial.append(self.bankroll)

        return round(self.bankroll, 2)

    def get_drawdown(self):
        """
        Calcula drawdown máximo
        """
        peak = self.bankroll_inicial
        max_dd = 0

        for valor in self.historial:
            if valor > peak:
                peak = valor

            dd = (peak - valor) / peak

            if dd > max_dd:
                max_dd = dd

        return round(max_dd, 4)

    def resumen(self):
        return {
            "bankroll_actual": round(self.bankroll, 2),
            "ganancia_total": round(self.bankroll - self.bankroll_inicial, 2),
            "drawdown": self.get_drawdown()
        }
