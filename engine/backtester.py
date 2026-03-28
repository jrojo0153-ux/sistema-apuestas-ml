"""
engine/backtester.py — Simulación de apuestas (backtesting)
"""

import os
import matplotlib
matplotlib.use('Agg')  # IMPORTANTE para GitHub

import matplotlib.pyplot as plt


class Backtester:
    """
    Simula apuestas y genera resultados
    """

    def __init__(self, bankroll_inicial=1000):
        self.bankroll = bankroll_inicial
        self.historial = [bankroll_inicial]

    def run(self, df, señales):
        """
        Ejecuta simulación
        """
        for señal in señales:
            idx = señal["index"]
            tipo = señal["tipo"]
            cuota = señal["cuota"]
            prob = señal["prob"]

            resultado_real = df.iloc[idx]["resultado"]

            apuesta = self.bankroll * 0.02  # 2% fijo

            if tipo == resultado_real:
                ganancia = apuesta * (cuota - 1)
                self.bankroll += ganancia
            else:
                self.bankroll -= apuesta

            self.historial.append(self.bankroll)

        print(f"💰 Bankroll final: {self.bankroll:.2f}")

        self._guardar_grafico()

        return self.bankroll

    def _guardar_grafico(self):
        """
        Guarda gráfica de evolución
        """
        os.makedirs("reports", exist_ok=True)

        plt.figure()
        plt.plot(self.historial)
        plt.title("Evolución del Bankroll")
        plt.xlabel("Apuestas")
        plt.ylabel("Dinero")
        plt.grid()

        plt.savefig("reports/bankroll.png")
        plt.close()

        print("📊 Gráfico guardado en /reports/")
