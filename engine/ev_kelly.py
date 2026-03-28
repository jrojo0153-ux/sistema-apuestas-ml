"""
engine/ev_kelly.py — Cálculo de valor esperado + criterio de Kelly
"""

import numpy as np


class MotorEV:
    """
    Genera señales de apuestas basadas en valor esperado (EV)
    """

    def __init__(self, ev_min=0.02):
        self.ev_min = ev_min

    def calcular_ev(self, prob, cuota):
        """
        EV = (probabilidad * cuota) - 1
        """
        return (prob * cuota) - 1

    def kelly(self, prob, cuota):
        """
        Fórmula de Kelly
        """
        b = cuota - 1
        q = 1 - prob

        k = (b * prob - q) / b

        return max(k, 0)  # nunca negativo

    def generar_senales(self, df, probs):
        """
        Genera lista de apuestas recomendadas
        """
        señales = []

        for i in range(len(df)):
            cuotas = [
                df.iloc[i]["cuota_local"],
                df.iloc[i]["cuota_empate"],
                df.iloc[i]["cuota_visitante"]
            ]

            for j in range(3):  # 0=local, 1=empate, 2=visitante
                prob = probs[i][j]
                cuota = cuotas[j]

                ev = self.calcular_ev(prob, cuota)

                if ev > self.ev_min:
                    k = self.kelly(prob, cuota)

                    señal = {
                        "index": i,
                        "tipo": j,
                        "prob": prob,
                        "cuota": cuota,
                        "ev": ev,
                        "kelly": k
                    }

                    señales.append(señal)

        print(f"📊 Señales generadas: {len(señales)}")

        return señales
