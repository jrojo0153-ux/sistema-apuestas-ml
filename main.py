# main.py

from data.live_matches import obtener_partidos
from ml.model import entrenar_modelo, cargar_modelo, predecir
from core.value import evaluar_apuesta
from portfolio.bankroll import calcular_apuesta

from config import *


def main():
    print("🚀 SISTEMA PRO IA INICIADO")

    partidos = obtener_partidos()

    if not partidos:
        print("❌ No hay partidos")
        return

    modelo = cargar_modelo()

    if modelo is None:
        print("⚠️ Entrenando modelo...")
        modelo = entrenar_modelo()

    bankroll = BANKROLL_INICIAL
    apuestas = 0

    for partido in partidos:

        if apuestas >= MAX_APUESTAS_DIA:
            print("📛 Límite diario alcanzado")
            break

        odds = partido.get("home_odds")

        if odds is None or not (CUOTA_MINIMA <= odds <= CUOTA_MAXIMA):
            continue

        prob = predecir(modelo, partido)

        if prob is None or not (PROB_MINIMA <= prob <= PROB_MAXIMA):
            continue

        ev, kelly = evaluar_apuesta(prob, odds)

        edge = prob - (1 / odds)

        if (
            ev < EV_MINIMO or
            kelly <= 0 or
            edge < EDGE_MINIMO
        ):
            continue

        kelly_ajustado = kelly * KELLY_FRACCION

        stake = min(
            calcular_apuesta(kelly_ajustado, bankroll),
            bankroll * APUESTA_MAXIMA_PCT
        )

        if stake <= 0:
            continue

        apuestas += 1

        print("\n🔥 VALUE BET")
        print(f"{partido['home_team']} vs {partido['away_team']}")
        print(f"Prob: {round(prob, 3)}")
        print(f"EV: {round(ev, 3)}")
        print(f"Stake: ${stake}")

    print("\n✅ FINALIZADO")


if __name__ == "__main__":
    main()
