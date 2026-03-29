from data.live_matches import obtener_partidos_hoy
from data.odds_api import obtener_cuotas

from models.modelo_ia import cargar_modelo, predecir
from engine.ev_kelly import calcular_ev_y_kelly
from portfolio.bankroll import calcular_apuesta

from alerts.telegram import enviar_alerta


def main():
    print("🚀 SISTEMA PRO IA INICIADO")

    partidos = obtener_partidos_hoy()

    print(f"📊 Partidos encontrados: {len(partidos)}")

    if not partidos:
        print("❌ No hay partidos")
        return

    cuotas = obtener_cuotas(partidos)

    modelo = cargar_modelo()

    probabilidades = []

    for partido, cuota in zip(partidos, cuotas):
        prob = predecir(
            modelo,
            cuota["home_odds"],
            cuota["away_odds"]
        )
        probabilidades.append(prob)

    resultados = calcular_ev_y_kelly(probabilidades, [c["home_odds"] for c in cuotas])

    bankroll = 1000

    for partido, res in zip(partidos, resultados):

        if res["ev"] > 0:

            apuesta = calcular_apuesta(bankroll, res["kelly"])

            mensaje = f"""
🔥 VALUE BET DETECTADO

⚽ {partido['home']} vs {partido['away']}
📈 EV: {res['ev']}
🧠 Probabilidad IA: {round(probabilidades[0], 2)}
💰 Kelly: {res['kelly']}
💵 Apuesta: ${apuesta}
"""

            print(mensaje)

            enviar_alerta(mensaje)

    print("✅ SISTEMA FINALIZADO")


if __name__ == "__main__":
    main()
