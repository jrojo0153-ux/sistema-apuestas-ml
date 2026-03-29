from data.live_matches import obtener_partidos_hoy
from data.loader import obtener_cuotas
from engine.ev_kelly import calcular_ev_y_kelly
from models.model import cargar_modelo, predecir
from portfolio.bankroll import calcular_apuesta

import os
import requests


def enviar_telegram(msg):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"

    requests.post(url, json={
        "chat_id": chat_id,
        "text": msg
    })


def main():
    print("🚀 SISTEMA PRO INICIADO")

    partidos = obtener_partidos_hoy()
    print(f"📊 Partidos encontrados: {len(partidos)}")

    if not partidos:
        return

    cuotas = obtener_cuotas(partidos)

    modelo = cargar_modelo()
    probabilidades = predecir(modelo, len(partidos))

    cuotas_home = [c["cuota_home"] for c in cuotas]

    resultados = calcular_ev_y_kelly(probabilidades, cuotas_home)

    bankroll = 1000

    for partido, res, cuota in zip(partidos, resultados, cuotas_home):

        if res["ev"] > 0:
            apuesta = calcular_apuesta(bankroll, res["kelly"])

            msg = f"""
🔥 VALUE BET
{partido['home']} vs {partido['away']}
Cuota: {cuota}
EV: {res['ev']}
Kelly: {res['kelly']}
Apuesta: ${apuesta}
"""

            print(msg)
            enviar_telegram(msg)

    print("✅ SISTEMA FINALIZADO")


if __name__ == "__main__":
    main()
