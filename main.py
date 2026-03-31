import os
import requests
from data.live_matches import obtener_partidos
from ml.model import entrenar_modelo, cargar_modelo, predecir
from core.value import evaluar_apuesta
from portfolio.bankroll import calcular_apuesta
from config import *

def enviar_telegram(mensaje):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if token and chat_id:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": mensaje, "parse_mode": "Markdown"}
        try:
            requests.post(url, json=payload)
        except:
            print("❌ Error enviando a Telegram")

def main():
    print("🚀 SISTEMA PRO IA INICIADO")
    partidos = obtener_partidos()

    if not partidos:
        print("❌ No hay partidos disponibles hoy")
        return

    modelo = cargar_modelo()
    if modelo is None:
        print("⚠️ Modelo no encontrado. Entrenando...")
        modelo = entrenar_modelo()

    bankroll = BANKROLL_INICIAL
    apuestas_realizadas = 0

    for partido in partidos:
        if apuestas_realizadas >= MAX_APUESTAS_DIA:
            break

        odds = partido.get("home_odds")
        if odds is None or not (CUOTA_MINIMA <= odds <= CUOTA_MAXIMA):
            continue

        prob = predecir(modelo, partido)
        if prob is None or not (PROB_MINIMA <= prob <= PROB_MAXIMA):
            continue

        ev, kelly = evaluar_apuesta(prob, odds)
        edge = prob - (1 / odds)

        if ev < EV_MINIMO or kelly <= 0 or edge < EDGE_MINIMO:
            continue

        kelly_ajustado = kelly * KELLY_FRACCION
        stake = min(
            calcular_apuesta(kelly_ajustado, bankroll),
            bankroll * APUESTA_MAXIMA_PCT
        )

        if stake > 0:
            apuestas_realizadas += 1
            print(f"🔥 VALUE BET DETECTADA: {partido['home_team']}")
            
            # Notificación detallada para Telegram
            msg = (
                f"🎯 *NUEVA VALUE BET*\n"
                f"⚽ {partido['home_team']} vs {partido['away_team']}\n"
                f"📈 Probabilidad: {round(prob * 100, 2)}%\n"
                f"💰 Cuota: {odds}\n"
                f"📊 Stake sugerido: ${round(stake, 2)}"
            )
            enviar_telegram(msg)

    print(f"\n✅ FINALIZADO. Apuestas encontradas: {apuestas_realizadas}")

if __name__ == "__main__":
    main()
