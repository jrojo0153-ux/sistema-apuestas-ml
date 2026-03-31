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
        except Exception:
            print("❌ Error Telegram")

def main():
    print("🚀 SISTEMA PRO IA INICIADO")
    enviar_telegram("🤖 *SISTEMA PRO IA:* Analizando jornada...")

    modelo = cargar_modelo()
    if modelo is None:
        print("⚠️ Entrenando modelo...")
        modelo = entrenar_modelo()

    partidos = obtener_partidos()
    if not partidos:
        print("❌ No hay partidos")
        enviar_telegram("⚠️ No se encontraron partidos hoy.")
        print("\n✅ FINALIZADO")
        return

    bankroll = BANKROLL_INICIAL
    apuestas_count = 0

    for partido in partidos:
        if apuestas_count >= MAX_APUESTAS_DIA:
            break

        odds = partido.get("home_odds")
        prob = predecir(modelo, partido)

        if prob and (PROB_MINIMA <= prob <= PROB_MAXIMA):
            ev, kelly = evaluar_apuesta(prob, odds)
            if ev >= EV_MINIMO and kelly > 0:
                apuestas_count += 1
                stake = min(calcular_apuesta(kelly * KELLY_FRACCION, bankroll), bankroll * APUESTA_MAXIMA_PCT)
                
                msg = (f"🎯 *VALUE BET*\n"
                       f"⚽ {partido['home_team']} vs {partido['away_team']}\n"
                       f"📈 Prob: {round(prob*100,1)}%\n"
                       f"💰 Cuota: {odds}\n"
                       f"📊 Stake: ${round(stake,2)}")
                enviar_telegram(msg)

    print(f"✅ FINALIZADO. Picks: {apuestas_count}")

if __name__ == "__main__":
    main()
