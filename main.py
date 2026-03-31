import os
import requests
from data.live_matches import obtener_partidos
from ml.model import entrenar_modelo, cargar_modelo, predecir, guardar_edge_alto
from core.value import evaluar_apuesta
from portfolio.bankroll import calcular_apuesta
from config import *

def enviar_telegram(mensaje):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if token and chat_id:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        try:
            requests.post(url, json={"chat_id": chat_id, "text": mensaje, "parse_mode": "Markdown"})
        except:
            print("❌ Error Telegram")

def main():
    print("🚀 SISTEMA EDGE PRO ACTIVADO")
    enviar_telegram("📡 *Escaneando SofaScore...* Buscando Edge de alto valor.")

    modelo = cargar_modelo() or entrenar_modelo()
    partidos = obtener_partidos()

    for partido in partidos:
        odds = partido.get("home_odds")
        prob = predecir(modelo, partido) if modelo else 0.52 # Prob base si no hay modelo

        if prob:
            prob_casa = 1 / odds
            edge = prob - prob_casa

            # DETECCIÓN DE ALTO VALOR (EDGE)
            if edge > EDGE_MINIMO: # Configura esto en 0.05 (5%) en config.py
                print(f"💎 EDGE DETECTADO: {edge}")
                
                # Guardar dato para re-entrenar el ML después
                guardar_edge_alto(partido, prob, edge)
                
                ev, kelly = evaluar_apuesta(prob, odds)
                stake = calcular_apuesta(kelly * KELLY_FRACCION, BANKROLL_INICIAL)

                msg = (f"🚩 *ALERTA DE EDGE ALTO*\n"
                       f"⚽ {partido['home_team']} vs {partido['away_team']}\n"
                       f"📈 Edge: {round(edge*100, 2)}%\n"
                       f"💰 Cuota: {odds}\n"
                       f"📊 Stake sugerido: ${round(stake, 2)}")
                enviar_telegram(msg)

    print("✅ CICLO COMPLETADO")

if __name__ == "__main__":
    main()
