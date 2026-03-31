import os
from data.live_matches import obtener_partidos
from ml.model import entrenar_modelo, cargar_modelo, predecir
from core.value import evaluar_apuesta
from portfolio.bankroll import calcular_apuesta
from config import *
import requests

def enviar_telegram(mensaje):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if token and chat_id:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        try:
            requests.post(url, json={"chat_id": chat_id, "text": mensaje, "parse_mode": "Markdown"})
        except Exception as e:
            print(f"Error Telegram: {e}")

def main():
    print("🚀 SISTEMA PRO IA - MODO SCRAPING")
    
    # Cargar o crear modelo
    modelo = cargar_modelo()
    if modelo is None:
        print("⚠️ Modelo no detectado. Entrenando...")
        modelo = entrenar_modelo()

    if modelo is None:
        print("❌ No se puede continuar sin modelo.")
        return

    # Obtener partidos vía Scraping
    partidos = obtener_partidos()
    
    if not partidos:
        print("❌ No se pudieron extraer partidos.")
        return

    enviar_telegram(f"📡 *Scraping Activo:* Analizando {len(partidos)} eventos desde SofaScore...")

    for partido in partidos:
        prob = predecir(modelo, partido)
        if prob:
            odds = partido['home_odds']
            edge = prob - (1/odds)
            
            if edge > EDGE_MINIMO:
                ev, kelly = evaluar_apuesta(prob, odds)
                if kelly > 0:
                    stake = calcular_apuesta(kelly * KELLY_FRACCION, BANKROLL_INICIAL)
                    msg = (f"🎯 *VALUE BET (SCRAPING)*\n"
                           f"⚽ {partido['home_team']} vs {partido['away_team']}\n"
                           f"📈 Edge: {round(edge*100, 2)}%\n"
                           f"💰 Cuota: {odds}\n"
                           f"📊 Stake: ${round(stake, 2)}")
                    enviar_telegram(msg)

    print("✅ Proceso completado.")

if __name__ == "__main__":
    main()
