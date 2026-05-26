import os
import requests
from data.live_matches import obtener_partidos
from ml.model import entrenar_modelo, cargar_modelo, predecir
from core.value import evaluar_apuesta
from portfolio.bankroll import calcular_apuesta
from config import *

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
session = requests.Session()

def enviar_telegram(mensaje):
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": mensaje,
            "parse_mode": "Markdown"
        }
        try:
            response = session.post(url, json=payload, timeout=10)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Error Telegram: {e}")
    else:
        print("⚠️ Alerta: TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID no configurados.")

def main():
    print("🚀 SISTEMA QUANTBET AI - MODO API ESPN")
    
    modelo = cargar_modelo()
    if modelo is None:
        print("⚠️ Modelo no detectado. Entrenando...")
        modelo = entrenar_modelo()

    if modelo is None:
        print("❌ No se puede continuar sin modelo.")
        return

    partidos = obtener_partidos()
    if not partidos:
        print("❌ No se pudieron obtener partidos.")
        return

    enviar_telegram(f"📡 *API ESPN Activa:* Analizando {len(partidos)} eventos en tiempo real...")

    for partido in partidos:
        prob = predecir(modelo, partido)
        if prob is not None:
            odds = partido.get('home_odds', 1.95)
            if odds <= 1:
                continue
                
            edge = prob - (1 / odds)
            
            if edge > EDGE_MINIMO:
                ev, kelly = evaluar_apuesta(prob, odds)
                if kelly > 0:
                    stake = calcular_apuesta(kelly * KELLY_FRACCION, BANKROLL_INICIAL)
                    home_team = partido.get('home_team', 'Local')
                    away_team = partido.get('away_team', 'Visitante')
                    msg = (f"🎯 *SIGNAL QUANTBET AI*\n"
                           f"⚽ {home_team} vs {away_team}\n"
                           f"📈 Edge: {round(edge*100, 2)}%\n"
                           f"💰 Cuota: {odds}\n"
                           f"📊 Stake: ${round(stake, 2)}")
                    enviar_telegram(msg)

    print("✅ Proceso completado exitosamente.")

if __name__ == "__main__":
    main()