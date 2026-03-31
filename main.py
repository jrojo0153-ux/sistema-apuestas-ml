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
    if not token or not chat_id:
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": mensaje, "parse_mode": "Markdown"}
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"❌ Error de conexión con Telegram: {e}")

def main():
    print("🚀 SISTEMA PRO IA INICIADO")
    
    # Carga segura del modelo
    modelo = cargar_modelo()
    if modelo is None:
        print("⚠️ Modelo no detectado. Intentando entrenar...")
        modelo = entrenar_modelo()

    if modelo is None:
        print("❌ Error crítico: No se pudo inicializar la IA.")
        return

    partidos = obtener_partidos()
    if not partidos:
        print("❌ No hay partidos hoy.")
        return

    enviar_telegram(f"🤖 *IA Activa:* Analizando {len(partidos)} partidos...")

    for partido in partidos:
        prob = predecir(modelo, partido)
        if prob:
            odds = partido.get('home_odds', 0)
            if odds <= 1: continue
            
            edge = prob - (1/odds)
            if edge > EDGE_MINIMO:
                ev, kelly = evaluar_apuesta(prob, odds)
                if kelly > 0:
                    stake = calcular_apuesta(kelly * KELLY_FRACCION, BANKROLL_INICIAL)
                    msg = (f"🎯 *VALUE BET*\n"
                           f"⚽ {partido['home_team']} vs {partido['away_team']}\n"
                           f"📈 Edge: {round(edge*100, 2)}%\n"
                           f"💰 Cuota: {odds}\n"
                           f"📊 Stake: ${round(stake, 2)}")
                    enviar_telegram(msg)

    print("✅ Proceso finalizado.")

if __name__ == "__main__":
    main()
