import os
import requests
# 1. IMPORTACIONES (Asegúrate de que estas rutas sean correctas)
from data.live_matches import obtener_partidos
from ml.model import entrenar_modelo, cargar_modelo, predecir
from core.value import evaluar_apuesta
from portfolio.bankroll import calcular_apuesta
from config import *

# 2. DEFINICIÓN DE FUNCIONES (Deben ir antes del main)
def enviar_telegram(mensaje):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if token and chat_id:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        try:
            # timeout de 10s para que no se trabe el bot si falla Telegram
            requests.post(url, json={"chat_id": chat_id, "text": mensaje, "parse_mode": "Markdown"}, timeout=10)
        except Exception as e:
            print(f"Error Telegram: {e}")
    else:
        print("⚠️ Alerta: TELEGRAM_BOT_TOKEN o CHAT_ID no configurados.")

def main():
    print("🚀 SISTEMA QUANTBET AI - MODO API ESPN")
    
    # Cargar modelo
    modelo = cargar_modelo()
    if modelo is None:
        print("⚠️ Modelo no detectado. Entrenando...")
        modelo = entrenar_modelo()

    if modelo is None:
        print("❌ No se puede continuar sin modelo.")
        return

    # Obtener partidos vía API ESPN
    partidos = obtener_partidos()
    
    if not partidos:
        print("❌ No se pudieron obtener partidos.")
        return

    # Ahora sí, enviar_telegram está definida y lista para usarse
    enviar_telegram(f"📡 *API ESPN Activa:* Analizando {len(partidos)} eventos en tiempo real...")

    for partido in partidos:
        prob = predecir(modelo, partido)
        if prob:
            # Usamos .get() por seguridad si la API de ESPN cambia
            odds = partido.get('home_odds', 1.95)
            edge = prob - (1/odds)
            
            if edge > EDGE_MINIMO:
                ev, kelly = evaluar_apuesta(prob, odds)
                if kelly > 0:
                    stake = calcular_apuesta(kelly * KELLY_FRACCION, BANKROLL_INICIAL)
                    msg = (f"🎯 *SIGNAL QUANTBET AI*\n"
                           f"⚽ {partido['home_team']} vs {partido['away_team']}\n"
                           f"📈 Edge: {round(edge*100, 2)}%\n"
                           f"💰 Cuota: {odds}\n"
                           f"📊 Stake: ${round(stake, 2)}")
                    enviar_telegram(msg)

    print("✅ Proceso completado exitosamente.")

# 3. PUNTO DE ENTRADA
if __name__ == "__main__":
    main()
