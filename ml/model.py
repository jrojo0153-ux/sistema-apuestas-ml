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
        try:
            requests.post(url, json={"chat_id": chat_id, "text": mensaje, "parse_mode": "Markdown"})
        except:
            pass

def main():
    print("🚀 SISTEMA PRO IA INICIADO")
    
    # Intentar cargar, si falla (o no existe la carpeta), entrena y la crea
    modelo = cargar_modelo()
    if modelo is None:
        print("⚠️ Iniciando entrenamiento y creación de archivos...")
        modelo = entrenar_modelo()

    if modelo is None:
        print("❌ Error crítico: No se pudo establecer el modelo de IA.")
        return

    partidos = obtener_partidos()
    if not partidos:
        print("❌ No se encontraron partidos hoy.")
        return

    enviar_telegram(f"🤖 *IA en línea:* Analizando {len(partidos)} partidos del día...")

    for partido in partidos:
        prob = predecir(modelo, partido)
        if prob:
            odds = partido['home_odds']
            edge = prob - (1/odds)
            
            # Solo enviar si hay ventaja real
            if edge > EDGE_MINIMO:
                ev, kelly = evaluar_apuesta(prob, odds)
                if kelly > 0:
                    stake = calcular_apuesta(kelly * KELLY_FRACCION, BANKROLL_INICIAL)
                    msg = (f"🎯 *VALUE BET DETECTADA*\n"
                           f"⚽ {partido['home_team']} vs {partido['away_team']}\n"
                           f"📈 Edge: {round(edge*100, 2)}%\n"
                           f"💰 Cuota: {odds}\n"
                           f"📊 Stake: ${round(stake, 2)}")
                    enviar_telegram(msg)

    print("✅ Proceso finalizado correctamente.")

if __name__ == "__main__":
    main()
