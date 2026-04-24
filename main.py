import os
import requests
# Importar las funciones específicas desde la carpeta 'ml' y el archivo 'model'
from ml.model import entrenar_modelo, cargar_modelo, predecir 
from data.live_matches import obtener_partidos
from core.value import evaluar_apuesta
from portfolio.bankroll import calcular_apuesta
from config import *


def main():
    # 1. Cambiamos el mensaje de inicio para reflejar la realidad técnica
    print("🚀 SISTEMA QUANTBET AI - MODO API ESPN")
    
    modelo = cargar_modelo()
    if modelo is None:
        print("⚠️ Modelo no detectado. Entrenando...")
        modelo = entrenar_modelo()

    if modelo is None:
        print("❌ No se puede continuar sin modelo.")
        return

    # 2. Ahora llama a la nueva lógica de ESPN
    partidos = obtener_partidos()
    
    if not partidos:
        print("❌ No se obtuvieron partidos de la API.")
        return

    # 3. Mensaje de Telegram más profesional
    enviar_telegram(f"📡 *API ESPN Activa:* Analizando {len(partidos)} eventos en tiempo real...")

    for partido in partidos:
        prob = predecir(modelo, partido)
        if prob:
            odds = partido['home_odds']
            edge = prob - (1/odds)
            
            if edge > EDGE_MINIMO:
                ev, kelly = evaluar_apuesta(prob, odds)
                if kelly > 0:
                    stake = calcular_apuesta(kelly * KELLY_FRACCION, BANKROLL_INICIAL)
                    # 4. Etiqueta de la señal cambiada de SCRAPING a ALGORITMO
                    msg = (f"🎯 *SIGNAL QUANTBET AI*\n"
                           f"⚽ {partido['home_team']} vs {partido['away_team']}\n"
                           f"📈 Edge: {round(edge*100, 2)}%\n"
                           f"💰 Cuota: {odds}\n"
                           f"📊 Stake sugerido: ${round(stake, 2)}")
                    enviar_telegram(msg)

    print("✅ Ciclo de análisis completado.")

if __name__ == "__main__":
    main()
