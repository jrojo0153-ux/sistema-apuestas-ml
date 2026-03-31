import os
import requests
from data.live_matches import obtener_partidos
from ml.model import entrenar_modelo, cargar_modelo, predecir
from core.value import evaluar_apuesta
from portfolio.bankroll import calcular_apuesta
from config import *

def enviar_telegram(mensaje):
    """
    Envía notificaciones a Telegram gestionando errores de conexión
    para cumplir con las normas de seguridad.
    """
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    
    if not token or not chat_id:
        print("⚠️ Configuración de Telegram incompleta (Secrets).")
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id, 
        "text": mensaje, 
        "parse_mode": "Markdown"
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        # Registro del error en lugar de ignorarlo (evita avisos de Bandit/CodeFactor)
        print(f"❌ Fallo al enviar Telegram: {e}")

def main():
    print("🚀 SISTEMA EDGE PRO ACTIVADO")
    
    # 1. Gestión del Modelo: Cargar existente o crear uno nuevo si no existe la carpeta
    modelo = cargar_modelo()
    if modelo is None:
        print("⚠️ Modelo no detectado o corrupto. Iniciando entrenamiento...")
        modelo = entrenar_modelo()

    if modelo is None:
        print("❌ Error crítico: El sistema no pudo inicializar el modelo de IA.")
        return

    # 2. Obtención de datos de SofaScore
    partidos = obtener_partidos()
    if not partidos:
        print("❌ No se encontraron partidos procesables para hoy.")
        # Opcional: enviar_telegram("⚠️ No hay partidos disponibles en SofaScore hoy.")
        return

    enviar_telegram(f"📡 *SofaScore:* Analizando {len(partidos)} eventos en busca de Edge...")

    apuestas_encontradas = 0
    for partido in partidos:
        # 3. Predicción de la IA
        prob = predecir(modelo, partido)
        
        if prob:
            odds = partido.get('home_odds', 0)
            if odds <= 1:
                continue
                
            # 4. Cálculo del Edge (Ventaja sobre la casa)
            prob_casa = 1 / odds
            edge = prob - prob_casa
            
            # 5. Filtro de Alto Valor (Configurado en config.py)
            if edge > EDGE_MINIMO:
                ev, kelly = evaluar_apuesta(prob, odds)
                
                if kelly > 0:
                    apuestas_encontradas += 1
                    # Gestión de Bankroll
                    stake = calcular_apuesta(kelly * KELLY_FRACCION, BANKROLL_INICIAL)
                    
                    msg = (f"💎 *ALERTA DE ALTO VALOR*\n"
                           f"⚽ {partido['home_team']} vs {partido['away_team']}\n"
                           f"📈 *Edge:* {round(edge*100, 2)}%\n"
                           f"📈 *Prob. IA:* {round(prob*100, 1)}%\n"
                           f"💰 *Cuota:* {odds}\n"
                           f"📊 *Stake Sugerido:* ${round(stake, 2)}")
                    
                    enviar_telegram(msg)

    print(f"✅ Ciclo completado. Apuestas enviadas: {apuestas_encontradas}")

if __name__ == "__main__":
    main()
