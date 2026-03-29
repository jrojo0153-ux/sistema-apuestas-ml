from data.live_matches import obtener_partidos_hoy
from data.loader import obtener_cuotas
from engine.ev_kelly import calcular_ev_y_kelly
from models.model import cargar_modelo, predecir
from portfolio.bankroll import BankrollManager

import os
import requests


# =========================
# 📩 TELEGRAM
# =========================
def enviar_telegram(msg):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        print("⚠️ Telegram no configurado")
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"

    try:
        requests.post(url, json={
            "chat_id": chat_id,
            "text": msg
        })
        print("📩 Alerta enviada a Telegram")
    except Exception as e:
        print(f"❌ Error Telegram: {e}")


# =========================
# 🧠 MAIN BOT
# =========================
def main():
    print("🚀 SISTEMA PRO INICIADO")

    # 1. Obtener partidos
    partidos = obtener_partidos_hoy()
    print(f"📊 Partidos encontrados: {len(partidos)}")

    if not partidos:
        print("❌ No hay partidos")
        return

    # 2. Obtener cuotas
    cuotas = obtener_cuotas(partidos)

    if not cuotas:
        print("❌ No hay cuotas")
        return

    # 3. Cargar modelo IA
    modelo = cargar_modelo()

    # 4. Predecir probabilidades
    probabilidades = predecir(modelo, len(partidos))

    # 5. Preparar cuotas
    cuotas_home = [c["cuota_home"] for c in cuotas]

    # 6. Calcular EV + Kelly
    resultados = calcular_ev_y_kelly(probabilidades, cuotas_home)

    # 7. Inicializar bankroll
    bank = BankrollManager(1000)

    # =========================
    # 🔥 LOOP PRINCIPAL
    # =========================
    for partido, res, cuota in zip(partidos, resultados, cuotas_home):

        if res["ev"] > 0:

            apuesta = bank.calcular_apuesta(res["kelly"])

            if apuesta <= 0:
                continue

            msg = f"""
🔥 VALUE BET

⚽ {partido['home']} vs {partido['away']}

📈 Cuota: {cuota}
📊 EV: {res['ev']}
🧠 Kelly: {res['kelly']}
💰 Apuesta: ${apuesta}
"""

            print(msg)

            # Enviar alerta
            enviar_telegram(msg)

            # Simulación (luego conectamos resultados reales)
            gano = res["ev"] > 0.05

            # Actualizar bankroll
            bank.actualizar_bankroll(apuesta, cuota, gano)

    # =========================
    # 📊 RESUMEN FINAL
    # =========================
    resumen = bank.resumen()

    print("\n📊 RESUMEN:")
    print(resumen)

    print("✅ SISTEMA FINALIZADO")


# =========================
# ▶️ RUN
# =========================
if __name__ == "__main__":
    main()
