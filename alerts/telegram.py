import requests
import os

def enviar_alerta(mensaje: str):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        print("⚠️ Telegram no configurado correctamente")
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"

    try:
        requests.post(url, data={
            "chat_id": chat_id,
            "text": mensaje
        })
        print("📩 Alerta enviada a Telegram")
    except Exception as e:
        print(f"❌ Error enviando alerta: {e}")
