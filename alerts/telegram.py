import requests
import os

def enviar_alerta(mensaje):
    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        print("⚠️ Telegram no configurado")
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"

    requests.post(url, json={
        "chat_id": chat_id,
        "text": mensaje
    })
