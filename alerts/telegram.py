import requests
import os

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def enviar_alerta(mensaje):
    try:
        url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"

        payload = {
            "chat_id": CHAT_ID,
            "text": mensaje
        }

        response = requests.post(url, json=payload)

        if response.status_code != 200:
            print("❌ Error enviando Telegram:", response.text)

    except Exception as e:
        print("❌ Error Telegram:", e)
