import requests
import os

API_KEY = os.getenv("API_KEY_ODDS")

def obtener_cuotas(partidos):
    try:
        url = f"https://api.the-odds-api.com/v4/sports/soccer/odds"

        params = {
            "apiKey": API_KEY,
            "regions": "eu",
            "markets": "h2h"
        }

        response = requests.get(url, params=params)

        if response.status_code != 200:
            print("❌ Error API cuotas:", response.text)
            return []

        data = response.json()

        cuotas = []

        for partido in partidos:
            local = partido.get("home")
            visitante = partido.get("away")

            for game in data:
                if game["home_team"] == local and game["away_team"] == visitante:

                    try:
                        odds = game["bookmakers"][0]["markets"][0]["outcomes"]

                        cuota_local = odds[0]["price"]
                        cuota_visitante = odds[1]["price"]

                        cuotas.append({
                            "partido": f"{local} vs {visitante}",
                            "local": cuota_local,
                            "visitante": cuota_visitante
                        })

                    except Exception as e:
                        print("⚠️ Error extrayendo cuotas:", e)

        return cuotas

    except Exception as e:
        print("❌ Error general cuotas:", e)
        return []
