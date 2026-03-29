import requests
import os

API_KEY = os.getenv("API_KEY_ODDS")


def obtener_cuotas(partidos):
    """
    Obtiene cuotas reales desde The Odds API
    Retorna lista de dicts con:
    - partido
    - home_odds
    - away_odds
    """

    if not partidos:
        print("⚠️ No hay partidos para obtener cuotas")
        return []

    try:
        url = "https://api.the-odds-api.com/v4/sports/soccer/odds"

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
            home = partido.get("home")
            away = partido.get("away")

            if not home or not away:
                continue

            encontrado = False

            for game in data:
                if game.get("home_team") == home and game.get("away_team") == away:

                    try:
                        bookmakers = game.get("bookmakers", [])
                        if not bookmakers:
                            continue

                        markets = bookmakers[0].get("markets", [])
                        if not markets:
                            continue

                        outcomes = markets[0].get("outcomes", [])
                        if len(outcomes) < 2:
                            continue

                        cuota_home = outcomes[0].get("price")
                        cuota_away = outcomes[1].get("price")

                        if cuota_home and cuota_away:
                            cuotas.append({
                                "partido": f"{home} vs {away}",
                                "home_odds": float(cuota_home),
                                "away_odds": float(cuota_away)
                            })
                            encontrado = True

                    except Exception as e:
                        print(f"⚠️ Error procesando cuotas {home vs away}:", e)

            if not encontrado:
                print(f"⚠️ No se encontraron cuotas para {home} vs {away}")

        print(f"📊 Cuotas obtenidas: {len(cuotas)}")

        return cuotas

    except Exception as e:
        print("❌ Error general obteniendo cuotas:", e)
        return []
