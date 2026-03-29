import requests

def obtener_partidos_hoy():
    try:
        url = "https://api-football-v1.p.rapidapi.com/v3/fixtures"
        headers = {
            "X-RapidAPI-Key": "TU_API_KEY",
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }
        params = {"live": "all"}

        response = requests.get(url, headers=headers, params=params)

        if response.status_code != 200:
            print("❌ Error API partidos")
            return []

        data = response.json()

        partidos = []
        for match in data.get("response", []):
            home = match["teams"]["home"]["name"]
            away = match["teams"]["away"]["name"]

            partidos.append({
                "home": home,
                "away": away
            })

        return partidos

    except Exception as e:
        print("❌ Error obteniendo partidos:", e)
        return []
