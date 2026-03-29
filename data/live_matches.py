import requests
import pandas as pd

def obtener_partidos_hoy():
    url = "https://api.sofascore.com/api/v1/sport/football/events/live"

    try:
        response = requests.get(url, timeout=10)

        if response.status_code != 200:
            print("⚠️ Error SofaScore, usando histórico")
            return obtener_fallback()

        data = response.json()
        partidos = []

        for match in data.get("events", []):
            partidos.append({
                "home": match["homeTeam"]["name"],
                "away": match["awayTeam"]["name"]
            })

        if not partidos:
            print("⚠️ Usando histórico como fallback")
            return obtener_fallback()

        return partidos

    except Exception as e:
        print("⚠️ Error SofaScore:", e)
        return obtener_fallback()


def obtener_fallback():
    try:
        df = pd.read_csv("data/historico.csv")

        # 🔥 VALIDACIÓN FUERTE
        if "home" not in df.columns or "away" not in df.columns:
            print("❌ CSV mal formado. Debe tener columnas: home, away")
            return []

        partidos = []

        for _, row in df.tail(10).iterrows():
            partidos.append({
                "home": str(row["home"]),
                "away": str(row["away"])
            })

        return partidos

    except Exception as e:
        print("❌ Error fallback:", e)
        return []
