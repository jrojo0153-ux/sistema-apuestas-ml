import pandas as pd

def obtener_partidos():
    try:
        df = pd.read_csv("data/historico.csv")

        partidos = []
        for _, row in df.tail(10).iterrows():
            partidos.append({
                "home_team": row.get("home_team", "Team A"),
                "away_team": row.get("away_team", "Team B"),
                "home_odds": float(row.get("home_odds", 2.0)),
                "away_odds": float(row.get("away_odds", 2.0)),
                "resultado": int(row.get("resultado", 0))
            })

        return partidos

    except Exception as e:
        print(f"❌ Error cargando partidos: {e}")
        return []
