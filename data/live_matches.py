import pandas as pd

def obtener_partidos():
    try:
        df = pd.read_csv("data/historico.csv")

        df.columns = [c.lower().strip() for c in df.columns]

        posibles_home = ["home_team", "equipo_local"]
        posibles_away = ["away_team", "equipo_visitante"]
        posibles_home_odds = ["home_odds", "odd_home", "cuota_local"]
        posibles_away_odds = ["away_odds", "odd_away", "cuota_visitante"]

        def encontrar_col(posibles):
            for p in posibles:
                if p in df.columns:
                    return p
            return None

        c_home = encontrar_col(posibles_home)
        c_away = encontrar_col(posibles_away)
        c_hodds = encontrar_col(posibles_home_odds)
        c_aodds = encontrar_col(posibles_away_odds)

        partidos = []

        for _, row in df.tail(10).iterrows():
            try:
                partidos.append({
                    "home_team": row[c_home] if c_home else "Team A",
                    "away_team": row[c_away] if c_away else "Team B",
                    "home_odds": float(row[c_hodds]) if c_hodds else 2.0,
                    "away_odds": float(row[c_aodds]) if c_aodds else 2.0,
                })
            except:
                continue

        return partidos

    except Exception as e:
        print(f"❌ Error cargando partidos: {e}")
        return []
