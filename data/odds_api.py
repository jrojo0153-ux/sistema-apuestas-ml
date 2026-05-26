import random

def obtener_cuotas(partidos):
    return [
        {
            "partido": f"{p.get('home', 'Local')} vs {p.get('away', 'Visitante')}",
            "home_odds": round(random.uniform(1.5, 2.8), 2),
            "away_odds": round(random.uniform(1.8, 3.5), 2)
        }
        for p in partidos
    ]