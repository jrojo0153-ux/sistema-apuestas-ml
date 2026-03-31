import requests
import os

def obtener_partidos():
    api_key = os.getenv("API_FOOTBALL_KEY")
    # Endpoint para partidos del día (puedes filtrar por liga con &league=XXX)
    url = "https://v3.football.api-sports.io/fixtures?live=all"
    
    headers = {
        'x-rapidapi-key': api_key,
        'x-rapidapi-host': 'v3.football.api-sports.io'
    }
    
    try:
        response = requests.get(url, headers=headers)
        data = response.json()
        partidos_procesados = []
        
        for item in data.get('response', []):
            partidos_procesados.append({
                "home_team": item['teams']['home']['name'],
                "away_team": item['teams']['away']['name'],
                # Si tu plan no incluye odds, usamos valores base para no romper el flujo
                "home_odds": 1.90, 
                "away_odds": 2.10
            })
        return partidos_procesados
    except Exception as e:
        print(f"❌ Error al conectar con API Football: {e}")
        return []
