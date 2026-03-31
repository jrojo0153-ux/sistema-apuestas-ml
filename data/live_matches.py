import requests
import os

def obtener_partidos():
    # GitHub Actions inyecta automáticamente el Secret aquí
    api_key = os.getenv("API_FOOTBALL_KEY")
    url = "https://v3.football.api-sports.io/fixtures?live=all" # O por fecha
    
    headers = {
        'x-rapidapi-key': api_key,
        'x-rapidapi-host': 'v3.football.api-sports.io'
    }
    
    try:
        response = requests.get(url, headers=headers)
        data = response.json()
        
        partidos_procesados = []
        for item in data['response']:
            # Mantenemos los nombres de llaves que espera tu main.py
            partido = {
                "home_team": item['teams']['home']['name'],
                "away_team": item['teams']['away']['name'],
                # Aquí podrías integrar otra API de cuotas o usar valores base
                "home_odds": 1.85, # Valor temporal si la API de fútbol no trae odds
                "away_odds": 2.10
            }
            partidos_procesados.append(partido)
            
        return partidos_procesados
    except Exception as e:
        print(f"❌ Error API: {e}")
        return []
