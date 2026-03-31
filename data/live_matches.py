import requests
import os
from datetime import datetime

def obtener_partidos():
    api_key = os.getenv("API_FOOTBALL_KEY")
    hoy = datetime.now().strftime('%Y-%m-%d')
    url = f"https://v3.football.api-sports.io/fixtures?date={hoy}"
    
    headers = {
        'x-rapidapi-key': api_key,
        'x-rapidapi-host': 'v3.football.api-sports.io'
    }
    
    try:
        response = requests.get(url, headers=headers)
        data = response.json()
        partidos_res = data.get('response', [])
        
        if not partidos_res:
            return []
            
        return [{
            "home_team": item['teams']['home']['name'],
            "away_team": item['teams']['away']['name'],
            "home_odds": 1.95, 
            "away_odds": 2.05
        } for item in partidos_res]
    except
        return []
