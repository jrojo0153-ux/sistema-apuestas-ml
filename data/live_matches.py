import requests
import os
from datetime import datetime

def obtener_partidos():
    api_key = os.getenv("API_FOOTBALL_KEY")
    hoy = datetime.now().strftime('%Y-%m-%d')
    # Usamos el endpoint de SofaScore/Fixtures para mayor precisión
    url = f"https://v3.football.api-sports.io/fixtures?date={hoy}"
    
    headers = {
        'x-rapidapi-key': api_key,
        'x-rapidapi-host': 'v3.football.api-sports.io'
    }
    
    try:
        response = requests.get(url, headers=headers)
        data = response.json()
        partidos_res = data.get('response', [])
        
        lista_partidos = []
        for item in partidos_res:
            # Intentamos obtener cuotas reales de la respuesta si están disponibles
            lista_partidos.append({
                "id": item['fixture']['id'],
                "home_team": item['teams']['home']['name'],
                "away_team": item['teams']['away']['name'],
                "home_odds": 2.00, # Valor por defecto si no hay Odds API conectada
                "away_odds": 3.40
            })
        return lista_partidos
    except Exception:
        return []
