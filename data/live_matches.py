import requests
import os
from datetime import datetime, timedelta

def obtener_partidos():
    api_key = os.getenv("API_FOOTBALL_KEY")
    # Buscamos hoy y mañana para asegurar que siempre haya cartelera disponible
    hoy = datetime.now().strftime('%Y-%m-%d')
    
    # Endpoint principal de SofaScore/API-Football para partidos del día
    url = f"https://v3.football.api-sports.io/fixtures?date={hoy}"
    
    headers = {
        'x-rapidapi-key': api_key,
        'x-rapidapi-host': 'v3.football.api-sports.io'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        data = response.json()
        partidos_res = data.get('response', [])
        
        if not partidos_res:
            print("⚠️ API no devolvió partidos para hoy. Intentando modo liga...")
            return []

        lista_final = []
        for item in partidos_res:
            # Filtramos solo ligas importantes para que el ML tenga sentido (NBA, Liga MX, Premier, etc.)
            # O simplemente traemos todos y dejamos que el modelo decida
            status = item['fixture']['status']['short']
            
            # Solo partidos programados (NS) o en vivo (1H, 2H, HT)
            if status in ['NS', '1H', '2H', 'HT']:
                lista_final.append({
                    "id": item['fixture']['id'],
                    "home_team": item['teams']['home']['name'],
                    "away_team": item['teams']['away']['name'],
                    # IMPORTANTE: Si la API no trae cuotas, asignamos unas base 
                    # para que el modelo pueda calcular el Edge inicial
                    "home_odds": 1.85, 
                    "away_odds": 3.50
                })
        
        print(f"✅ Se encontraron {len(lista_final)} partidos procesables.")
        return lista_final

    except Exception as e:
        print(f"❌ Error al conectar con la API: {e}")
        return []
