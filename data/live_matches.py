import requests
import os

def obtener_partidos():
    print("🌐 Consultando API de ESPN (Sustituyendo SofaScore)...")
    
    # Endpoints de ESPN para Fútbol (puedes cambiar 'usa.1' por 'mex.1' para Liga MX o 'eng.1' etc.)
    # El endpoint 'soccer/scoreboard' trae los partidos más importantes del día
    url = "https://site.api.espn.com/apis/site/v2/sports/soccer/all/scoreboard"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            print(f"❌ Error API ESPN: {response.status_code}")
            return []

        datos = response.json()
        eventos = datos.get('events', [])
        
        lista_partidos = []
        
        for evento in eventos:
            try:
                # Extraer información básica
                competencia = evento['competitions'][0]
                home_team = competencia['competitors'][0]['team']['displayName']
                away_team = competencia['competitors'][1]['team']['displayName']
                
                # Intentar extraer cuotas (Odds) de la API si están disponibles
                # ESPN a veces provee 'odds', si no, usamos tu base de 1.95
                odds_data = competencia.get('odds', [{}])[0]
                home_odds = odds_data.get('home', {}).get('odds', 1.95)
                away_odds = odds_data.get('away', {}).get('odds', 3.40)

                lista_partidos.append({
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_odds": float(home_odds),
                    "away_odds": float(away_odds)
                })
            except (KeyError, IndexError):
                continue

        if not lista_partidos:
            print("⚠️ No se encontraron partidos activos en ESPN.")
            return []

        print(f"✅ API exitosa: {len(lista_partidos)} partidos obtenidos de ESPN.")
        return lista_partidos

    except Exception as e:
        print(f"❌ Error crítico en API ESPN: {e}")
        return []
