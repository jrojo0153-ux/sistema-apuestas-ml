import requests

def obtener_partidos():
    print("🌐 Consultando API de ESPN (Sustituyendo SofaScore)...")
    
    # Endpoint general de fútbol
    url = "https://site.api.espn.com/apis/site/v2/sports/soccer/all/scoreboard"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        datos = response.json()
        
        # Validación de seguridad: ¿Hay eventos hoy?
        eventos = datos.get('events')
        if eventos is None:
            print("⚠️ No hay eventos programados en el JSON de ESPN.")
            return []
        
        lista_partidos = []
        
        for evento in eventos:
            try:
                # Acceso seguro a las competiciones
                competencias = evento.get('competitions', [{}])
                comp = competencias[0]
                
                # Extraer equipos
                competitors = comp.get('competitors', [])
                if len(competitors) < 2: continue
                
                home_team = competitors[0].get('team', {}).get('displayName', 'Local')
                away_team = competitors[1].get('team', {}).get('displayName', 'Visitante')
                
                # --- MANEJO ROBUSTO DE ODDS ---
                # ESPN no siempre tiene cuotas para todas las ligas
                odds_list = comp.get('odds', [])
                home_odds = 1.95
                away_odds = 3.40
                
                if odds_list and isinstance(odds_list, list):
                    item = odds_list[0]
                    # Intentar sacar cuotas reales si existen
                    home_odds = item.get('home', {}).get('odds', 1.95)
                    away_odds = item.get('away', {}).get('odds', 3.40)

                lista_partidos.append({
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_odds": float(home_odds),
                    "away_odds": float(away_odds)
                })
                
            except Exception as e:
                # Si un partido falla, pasamos al siguiente sin tumbar todo el script
                continue

        print(f"✅ API exitosa: {len(lista_partidos)} partidos procesados.")
        return lista_partidos

    except Exception as e:
        print(f"❌ Error crítico en API ESPN: {e}")
        return []
