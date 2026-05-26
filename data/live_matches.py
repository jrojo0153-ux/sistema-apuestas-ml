import requests

def obtener_partidos():
    print("🌐 Consultando API de ESPN...")
    url = "https://site.api.espn.com/apis/site/v2/sports/soccer/all/scoreboard"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    }
    
    try:
        with requests.Session() as session:
            response = session.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            datos = response.json()
        
        eventos = datos.get('events')
        if not eventos:
            print("⚠️ No hay eventos programados en el JSON de ESPN.")
            return []
        
        lista_partidos = []
        
        for evento in eventos:
            try:
                competencias = evento.get('competitions', [])
                if not competencias:
                    continue
                comp = competencias[0]
                
                competitors = comp.get('competitors', [])
                if len(competitors) < 2:
                    continue
                
                home_team = 'Local'
                away_team = 'Visitante'
                for competitor in competitors:
                    role = competitor.get('homeAway')
                    name = competitor.get('team', {}).get('displayName')
                    if role == 'home':
                        home_team = name or home_team
                    elif role == 'away':
                        away_team = name or away_team
                
                odds_list = comp.get('odds', [])
                home_odds = 1.95
                away_odds = 3.40
                
                if odds_list and isinstance(odds_list, list):
                    item = odds_list[0]
                    raw_home = item.get('homeTeamOdds', {}).get('moneyLine') or item.get('home', {}).get('odds')
                    raw_away = item.get('awayTeamOdds', {}).get('moneyLine') or item.get('away', {}).get('odds')
                    
                    try:
                        if raw_home is not None:
                            home_odds = float(raw_home)
                    except (ValueError, TypeError):
                        pass
                        
                    try:
                        if raw_away is not None:
                            away_odds = float(raw_away)
                    except (ValueError, TypeError):
                        pass

                lista_partidos.append({
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_odds": float(home_odds),
                    "away_odds": float(away_odds)
                })
                
            except Exception:
                continue

        print(f"✅ API exitosa: {len(lista_partidos)} partidos procesados.")
        return lista_partidos

    except Exception as e:
        print(f"❌ Error crítico en API ESPN: {e}")
        return []