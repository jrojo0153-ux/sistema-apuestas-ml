from data.api_football import get_matches
from data.odds_api import get_odds
from core.parlay_builder import build_parlays
from utils.telegram import send_telegram_message

def run_pipeline():
    print("📊 Ejecutando pipeline...")

    matches = get_matches()
    odds_data = get_odds()

    print(f"📅 Partidos: {len(matches)}")
    print(f"💰 Odds: {len(odds_data)}")

    picks = []

    # ✅ 1. INTENTAR CON API FOOTBALL
    for match in matches[:10]:
        try:
            home = match["teams"]["home"]["name"]
            away = match["teams"]["away"]["name"]

            picks.append({
                "match": f"{home} vs {away}",
                "pick": home,
                "odds": 2.0
            })
        except:
            continue

    # ✅ 2. SI NO HAY → USAR ODDS API (FIX REAL)
    if not picks:
        print("⚠️ Usando Odds API...")

        for game in odds_data:
            try:
                home = game["home_team"]
                away = game["away_team"]

                # 🔥 obtener cuota real
                bookmakers = game.get("bookmakers", [])

                if bookmakers:
                    markets = bookmakers[0].get("markets", [])
                    if markets:
                        outcomes = markets[0].get("outcomes", [])

                        if outcomes:
                            odd = outcomes[0]["price"]
                        else:
                            odd = 2.0
                    else:
                        odd = 2.0
                else:
                    odd = 2.0

                picks.append({
                    "match": f"{home} vs {away}",
                    "pick": home,
                    "odds": odd
                })

            except Exception as e:
                print("Error odds:", e)
                continue

    # ❌ SI AÚN NO HAY PICKS
    if not picks:
        print("❌ No picks generados")
        return

    # 🔥 SIEMPRE GENERAR PARLAYS
    parlays = build_parlays(picks)

    message = "🔥 PARLAYS DEL DÍA\n\n"

    for p in parlays:
        message += f"{p['type']}:\n"
        for leg in p["legs"]:
            message += f"• {leg['match']} → {leg['pick']} (cuota {leg['odds']})\n"
        message += f"💰 Total: {p['odds']}\n\n"

    print(message)
    send_telegram_message(message)
