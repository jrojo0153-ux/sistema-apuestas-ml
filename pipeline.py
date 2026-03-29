from data.api_football import get_matches
from data.odds_api import get_odds
from core.value import calculate_edge
from core.parlay_builder import build_parlays
from utils.telegram import send_telegram_message

def run_pipeline():
    print("📊 Ejecutando pipeline...")

    matches = get_matches()
    odds_data = get_odds()

    picks = []

    # 🔥 USAR ODDS API COMO BASE REAL
    for game in odds_data:
        try:
            home = game["home_team"]
            away = game["away_team"]

            bookmakers = game.get("bookmakers", [])

            if bookmakers:
                markets = bookmakers[0].get("markets", [])
                outcomes = markets[0].get("outcomes", [])

                for o in outcomes:
                    pick_name = o["name"]
                    odd = o["price"]

                    edge = calculate_edge(odd)

                    picks.append({
                        "match": f"{home} vs {away}",
                        "pick": pick_name,
                        "odds": odd,
                        "edge": edge
                    })

        except Exception as e:
            continue

    if not picks:
        print("❌ No picks generados")
        return

    # 🔥 FILTRO VEGAS (opcional pero potente)
    picks = sorted(picks, key=lambda x: x["edge"], reverse=True)

    parlays = build_parlays(picks)

    message = "🔥 PARLAYS INTELIGENTES (ML)\n\n"

    for p in parlays:
        message += f"{p['type']}:\n"
        for leg in p["legs"]:
            message += f"• {leg['match']} → {leg['pick']} (cuota {leg['odds']})\n"
        message += f"💰 Total: {p['odds']}\n\n"

    print(message)
    send_telegram_message(message)
