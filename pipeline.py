from data.api_football import get_matches
from data.odds_api import get_odds
from core.value import calculate_edge
from core.parlay_builder import build_parlays
from utils.telegram import send_telegram_message

def run_pipeline():
    print("📊 Ejecutando pipeline...")

    matches = get_matches()
    odds_data = get_odds()

    print(f"📅 Partidos: {len(matches)}")
    print(f"💰 Odds: {len(odds_data)}")

    picks = []

    # 🔥 GENERAR PICKS SIEMPRE (aunque no haya edge real)
    for i, match in enumerate(matches[:10]):
        try:
            home = match["teams"]["home"]["name"]
            away = match["teams"]["away"]["name"]

            # fallback odds si no hay datos
            odd = 2.0
            prob = 0.5

            edge = calculate_edge(prob, odd)

            picks.append({
                "match": f"{home} vs {away}",
                "pick": home,
                "odds": odd,
                "edge": edge
            })

        except Exception as e:
            print("Error match:", e)

    # 🔥 SI NO HAY MATCHES → USAR ODDS DIRECTO
    if not picks and odds_data:
        for game in odds_data[:10]:
            try:
                teams = game["teams"]
                home = teams[0]
                away = teams[1]

                picks.append({
                    "match": f"{home} vs {away}",
                    "pick": home,
                    "odds": 2.0,
                    "edge": 0.01
                })
            except:
                continue

    if not picks:
        print("❌ No picks generados")
        return

    parlays = build_parlays(picks)

    message = "🔥 PARLAYS DEL DÍA\n\n"

    for p in parlays:
        message += f"{p['type']}:\n"
        for leg in p["legs"]:
            message += f"• {leg['match']} → {leg['pick']}\n"
        message += f"Cuota total: {p['odds']}\n\n"

    print(message)
    send_telegram_message(message)
