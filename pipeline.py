from data.api_football import get_matches
from data.odds_api import get_odds
from core.value import calculate_edge
from core.parlay_builder import build_parlays
from utils.telegram import send_telegram_message

from ml.storage import save_picks
from ml.update_results import update_results


def run_pipeline():
    print("📊 Ejecutando pipeline...")

    # 🧠 1. ACTUALIZAR RESULTADOS PASADOS (ML AUTO)
    print("🧠 Actualizando resultados...")
    update_results()

    # 📡 2. OBTENER DATOS
    matches = get_matches()
    odds_data = get_odds()

    print(f"📅 Partidos API-Football: {len(matches)}")
    print(f"💰 Odds disponibles: {len(odds_data)}")

    picks = []

    # 🔥 3. GENERAR PICKS DESDE ODDS API (FUENTE PRINCIPAL)
    for game in odds_data:
        try:
            home = game.get("home_team")
            away = game.get("away_team")

            bookmakers = game.get("bookmakers", [])

            if not bookmakers:
                continue

            markets = bookmakers[0].get("markets", [])
            if not markets:
                continue

            outcomes = markets[0].get("outcomes", [])
            if not outcomes:
                continue

            for o in outcomes:
                pick_name = o.get("name")
                odd = o.get("price", 2.0)

                if not pick_name or not odd:
                    continue

                # 🧠 EDGE REAL (ML)
                edge = calculate_edge(odd)

                picks.append({
                    "match": f"{home} vs {away}",
                    "pick": pick_name,
                    "odds": odd,
                    "edge": edge
                })

        except Exception as e:
            print("⚠️ Error procesando odds:", e)
            continue

    # 🔥 4. FALLBACK SI ODDS FALLA (USA API FOOTBALL)
    if not picks:
        print("⚠️ Usando fallback con API-Football...")

        for match in matches:
            try:
                home = match["teams"]["home"]["name"]
                away = match["teams"]["away"]["name"]

                picks.append({
                    "match": f"{home} vs {away}",
                    "pick": home,
                    "odds": 2.0,
                    "edge": 0.01
                })

            except:
                continue

    # ❌ SI AÚN NO HAY PICKS
    if not picks:
        print("❌ No picks generados")
        return

    # 🔥 5. ORDENAR POR EDGE (ESTILO VEGAS)
    picks = sorted(picks, key=lambda x: x["edge"], reverse=True)

    # 🔥 6. CREAR PARLAYS SIEMPRE
    parlays = build_parlays(picks)

    # 🧠 7. GUARDAR PICKS PARA APRENDIZAJE
    save_picks(picks)

    # 📲 8. FORMATEAR MENSAJE
    message = "🔥 PARLAYS INTELIGENTES (ML + EDGE)\n\n"

    for p in parlays:
        message += f"{p['type']}:\n"
        for leg in p["legs"]:
            message += (
                f"• {leg['match']} → {leg['pick']} "
                f"(cuota {leg['odds']}, edge {round(leg['edge'], 3)})\n"
            )
        message += f"💰 Cuota total: {p['odds']}\n\n"

    print(message)

    # 📲 9. ENVIAR A TELEGRAM
    send_telegram_message(message)
