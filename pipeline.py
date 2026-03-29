from data.api_football import get_matches
from data.odds_api import get_odds
from core.value import calculate_edge
from core.parlay_builder import build_parlays
from utils.telegram import send_telegram_message

from ml.storage import save_picks
from ml.update_results import update_results


def run_pipeline():
    print("📊 Ejecutando pipeline...")

    # 🧠 1. ACTUALIZAR RESULTADOS (ML)
    print("🧠 Actualizando resultados...")
    update_results()

    # 📡 2. OBTENER DATOS
    matches = get_matches()
    odds_data = get_odds()

    print(f"📅 Partidos API: {len(matches)}")
    print(f"💰 Odds API: {len(odds_data)}")

    picks = []
    used_matches = set()

    # 🔥 3. GENERAR PICKS (SHARP LOGIC)
    for game in odds_data:
        try:
            home = game.get("home_team")
            away = game.get("away_team")

            if not home or not away:
                continue

            match_id = f"{home}-{away}"

            # ❌ evitar duplicados
            if match_id in used_matches:
                continue

            bookmakers = game.get("bookmakers", [])
            if not bookmakers:
                continue

            markets = bookmakers[0].get("markets", [])
            if not markets:
                continue

            outcomes = markets[0].get("outcomes", [])
            if not outcomes:
                continue

            best_pick = None
            best_edge = -999

            for o in outcomes:
                pick_name = o.get("name")
                odd = o.get("price")

                if not pick_name or not odd:
                    continue

                # ❌ FILTRO 1: evitar cuotas basura
                if odd < 1.4 or odd > 4.0:
                    continue

                # ❌ FILTRO 2: evitar empates (opcional pero recomendado)
                if pick_name.lower() == "draw":
                    continue

                edge = calculate_edge(odd)

                # 🔥 elegir mejor pick por partido
                if edge > best_edge:
                    best_edge = edge
                    best_pick = {
                        "match": f"{home} vs {away}",
                        "pick": pick_name,
                        "odds": odd,
                        "edge": edge
                    }

            if best_pick:
                picks.append(best_pick)
                used_matches.add(match_id)

        except Exception as e:
            print("⚠️ Error odds:", e)
            continue

    # 🔥 4. FALLBACK (SI ODDS FALLA)
    if not picks:
        print("⚠️ Fallback API-Football...")

        for match in matches:
            try:
                home = match["teams"]["home"]["name"]
                away = match["teams"]["away"]["name"]

                picks.append({
                    "match": f"{home} vs {away}",
                    "pick": home,
                    "odds": 1.8,
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

    # 🔥 6. CREAR PARLAYS
    parlays = build_parlays(picks)

    # 🧠 7. GUARDAR PICKS (ML)
    save_picks(picks)

    # 📲 8. MENSAJE LIMPIO
    message = "🔥 PARLAYS SHARP (ML + EDGE REAL)\n\n"

    for p in parlays:
        message += f"{p['type']}:\n"

        for leg in p["legs"]:
            message += (
                f"• {leg['match']} → {leg['pick']} "
                f"(cuota {leg['odds']}, edge {round(leg['edge'], 3)})\n"
            )

        message += f"💰 Cuota total: {p['odds']}\n\n"

    print(message)

    # 📲 9. ENVIAR TELEGRAM
    send_telegram_message(message)
