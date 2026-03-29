def build_parlays(picks):
    # 🔥 ordenar por edge (los mejores primero)
    picks = sorted(picks, key=lambda x: x["edge"], reverse=True)

    def create_parlay(name, legs):
        total_odds = 1
        for p in legs:
            total_odds *= p["odds"]

        return {
            "type": name,
            "legs": legs,
            "odds": round(total_odds, 2)
        }

    # 🛡️ CONSERVADOR (mejores edges)
    conservative = create_parlay("🛡️ Conservador", picks[:2])

    # ⚖️ BALANCEADO (mix)
    balanced = create_parlay("⚖️ Balanceado", picks[:4])

    # 💣 AGRESIVO (más riesgo + payout)
    aggressive = create_parlay("💣 Agresivo", picks[:6])

    return [conservative, balanced, aggressive]
