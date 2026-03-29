import random

def build_parlays(picks):
    random.shuffle(picks)

    def create_parlay(name, size):
        selected = picks[:size]
        odds = 1
        for p in selected:
            odds *= p["odds"]

        return {
            "type": name,
            "legs": selected,
            "odds": round(odds, 2)
        }

    return [
        create_parlay("🛡️ Conservador", 2),
        create_parlay("⚖️ Balanceado", 3),
        create_parlay("💣 Agresivo", 4)
    ]
