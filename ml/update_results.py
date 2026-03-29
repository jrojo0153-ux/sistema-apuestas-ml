import json
import os
from ml.results import get_result
from ml.model import save_result

FILE = "ml/picks_history.json"

def update_results():
    if not os.path.exists(FILE):
        return

    with open(FILE, "r") as f:
        data = json.load(f)

    updated = False

    for p in data:
        if p["result"] is not None:
            continue

        try:
            teams = p["match"].split(" vs ")
            home = teams[0]
            away = teams[1]

            real_winner = get_result(home, away)

            if real_winner is None:
                continue

            if p["pick"] == real_winner:
                result = 1
            else:
                result = 0

            p["result"] = result

            # 🔥 guardar en ML
            save_result(p, result)

            updated = True

        except:
            continue

    if updated:
        with open(FILE, "w") as f:
            json.dump(data, f, indent=2)
