import json
import os
from datetime import datetime

FILE = "ml/picks_history.json"

def save_picks(picks):
    data = []

    if os.path.exists(FILE):
        with open(FILE, "r") as f:
            data = json.load(f)

    for p in picks:
        data.append({
            "match": p["match"],
            "pick": p["pick"],
            "odds": p["odds"],
            "date": str(datetime.now()),
            "result": None
        })

    with open(FILE, "w") as f:
        json.dump(data, f, indent=2)
