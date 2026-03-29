import json
import os

MODEL_FILE = "ml/data.json"

def load_data():
    if not os.path.exists(MODEL_FILE):
        return []
    
    with open(MODEL_FILE, "r") as f:
        return json.load(f)

def save_result(pick, result):
    data = load_data()

    data.append({
        "match": pick["match"],
        "pick": pick["pick"],
        "odds": pick["odds"],
        "result": result  # 1 = win, 0 = lose
    })

    with open(MODEL_FILE, "w") as f:
        json.dump(data, f, indent=2)

def get_winrate():
    data = load_data()

    if not data:
        return 0.5  # default

    wins = sum(1 for d in data if d["result"] == 1)
    return wins / len(data)
