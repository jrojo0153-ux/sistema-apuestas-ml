import requests
import os

API_KEY = os.getenv("API_FOOTBALL_KEY")

def get_result(home, away):
    url = "https://v3.football.api-sports.io/fixtures"

    headers = {
        "x-apisports-key": API_KEY
    }

    params = {
        "last": 20
    }

    try:
        res = requests.get(url, headers=headers, params=params)

        if res.status_code != 200:
            return None

        data = res.json()["response"]

        for match in data:
            h = match["teams"]["home"]["name"]
            a = match["teams"]["away"]["name"]

            if home in h and away in a:
                goals_home = match["goals"]["home"]
                goals_away = match["goals"]["away"]

                if goals_home > goals_away:
                    return h
                elif goals_away > goals_home:
                    return a
                else:
                    return "Draw"

        return None

    except:
        return None
