import os
from datetime import date

import requests
from dotenv import load_dotenv


def main():
    load_dotenv()
    api_key = os.getenv("FOOTBALL_KEY")
    if not api_key:
        raise SystemExit(
            "FOOTBALL_KEY is missing. Add it to a .env file in the project root."
        )

    url = "https://v3.football.api-sports.io/fixtures"
    headers = {"x-apisports-key": api_key}
    params = {
        "league": 39,  # Premier League
        "season": date.today().year,
        "date": date.today().isoformat(),
    }

    try:
        response = requests.get(url, headers=headers, params=params, timeout=20)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise SystemExit(f"API request failed: {exc}") from exc

    try:
        data = response.json()
    except requests.exceptions.JSONDecodeError as exc:
        raise SystemExit("API returned invalid JSON.") from exc

    errors = data.get("errors")
    if errors:
        raise SystemExit(f"API-Sports error: {errors}")

    fixtures = data.get("response", [])
    if not fixtures:
        print(f"No Premier League fixtures found for {params['date']}.")
        return

    for item in fixtures:
        home = item["teams"]["home"]["name"]
        away = item["teams"]["away"]["name"]
        goals_home = item["goals"]["home"]
        goals_away = item["goals"]["away"]
        status = item["fixture"]["status"]["short"]
        print(f"{home} {goals_home} - {goals_away} {away} ({status})")


if __name__ == "__main__":
    main()
