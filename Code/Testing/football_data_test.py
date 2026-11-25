from datetime import date, timedelta
import os
import dotenv
import requests

dotenv.load_dotenv()

uri = "https://api.football-data.org/v4/matches"
headers = {"X-Auth-Token": os.getenv("FOOTBALL_DATA_KEY")}
today = date.today()
date_ranges = (
    ("Past matches", today - timedelta(days=7), today - timedelta(days=1)),
    ("Upcoming matches", today + timedelta(days=1), today + timedelta(days=7)),
)

for label, date_from, date_to in date_ranges:
    response = requests.get(
        uri,
        headers=headers,
        params={"dateFrom": date_from.isoformat(), "dateTo": date_to.isoformat()},
    )
    response.raise_for_status()
    print(f"--- {label} ({date_from} to {date_to}) ---")
    for match in response.json()["matches"]:
        print(match)