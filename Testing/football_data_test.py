import requests
import json
import os
import dotenv

dotenv.load_dotenv()


uri = 'https://api.football-data.org/v4/matches'
headers = { 'X-Auth-Token': os.getenv("FOOTBALL_DATA_KEY")}

response = requests.get(uri, headers=headers)
for match in response.json()['matches']:
    print(match)