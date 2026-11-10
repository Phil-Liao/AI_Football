import datetime
import requests
import dotenv
import os

dotenv.load_dotenv()

yesterday = (datetime.date.today() - datetime.timedelta(days=2)).isoformat()
# Premier League is league_id 15
# 2. Base URL includes the api/v1 prefix
base_url = "https://footballdata.io/api/v1"
endpoint = f"/matches/date/{yesterday}"
url = base_url + endpoint

headers = {'Authorization': "Bearer " + os.getenv("FOOTBALLDATAIO_KEY")}
print(os.getenv("FOOTBALLDATAIO_KEY"))

try:
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        payload = response.json()
        
        # 2. Extract match array using the verified payload structure: payload['data']['matches']
        data_obj = payload.get("data", {})
        matches = data_obj.get("matches", []) if isinstance(data_obj, dict) else []

        print(f"--- EPL Scores for Yesterday ({yesterday}) ---")
        found_any = False
        
        # 3. Iterate over matches
        print(len(matches))
        for match in matches:
            print(match)
except requests.exceptions.RequestException as e:
    print(f"Network error occurred: {e}")