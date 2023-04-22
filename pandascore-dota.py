from datetime import datetime, timedelta

import requests

url = "https://api.pandascore.co/dota2/tournaments/running?sort=&page=1&per_page=50"

headers = {
    "accept": "application/json",
    "Authorization": "gkuzT3vQrwVmxr6GLwZrDGjySh6t514Jgc-YgTnuuvA2RAW0IU4"
}

response = requests.get(url, headers=headers)

if response.status_code == 200:
    ongoing_tournaments = response.json()
    
    for tournament in ongoing_tournaments:
        print(f"Tournament: {tournament['name']}")
        
        matches_url = f"https://api.pandascore.co/dota2/matches/upcoming?filter[tournament_id]={tournament['id']}&range[begin_at]=,{datetime.utcnow().isoformat()}Z&range[end_at]={(datetime.utcnow() + timedelta(days=1)).isoformat()}Z"
        matches_response = requests.get(matches_url, headers=headers)
        
        if matches_response.status_code == 200:
            upcoming_matches = matches_response.json()
            
            if not upcoming_matches:
                print("No upcoming matches today.")
            else:
                for match in upcoming_matches:
                    print(f"Match: {match['name']} - Scheduled at: {match['begin_at']}")
        else:
            print(f"Request failed with status code {matches_response.status_code}")
        
        print("\n")
else:
    print(f"Request failed with status code {response.status_code}")
