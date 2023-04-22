import datetime

import pandas as pd
import requests
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


# Step 1: Fetch data from OpenDota API for all pro games played in the last 6 months
def fetch_pro_matches():
    url = "https://api.opendota.com/api/proMatches"
    response = requests.get(url)
    return response.json()

def filter_last_6_months(matches):
    six_months_ago = datetime.datetime.now() - datetime.timedelta(days=6*30)
    timestamp_six_months_ago = six_months_ago.timestamp()
    return [match for match in matches if match['start_time'] > timestamp_six_months_ago]

matches = fetch_pro_matches()
filtered_matches = filter_last_6_months(matches)

# Step 2: Process and prepare the data for modeling
def process_match_data(matches):
    processed_data = []
    for match in matches:
        processed_data.append({
            'match_id': match['match_id'],
            'duration': match['duration'],
            'radiant_win': match['radiant_win'],
            'radiant_team_id': match['radiant_team_id'],
            'dire_team_id': match['dire_team_id']
        })
    return processed_data

processed_matches = process_match_data(filtered_matches)
df = pd.DataFrame(processed_matches)

# Prepare data for training
X = df.drop(['radiant_win', 'match_id'], axis=1)
y = df['radiant_win']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train an XGBoost model to predict the outcome of future games
model = XGBClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
