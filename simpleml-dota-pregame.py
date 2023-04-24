import os
import pandas as pd
import xgboost as xgb
import requests
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Function to recursively scan a folder for CSV files and return their paths
def get_csv_paths(folder_path):
    csv_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    return csv_files

# Function to read a CSV file and return its column names (features)
def get_features_from_csv(csv_path):
    df = pd.read_csv(csv_path, usecols=lambda x: x not in ['radiant_win'])
    return df.columns.tolist()


folder_path = 'data'

# Get the paths of all CSV files in the folder and its subfolders
csv_paths = get_csv_paths(folder_path)

# Read the CSV files and extract their features
all_features = []
datasets = []
for csv_path in csv_paths:
    features = get_features_from_csv(csv_path)
    all_features.extend(features)
    datasets.append(pd.read_csv(csv_path))

# Remove duplicates from the list of features
selected_features = list(set(all_features))

# Combine datasets while keeping only the selected features
combined_dataset = pd.concat([dataset.reindex(selected_features, axis=1) for dataset in datasets], ignore_index=True)


# Select only the features that are accessible from the Opendota API
accessible_features = set(combined_dataset.columns) - {'radiant_win'}
selected_features = ['radiant_win'] + list(accessible_features)

X = combined_dataset[selected_features].drop('radiant_win', axis=1)
y = combined_dataset['radiant_win']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(use_label_encoder=False, objective="binary:logistic", eval_metric="logloss")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# (4) Fetch the next 5 Pro games using Opendota API
API_BASE_URL = "https://api.opendota.com/api"
UPCOMING_PRO_MATCHES_ENDPOINT = f"{API_BASE_URL}/proMatches"

response = requests.get(UPCOMING_PRO_MATCHES_ENDPOINT)
if response.status_code == 200:
    upcoming_pro_matches = response.json()[:5]
else:
    print("Error fetching upcoming pro matches.")
    exit(1)

# (5) Let the user choose which game to predict
print("\nUpcoming Pro matches:")
for i, match in enumerate(upcoming_pro_matches, start=1):
    print(f"{i}. Match ID: {match['match_id']}")
    
if len(upcoming_pro_matches) < 5:
    print("There are less than 5 upcoming pro matches available.")
    exit(1)

choice = int(input("\nChoose a match to predict (1-5): ")) - 1
if choice not in range(5):
    print("Invalid choice. Please enter a number between 1 and 5.")
    exit(1)
selected_match = upcoming_pro_matches[choice]

# Preprocess the selected match data
# Replace this with your own preprocessing logic
selected_features.remove('radiant_win')
X_match = pd.DataFrame([selected_match])[selected_features]

# (6) Print the prediction for that game
prediction = model.predict(X_match)
if prediction[0]:
    print("\nPrediction: Radiant will win.")
else:
    print("\nPrediction: Dire will win.")
