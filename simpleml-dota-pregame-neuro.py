import datetime

import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical


# Fetch data from OpenDota API for all pro games played in the last 6 months
def fetch_pro_matches():
    url = "https://api.opendota.com/api/proMatches"
    response = requests.get(url)
    return response.json()


def filter_last_6_months(matches):
    six_months_ago = datetime.datetime.now() - datetime.timedelta(days=4*30)
    timestamp_six_months_ago = six_months_ago.timestamp()
    return [match for match in matches if match['start_time'] > timestamp_six_months_ago]


matches = fetch_pro_matches()
filtered_matches = filter_last_6_months(matches)

# Process and prepare the data for modeling


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

X = df.drop(['radiant_win', 'match_id'], axis=1)
y = df['radiant_win']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Preprocess the data
num_classes = len(np.unique(y_train))
y_train_one_hot = to_categorical(y_train, num_classes)
y_test_one_hot = to_categorical(y_test, num_classes)

input_dim = X_train.shape[1]

# Define the neural network model
model = Sequential()
model.add(Dense(64, input_dim=input_dim, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train_one_hot, epochs=10,
          batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
score = model.evaluate(X_test, y_test_one_hot, verbose=0)
print(f'Test loss: {score[0]}, Test accuracy: {score[1]}')
score = model.evaluate(X_test, y_test_one_hot, verbose=0)
print(f'Test loss: {score[0]}, Test accuracy: {score[1]}')
