import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# Dota 2 Dota 2 Pro League Matches 2023
# https://www.kaggle.com/datasets/bwandowando/dota-2-pro-league-matches-2023

data = pd.read_csv('dota_matches.csv')

# Next, let's preprocess the data. We'll start by dropping any columns that aren't relevant to our prediction, such as the ID of the match or the names of the teams.
data = data.drop(['match_id', 'team1', 'team2'], axis=1)

# We'll also need to convert the categorical variables (such as the hero picks) into numerical values. For this example, we'll use one-hot encoding.
data = pd.get_dummies(data, columns=['hero1', 'hero2', 'hero3', 'hero4', 'hero5', 'hero6', 'hero7', 'hero8', 'hero9',
                                     'hero10'])

# Next, we'll split the data into training and testing sets.
train_data = data[data['date'] < '2022-01-01']
test_data = data[data['date'] >= '2022-01-01']

# We'll use the training data to train our LSTM model. We'll start by scaling the data to a range between 0 and 1.
scaler = MinMaxScaler()
train_data = scaler.fit_transform(train_data)

# Next, we'll create sequences of data to feed into our LSTM model. For this example, let's use a sequence length of 30 matches.
seq_length = 30
X_train = []
y_train = []
for i in range(seq_length, len(train_data)):
    X_train.append(train_data[i - seq_length:i])
    y_train.append(train_data[i, -1])
X_train, y_train = np.array(X_train), np.array(y_train)

# Next, let's build our LSTM model. We'll use a simple model with one LSTM layer and one dense layer.
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# We'll train our model on the training data.
model.fit(X_train, y_train, epochs=50, batch_size=32)

## Work in progress
# Finally, we can use our trained model to predict the outcome of new Dota tournament matches.
# Let's use the testing data for this.
test_data = scaler.transform(test_data)
X_test = []
y_test = []
for i in range(seq_length, len(test_data)):
    X_test.append(test_data[i - seq_length:i])
    y_test.append(test_data[i, -1])
X_test, y_test = np.array(X_test), np.array(y_test)

test_data = test_data.drop(['date'], axis=1)
test_data = pd.get_dummies(test_data, columns=[
    'hero1', 'hero2', 'hero3', 'hero4', 'hero5', 'hero6', 'hero7', 'hero8', 'hero9', 'hero10'])
