import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from skmine.propositionalization import Propositionalizer # scikit-mine, check import
from sklearn.linear_model import LogisticRegression


# Dota 2 Dota 2 Pro League Matches 2023
# https://www.kaggle.com/datasets/bwandowando/dota-2-pro-league-matches-2023

data = pd.read_csv('dota_matches.csv')

# Combine the tables using the join operation
data = pd.merge(data['players'], data['matches'], on='match_id')

# Extract relevant features
features = ['kills', 'deaths', 'assists', 'gold', 'xp', 'hero_damage', 'tower_damage', 'hero_healing', 'level']
team_features = ['radiant_win', 'radiant_tower_kills', 'dire_tower_kills']
data = data[features + team_features]

# Remove match records with fewer than 10 players
data = data.groupby('match_id').filter(lambda x: len(x) == 10)

# Split the dataset into training and testing sets
train_data, test_data, train_target, test_target = train_test_split(data[features], data['radiant_win'], test_size=0.2, random_state=42)

# Create a propositionalizer and fit it on the training data
prop = Propositionalizer()
prop.fit(train_data)

# Propositionalize the training and testing data
train_prop = prop.transform(train_data)
test_prop = prop.transform(test_data)

# Compute summary statistics
train_summary = train_prop.groupby(train_target).agg(['mean', 'std'])
test_summary = test_prop.groupby(test_target).agg(['mean', 'std'])

# Compute SimpleMI scores
simplemi_scores = {}
for col in train_prop.columns:
    mean_diff = abs(train_summary.loc[False, col]['mean'] - train_summary.loc[True, col]['mean'])
    std_avg = (train_summary.loc[False, col]['std'] + train_summary.loc[True, col]['std']) / 2
    simplemi_scores[col] = mean_diff / std_avg

# Sort the features by SimpleMI score
sorted_features = sorted(simplemi_scores.items(), key=lambda x: x[1], reverse=True)

# Select the top 10 features
selected_features = [f[0] for f in sorted_features[:10]]

# Train a logistic regression model on the selected features
lr = LogisticRegression()
lr.fit(train_prop[selected_features], train_target)

# Evaluate the model on the testing set
accuracy = lr.score(test_prop[selected_features], test_target)
print("Accuracy: {:.2f}%".format(accuracy * 100))
