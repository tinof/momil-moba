import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load the dataset from Kaggle (replace with the actual file paths)
matches = pd.read_csv('path/to/matches.csv')
players = pd.read_csv('path/to/players.csv')
champions = pd.read_csv('path/to/champions.csv')
items_purchased = pd.read_csv('path/to/items_purchased.csv')
team_bans = pd.read_csv('path/to/team_bans.csv')
statistics = pd.read_csv('path/to/statistics.csv')

# Combine the tables using join operation
combined_data = matches.merge(players).merge(champions).merge(
    items_purchased).merge(team_bans).merge(statistics)

# Extract 40 features including kills, assists, deaths, damages, rewards, and gold earned by each player
selected_features = ['kills', 'assists',
                     'deaths', 'damages', 'rewards', 'golds']
X = combined_data[selected_features]

# Preprocess the data: Divide the dataset into subsets for season-based analysis
seasons = [3, 4, 5, 6, 7]
season_data = {
    season: combined_data[combined_data['season'] == season] for season in seasons}

# Remove match records with fewer than 10 players
for season in seasons:
    season_data[season] = season_data[season][season_data[season]
                                              ['num_players'] == 10]

# Transform single-instance data into multi-instance data


def group_into_teams(X, team_size=5):
    return [X[i:i + team_size] for i in range(0, len(X), team_size)]


bags = {season: group_into_teams(season_data[season]) for season in seasons}

# Define a function to extract summary statistics from each bag


def summary_statistics(bag):
    min_values = np.min(bag, axis=0)
    max_values = np.max(bag, axis=0)
    mean_values = np.mean(bag, axis=0)
    std_values = np.std(bag, axis=0)

    return np.concatenate((min_values, max_values, mean_values, std_values))


# Apply the summary_statistics function to your dataset
X_summary = {season: np.array([summary_statistics(bag)
                              for bag in bags[season]]) for season in seasons}

# Prepare the target variable (replace 'target_column' with the actual target column name)
y = {season: season_data[season]['target_column'].values for season in seasons}

# Train a classifier on the entire dataset
X_all_seasons = np.vstack([X_summary[season] for season in seasons])
y_all_seasons = np.hstack([y[season] for season in seasons])

dt_classifier_all_seasons = DecisionTreeClassifier(
    criterion='entropy', random_state=42)
dt_classifier_all_seasons.fit(X_all_seasons, y_all_seasons)

# Perform season-based analysis
for season in seasons:
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_summary[season], y[season], test_size=0.2, random_state=42)

    # Build predictive models using SimpleMI: Transform a bag into a vector form and classify it using a standard learner
    dt_classifier = DecisionTreeClassifier(
        criterion='entropy', random_state=42)
    dt_classifier.fit(X_train, y_train)

    # Evaluate the model using 10-fold cross-validation
    cv_scores = cross_val_score(
        dt_classifier, X_summary[season], y[season], cv=10, scoring='accuracy')
    print(
        f"Season {season} - Accuracy: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")

    # Calculate F-measure
    y_pred = dt_classifier.predict(X_test)
    f_measure = f1_score(y_test, y_pred, average='weighted')
    print(f"Season {season} - F-measure: {f_measure:.2f}")


def predict_match_outcome(real_time_data, classifier):
    # Ensure real_time_data has the shape (10, num_features)
    assert real_time_data.shape == (10, len(selected_features))

    # Split the data into two bags (one for each team)
    team1_data = real_time_data[:5]
    team2_data = real_time_data[5:]

    # Compute summary statistics for each bag
    team1_summary = summary_statistics(team1_data)
    team2_summary = summary_statistics(team2_data)

    # Combine the summaries into a single input array
    input_data = np.vstack((team1_summary, team2_summary))

    # Predict the match outcome using the trained classifier
    predictions = classifier.predict(input_data)

    # Return the predicted outcome for each team
    return {
        'team1': 'win' if predictions[0] == 1 else 'loss',
        'team2': 'win' if predictions[1] == 1 else 'loss'
    }


real_time_data = np.array([
    # Player statistics for Team 1 (5 players)
    [10, 8, 3, 15000, 3000, 20000],
    [4, 12, 6, 12000, 2500, 18000],
    [7, 9, 5, 13000, 2700, 19000],
    [5, 11, 4, 14000, 2600, 21000],
    [8, 10, 2, 16000, 2800, 22000],

    # Player statistics for Team 2 (5 players)
    [6, 7, 4, 15500, 2900, 20500],
    [3, 13, 7, 12500, 2400, 18500],
    [9, 8, 6, 13500, 2600, 19500],
    [4, 10, 5, 14500, 2500, 21500],
    [7, 11, 3, 16500, 2700, 22500]
])

# Use the classifier trained on the entire dataset for predicting match outcomes
predicted_outcome = predict_match_outcome(
    real_time_data, dt_classifier_all_seasons)
print(predicted_outcome)
