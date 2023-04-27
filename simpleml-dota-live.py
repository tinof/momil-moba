import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier


def get_csv_paths(folder_path):
    csv_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    return csv_files


def get_features_from_csv(csv_path):
    df = pd.read_csv(csv_path, nrows=0)
    return df.columns.tolist()


def load_data(folder_path):
    csv_paths = get_csv_paths(folder_path)

    all_features = []
    datasets = []
    for csv_path in csv_paths:
        features = get_features_from_csv(csv_path)
        all_features.extend(features)
        datasets.append(pd.read_csv(csv_path))

    selected_features = list(set(all_features))
    combined_dataset = pd.concat(
        [dataset[selected_features] for dataset in datasets], ignore_index=True)

    return combined_dataset, selected_features


def preprocess_data(data, selected_features, seasons):
    season_data = {season: data[data['season'] == season] for season in seasons}

    for season in seasons:
        season_data[season] = season_data[season][season_data[season]['num_players'] == 10]

    X = {season: season_data[season][selected_features] for season in seasons}
    y = {season: season_data[season]['target_column'].values for season in seasons}

    return X, y


def group_into_teams(X, team_size=5):
    return [X[i:i + team_size] for i in range(0, len(X), team_size)]


def summary_statistics(bag):
    min_values = np.min(bag, axis=0)
    max_values = np.max(bag, axis=0)
    mean_values = np.mean(bag, axis=0)
    std_values = np.std(bag, axis=0)

    return np.concatenate((min_values, max_values, mean_values, std_values))


def season_based_analysis(X, y, seasons):
    dt_classifier_all_seasons = DecisionTreeClassifier(
        criterion='entropy', random_state=42)
    dt_classifier_all_seasons.fit(np.vstack([X[season] for season in seasons]),
                                  np.hstack([y[season] for season in seasons]))

    for season in seasons:
        X_train, X_test, y_train, y_test = train_test_split(
            X[season], y[season], test_size=0.2, random_state=42)

        dt_classifier = DecisionTreeClassifier(
            criterion='entropy', random_state=42)
        dt_classifier.fit(X_train, y_train)

        cv_scores = cross_val_score(
            dt_classifier, X[season], y[season], cv=10, scoring='accuracy')
        print(
            f"Season {season} - Accuracy: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")

        y_pred = dt_classifier.predict(X_test)
        f_measure = f1_score(y_test, y_pred, average='weighted')
        print(f"Season {season} - F-measure: {f_measure:.2f}")

    return dt_classifier_all_seasons


def predict_match_outcome(real_time_data, classifier, selected_features):
    assert real_time_data.shape == (10, len(selected_features))

    team1_data = real_time_data[:5]
    team2_data = real_time_data[5:]

    team1_summary = summary_statistics(team1_data)
    team2_summary = summary_statistics(team2_data)

    input_data = np.vstack((team1_summary, team2_summary))

    predictions = classifier.predict(input_data)

    return {
        'team1': 'win' if predictions[0] == 1 else 'loss',
        'team2': 'win' if predictions[1] == 1 else 'loss'
    }


def main():
    folder_path = 'path/to/your/folder'
    seasons = [3, 4, 5, 6, 7]

    combined_data, selected_features = load_data(folder_path)
    X_raw, y = preprocess_data(combined_data, selected_features, seasons)
    X_summary = {season: np.array([summary_statistics(bag)
                                   for bag in group_into_teams(X_raw[season])]) for season in seasons}

    dt_classifier_all_seasons = season_based_analysis(X_summary, y, seasons)

    real_time_data = np.array([
        [10, 8, 3, 15000, 3000, 20000],
        [4, 12, 6, 12000, 2500, 18000],
        [7, 9, 5, 13000, 2700, 19000],
        [5, 11, 4, 14000, 2600, 21000],
        [8, 10, 2, 16000, 2800, 22000],

        [6, 7, 4, 15500, 2900, 20500],
        [3, 13, 7, 12500, 2400, 18500],
        [9, 8, 6, 13500, 2600, 19500],
        [4, 10, 5, 14500, 2500, 21500],
        [7, 11, 3, 16500, 2700, 22500]
    ])

    predicted_outcome = predict_match_outcome(
        real_time_data, dt_classifier_all_seasons, selected_features)
    print(predicted_outcome)


if __name__ == '__main__':
    main()
