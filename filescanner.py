import os
import pandas as pd

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
    df = pd.read_csv(csv_path, nrows=0)  # Read only the header (nrows=0)
    return df.columns.tolist()

def main():
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

    print("CSV Paths:")
    print(csv_paths)
    print("\nSelected Features:")
    print(selected_features)

    # Combine datasets while keeping only the selected features
    combined_dataset = pd.concat([dataset.reindex(selected_features, axis=1) for dataset in datasets], ignore_index=True)

    print("\nCombined Dataset:")
    print(combined_dataset.head())

if __name__ == '__main__':
    main()
