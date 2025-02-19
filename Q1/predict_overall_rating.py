import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import time


def load_and_preprocess_data(file_path):
    # Load the dataset
    data_frame = pd.read_csv(file_path, low_memory=False)

    # Convert 'fifa_version' column to integer
    data_frame['fifa_version'] = data_frame['fifa_version'].astype(int)

    # Drop goalkeepers (because goalkeepers contain NaN values in some of the columns)
    if 'player_positions' in data_frame.columns:
        data_frame = data_frame[data_frame['player_positions'] != 'GK']

    # Select relevant columns
    relevant_columns = [
        "player_id", "fifa_version", "overall", "potential",
        "age", "height_cm", "weight_kg", "pace", "shooting", "passing",
        "dribbling", "defending", "physic", "weak_foot", "skill_moves",
        "power_stamina", "power_strength"
    ]

    # Creating a new DataFrame that filters only the specified columns
    data_frame_filtered = data_frame[relevant_columns]

    # Identify players who appear in at least 8 different FIFA versions (3 for features, 5 for targets)
    player_counts = data_frame_filtered["player_id"].value_counts()
    players_to_keep = player_counts[player_counts >= 8].index
    data_frame_filtered = data_frame_filtered[data_frame_filtered["player_id"].isin(players_to_keep)]

    # Sort data by player and FIFA version.
    data_frame_filtered = data_frame_filtered.sort_values(by=["player_id", "fifa_version"])

    print(f"Dataset size: {data_frame_filtered.shape}\n")

    return data_frame_filtered


def prepare_data(data_frame_filtered):
    # Create a dictionary to store player data
    player_dict = {}

    num_features = len(data_frame_filtered.columns) - 3  # Excluding 'player_id', 'fifa_version', 'potential'

    # Organizing data for each player
    for player_id, group in data_frame_filtered.groupby("player_id"):
        if len(group) < 8:
            continue

        # Extract first 3 versions as features
        first_3 = group.iloc[:3].drop(columns=["player_id", "fifa_version", "potential"])

        # Extract next 5 versions as targets (overall rating)
        next_5 = group.iloc[3:8]["overall"].values

        # Ensure correct shape before storing.
        if first_3.shape == (3, num_features) and next_5.shape == (5,):
            # Converts the (3, num_features) matrix into a 1D array
            player_dict[player_id] = (first_3.values.flatten(), next_5)

    # Convert dictionary to NumPy arrays
    X = np.array([data[0] for data in player_dict.values()])
    y = np.array([data[1] for data in player_dict.values()])

    # # Apply PCA for dimensionality reduction
    # pca = PCA(n_components=10)  # Reducing to 10 principal components
    # X = pca.fit_transform(X)
    # print(f"Explained variance by PCA components: {pca.explained_variance_ratio_.sum():.2f}")

    # Split the data into train and test sets (20% of the data for testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


def evaluate_model(model, model_name, X_train, X_test, y_train, y_test, results):
    # Train the model using the training data
    model.fit(X_train, y_train)

    # Predict the target values (overall ratings) for the test set
    y_predictions = model.predict(X_test)

    # Calculate performance metrics
    mae = mean_absolute_error(y_test, y_predictions)
    rmse = np.sqrt(mean_squared_error(y_test, y_predictions))

    results.append([model_name, mae, rmse])


# Main execution
if __name__ == "__main__":
    start_time = time.perf_counter()  # Start timing

    file_path = "male_players_2024-2015.csv"
    data_frame_filtered = load_and_preprocess_data(file_path)
    X_train, X_test, y_train, y_test = prepare_data(data_frame_filtered)

    results = []
    decision_tree_results = []
    knn_results = []

    evaluate_model(LinearRegression(), "Linear Regression", X_train, X_test, y_train, y_test, results)

    # Different hyperparameter values for Decision Tree and k-NN
    decision_tree_depths = [3, 5, 6, 7, 10]
    knn_neighbors_values = [3, 5, 6, 7, 8, 10]

    for max_depth in decision_tree_depths:
        model = DecisionTreeRegressor(max_depth=max_depth)
        model_name = f"Decision Tree (Depth={max_depth})"
        evaluate_model(model, model_name, X_train, X_test, y_train, y_test, decision_tree_results)

    for n_neighbors in knn_neighbors_values:
        model = KNeighborsRegressor(n_neighbors=n_neighbors)
        model_name = f"k-NN (k={n_neighbors})"
        evaluate_model(model, model_name, X_train, X_test, y_train, y_test, knn_results)

    # Print Linear Regression results
    results_df = pd.DataFrame(results, columns=["Model", "Mean Absolute Error", "Root Mean Squared Error"])
    print("\nLinear Regression Results:")
    print(results_df)

    # Print Decision Tree and k-NN results separately
    decision_tree_df = pd.DataFrame(decision_tree_results,
                                    columns=["Model", "Mean Absolute Error", "Root Mean Squared Error"])
    knn_df = pd.DataFrame(knn_results, columns=["Model", "Mean Absolute Error", "Root Mean Squared Error"])

    print("\nDecision Tree Results:")
    print(decision_tree_df)
    print("\nk-NN Results:")
    print(knn_df)

    end_time = time.perf_counter()
    print(f"Total runtime: {end_time - start_time:.2f} seconds")
