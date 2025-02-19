import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import time


def load_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path, low_memory=False)
    return df


def filter_gk(df):
    # Drop goalkeepers (because goalkeepers contain NaN values in some of the columns)
    return df[df['player_positions'] != 'GK']


def filter_young_players(df):
    # Select players younger than 20 and appearing in FIFA versions up to 21
    df_young = df[(df['age'] < 20) & (df['fifa_version'] <= 21)].copy()

    # Each player appears only once, at their earliest FIFA entry
    df_young = df_young.loc[df_young.groupby('player_id')['fifa_version'].idxmin()]
    print(f"Filtered young players (FIFA 2021 and below, first appearance only): {df_young.shape[0]} rows")

    return df_young


def define_target_variable(df, full_df):
    df = df.copy()

    # Store for each young player his actual rating from FIFA 2024
    future_overall_map = full_df[full_df['fifa_version'] == 24].set_index('player_id')['overall']
    df['future_overall_FIFA2024'] = df['player_id'].map(future_overall_map)

    # Remove players that is not in FIFA 2024
    df = df[df['future_overall_FIFA2024'].notna()]
    print(f"Remaining rows after defining target: {df.shape[0]}")

    return df


def split_and_scale_data(df, features, target):
    # Remove rows with missing values in feature columns
    df = df.dropna(subset=features)

    # Separate the feature variables (X) and the target variable (y)
    X = df[features]
    y = df[target]

    # Split the data into train and test sets (20% of the data for testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize feature values
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


def apply_pca(X_train, X_test, n_components=10):
    pca = PCA(n_components=n_components)

    # Fit PCA on the training data and transform it to the new lower-dimensional space
    X_train_pca = pca.fit_transform(X_train)

    # Apply the trained PCA transformation to the test data
    X_test_pca = pca.transform(X_test)

    return X_train_pca, X_test_pca


def evaluate_model(model, model_name, X_train, X_test, y_train, y_test, results):
    # Train the model using the training data
    model.fit(X_train, y_train)

    # Predict the target values for the test set
    y_predictions = model.predict(X_test)

    # measures the average absolute difference between predicted and actual values
    mae = mean_absolute_error(y_test, y_predictions)

    # penalizes larger errors
    rmse = np.sqrt(mean_squared_error(y_test, y_predictions))

    results.append([model_name, mae, rmse])


if __name__ == "__main__":
    start_time = time.perf_counter()  # Start timing

    file_path = "male_players_2024-2015.csv"
    data_frame = load_data(file_path)
    data_frame = filter_gk(data_frame)
    young_players_data_frame = filter_young_players(data_frame)

    features = [
        'potential', 'skill_ball_control', 'attacking_short_passing', 'attacking_crossing',
        'power_strength', 'power_stamina', 'defending_standing_tackle', 'defending_sliding_tackle',
        'value_eur', 'wage_eur'
    ]

    target = 'future_overall_FIFA2024'

    young_players_data_frame = define_target_variable(young_players_data_frame, data_frame)
    X_train, X_test, y_train, y_test = split_and_scale_data(young_players_data_frame, features, target)

    # Apply PCA
    X_train_pca, X_test_pca = apply_pca(X_train, X_test)

    results = []

    # Train and evaluate models
    evaluate_model(DecisionTreeRegressor(), "Decision Tree", X_train, X_test, y_train, y_test, results)
    evaluate_model(SVC(C=0.1, kernel='linear'), "Support Vector Machine", X_train, X_test, y_train, y_test, results)
    evaluate_model(AdaBoostClassifier(learning_rate=0.01, n_estimators=50), "AdaBoost", X_train, X_test, y_train, y_test, results)
    evaluate_model(LogisticRegression(C=0.1), "Logistic Regression", X_train, X_test, y_train, y_test, results)
    evaluate_model(KNeighborsClassifier(n_neighbors=3, weights='uniform'), "Nearest Neighbor", X_train, X_test, y_train, y_test, results)
    evaluate_model(DecisionTreeRegressor(), "Decision Tree (PCA)", X_train_pca, X_test_pca, y_train, y_test, results)

    # Print results
    results_df = pd.DataFrame(results, columns=["Model", "Mean Absolute Error", "Root Mean Squared Error"])
    print("\nModel Evaluation Results:")
    print(results_df)

    end_time = time.perf_counter()
    print(f"Total runtime: {end_time - start_time:.2f} seconds")
