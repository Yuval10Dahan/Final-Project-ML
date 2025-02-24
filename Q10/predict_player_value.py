import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
import time


def load_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path, low_memory=False)

    return df


def filter_data(df, fifa_version):
    # Include only players from the specific fifa_version and from league level of 1
    filtered_df = df[(df['fifa_version'] == fifa_version) & (df['league_level'] == 1)]

    return filtered_df


def preprocess_data(df, features):
    # Drop rows that have missing values
    df = df.dropna(subset=features + ["value_eur"])

    # Select only the features specified in the list
    X = df[features]

    # Extract the target variable - "value_eur"
    y = df["value_eur"]

    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    # Split the data into train and test sets (20% of the data for testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test


def train_model(model, X_train, y_train):
    # Train the model using the training data
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    # Predict the target values for the test set on the model
    y_predictions = model.predict(X_test)

    # Compute the coefficient of determination score to evaluate model performance
    r2 = r2_score(y_test, y_predictions)

    return r2





if __name__ == "__main__":
    start_time = time.perf_counter()  # Start timing

    file_path = "male_players_2024-2015.csv"

    features = ['overall', 'potential', 'age', 'club_team_id', 'league_id',
                'club_contract_valid_until_year', 'nationality_id', 'release_clause_eur']

    df = load_data(file_path)

    # Initialize different models
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "SVM": SVR(kernel='rbf'),
        "AdaBoost": AdaBoostRegressor(n_estimators=100, random_state=42),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "KNN": KNeighborsRegressor(n_neighbors=5)
    }

    results = {}
    model_avg_scores = {model_name: [] for model_name in models.keys()}

    # Train and evaluate each model
    for fifa_version in range(18, 25):
        df_filtered = filter_data(df, fifa_version)

        print(f"FIFA Version: {fifa_version}, Number of Samples: {len(df_filtered)}")

        if df_filtered.empty:
            continue

        X, y = preprocess_data(df_filtered, features)
        X_train, X_test, y_train, y_test = split_data(X, y)

        results[fifa_version] = {}
        for model_name, model in models.items():
            trained_model = train_model(model, X_train, y_train)
            r2 = evaluate_model(trained_model, X_test, y_test)
            results[fifa_version][model_name] = r2
            model_avg_scores[model_name].append(r2)

    print("\nFinal Results:")
    for fifa_version, model_results in sorted(results.items()):
        print(f"\nFIFA Version: {fifa_version}")
        for model_name, r2 in model_results.items():
            print(f"{model_name}: R² Score = {r2:.4f}")

    print("\nAverage R² Scores Across All FIFA Versions:")
    for model_name, scores in model_avg_scores.items():
        if scores:
            avg_score = sum(scores) / len(scores)
            print(f"{model_name}: Average R² Score = {avg_score:.4f}")

    end_time = time.perf_counter()
    print(f"\nTotal runtime: {end_time - start_time:.2f} seconds")