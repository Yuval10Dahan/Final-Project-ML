import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import time


def load_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path, low_memory=False)

    return df


def filter_data(df):
    # Include only players in FIFA 2024 and from brazil or germany
    filtered_df = df[(df["fifa_version"] == 24) & (df["nationality_id"].isin([21, 54]))]

    return filtered_df


def preprocess_data(df, features):
    # Drop rows that have missing values
    df = df.dropna(subset=features)

    # Select only the features specified in the list
    X = df[features]

    # Extract the target variable - "nationality_id"
    y = df["nationality_id"]

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

    # Comparing predictions with actual labels
    accuracy = accuracy_score(y_test, y_predictions)
    report = classification_report(y_test, y_predictions)

    return accuracy, report


if __name__ == "__main__":
    start_time = time.perf_counter()  # Start timing

    file_path = "male_players_2024-2015.csv"

    features = ['height_cm', 'defending', 'attacking_finishing', 'skill_dribbling', 'movement_balance',
                "weak_foot", "passing", "dribbling", "physic", "attacking_short_passing", "skill_ball_control",
                "movement_sprint_speed", "movement_agility", "movement_reactions"]

    df = load_data(file_path)
    df_filtered = filter_data(df)
    X, y = preprocess_data(df_filtered, features)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Initialize different models
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Support Vector Machine": SVC(kernel='linear', random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
    }

    # Train and evaluate each model
    results = {}
    for model_name, model in models.items():
        trained_model = train_model(model, X_train, y_train)
        accuracy, report = evaluate_model(trained_model, X_test, y_test)
        results[model_name] = {"accuracy": accuracy, "report": report}

    # Print final comparison of all models
    for model_name, result in results.items():
        print(f"\n{model_name} - Accuracy: {result['accuracy']:.4f}")
        print("Classification Report:\n", result["report"])

    end_time = time.perf_counter()
    print(f"\nTotal runtime: {end_time - start_time:.2f} seconds")
