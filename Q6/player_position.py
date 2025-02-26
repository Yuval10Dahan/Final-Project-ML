import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
import time


def load_and_preprocess_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path, low_memory=False)

    # Filter only rows where FIFA version is 24
    df = df[df['fifa_version'] == 24]

    # Features to exclude as they are not relevant for classification
    excluded_features = [
        'player_id', 'player_url', 'fifa_version', 'fifa_update', 'age', 'club_team_id', 'league_id', 'league_level',
        'club_jersey_number', 'nation_team_id', 'nation_jersey_number',
        'international_reputation', 'goalkeeping_diving', 'goalkeeping_handling',
        'goalkeeping_kicking', 'goalkeeping_positioning', 'goalkeeping_reflexes',
        'goalkeeping_speed', 'ls', 'st', 'rs', 'lw', 'lf', 'cf', 'rf', 'rw', 'lam',
        'cam', 'ram', 'lm', 'lcm', 'cm', 'rcm', 'rm', 'lwb', 'ldm', 'cdm', 'rdm',
        'rwb', 'lb', 'lcb', 'cb', 'rcb', 'rb', 'gk'
    ]
    df = df.drop(columns=excluded_features, errors='ignore')

    # Convert player positions into a list
    df['player_positions'] = df['player_positions'].apply(lambda x: x.split(', '))

    # Encode multiple positions into binary labels
    mlb = MultiLabelBinarizer()

    # One-hot encode positions
    positions_encoded = mlb.fit_transform(df['player_positions'])

    # Extract position labels
    position_labels = mlb.classes_

    # Convert to DataFrame
    df_positions = pd.DataFrame(positions_encoded, columns=position_labels)

    # Filter only the required positions
    selected_positions = ['CB', 'CM', 'ST']
    df_positions = df_positions[selected_positions]

    # Select numerical features only
    X = df.select_dtypes(include=['float64', 'int64'])

    # Fill missing values with median
    X = X.fillna(X.median())

    return X, df_positions, selected_positions


def train_and_evaluate_models(X, y, position_labels):
    # Normalize feature values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit PCA on the scaled data and transform it to the new lower-dimensional space
    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_scaled)

    # Split the data into train and test sets (20% of the data for testing)
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42,
                                                        stratify=y.sum(axis=1))

    models = {
        'SVM': MultiOutputClassifier(SVC(kernel='rbf', class_weight='balanced', C=3.5, gamma='auto')),
        'Decision Tree': MultiOutputClassifier(DecisionTreeClassifier(min_samples_leaf=10, max_depth=8)),
        'Random Forest': MultiOutputClassifier(
            RandomForestClassifier(n_estimators=500, max_depth=14, min_samples_leaf=3, random_state=42)),
        'Logistic Regression': MultiOutputClassifier(LogisticRegression(max_iter=2000, class_weight='balanced'))
    }

    results = {}

    # Train and evaluate each model
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Computes accuracy per label - if a model predicts
        # some correct labels for a sample, it still gets partial credit.
        acc = (y_pred == y_test.to_numpy()).mean()
        results[name] = acc
        print(f"{name} Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred, target_names=position_labels, zero_division=0))

    print(pd.DataFrame(results, index=['Accuracy']))


if __name__ == "__main__":
    start_time = time.perf_counter()  # Start timing

    file_path = 'male_players_2024-2015.csv'

    X, y, position_labels = load_and_preprocess_data(file_path)

    train_and_evaluate_models(X, y, position_labels)

    end_time = time.perf_counter()
    print(f"Total runtime: {end_time - start_time:.2f} seconds")
