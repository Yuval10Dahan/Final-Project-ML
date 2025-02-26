import pandas as pd
import numpy as np
import time
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve


def load_data(file_path):
    # Load dataset
    df = pd.read_csv(file_path, low_memory=False)

    # Required columns
    columns_needed = ['short_name', 'long_name', 'power_stamina', 'power_strength',
                      'movement_agility', 'movement_acceleration', 'movement_sprint_speed',
                      'player_traits', 'fifa_version', 'age', 'league_name', 'player_id', 'mentality_aggression']

    df = df[columns_needed].dropna()

    print("Total dataset size after dropping NaN values:", len(df))

    # Create binary injury-free column
    df['injury_free'] = df['player_traits'].astype(str).apply(
        lambda x: 1 if "Injury Free" in x or "Injury Prone" not in x else 0
    )
    df.drop(columns=['player_traits'], inplace=True)

    # Filter FIFA 2020 players meeting criteria
    df_2020 = df[(df['fifa_version'] == 20) &
                 (df['age'] >= 21) &
                 (df['power_stamina'] >= 60) &
                 (df['injury_free'] == 1)].copy()

    # Get FIFA 2024 players in top leagues
    target_players = df[(df['fifa_version'] == 24) &
                        (df['league_name'].isin(['La Liga', 'Premier League', 'Bundesliga', 'Ligue 1', 'Serie A']))]

    target_player_ids = set(target_players['player_id'])
    print("Total players in high-intensity leagues (2024):", len(target_player_ids))

    # Assign target labels using .loc to avoid SettingWithCopyWarning
    df_2020.loc[:, 'target'] = df_2020['player_id'].apply(lambda x: 1 if x in target_player_ids else 0)

    print("Players labeled as moving (target=1):", df_2020['target'].sum())

    return df_2020


def train_and_evaluate_models(df):
    # Select features and target
    feature_cols = ['power_stamina', 'power_strength', 'movement_agility',
                    'movement_acceleration', 'movement_sprint_speed', 'injury_free', 'mentality_aggression']

    X = df[feature_cols]
    y = df['target']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Handle class imbalance with SMOTE
    smote = SMOTE(sampling_strategy=0.6, random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # Standardization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models = {
        'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_split=5,
                                                min_samples_leaf=3, random_state=42, class_weight='balanced'),
        'Support Vector Machine': SVC(kernel='rbf', probability=True, class_weight='balanced', C=1, gamma='scale')
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred_probs = model.predict_proba(X_test)[:, 1]

        # Determine optimal threshold using precision-recall tradeoff
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_probs)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-6)
        best_threshold = thresholds[np.argmax(f1_scores)]
        y_pred = (y_pred_probs > best_threshold).astype(int)

        metrics = {
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1 Score': f1_score(y_test, y_pred, zero_division=0)
        }
        results.append(metrics)

    return pd.DataFrame(results)


if __name__ == "__main__":
    start_time = time.perf_counter()  # Start timing

    file_path = "male_players_2024-2015.csv"

    # Load and process data
    df = load_data(file_path)

    # Train and evaluate models
    results_df = train_and_evaluate_models(df)
    print(results_df)

    end_time = time.perf_counter()
    print(f"Total runtime: {end_time - start_time:.2f} seconds")