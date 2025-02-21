import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time


def load_data(file_path):
    # Load dataset
    data_frame = pd.read_csv(file_path, low_memory=False)

    columns_needed = ['short_name', 'long_name', 'power_stamina', 'power_strength',
                      'movement_agility', 'movement_acceleration', 'movement_sprint_speed',
                      'player_traits', 'fifa_version', 'age', 'league_name', 'player_id']

    # Create binary injury-free column
    data_frame['injury_free'] = data_frame['player_traits'].astype(str).apply(
        lambda x: 1 if "Injury Free" in x or "Injury Prone" not in x else 0
    )
    data_frame.drop(columns=['player_traits'], inplace=True)

    # Filter FIFA 2020 players meeting criteria
    df_2020 = data_frame[(data_frame['fifa_version'] == 20) &
                         (data_frame['age'] >= 21) &
                         (data_frame['power_stamina'] >= 60) &
                         (data_frame['injury_free'] == 1)].copy()

    # Get FIFA 2024 players in top leagues
    target_players = data_frame[(data_frame['fifa_version'] == 24) &
                                (data_frame['league_name'].isin(['La Liga', 'Premier League',
                                                                 'Bundesliga', 'Ligue 1', 'Serie A']))]

    target_player_ids = set(target_players['player_id'])
    print("Total players in high-intensity leagues (2024):", len(target_player_ids))

    # Assign target labels using .loc to avoid SettingWithCopyWarning
    df_2020.loc[:, 'target'] = df_2020['player_id'].apply(lambda x: 1 if x in target_player_ids else 0)

    print("Players labeled as moving (target=1):", df_2020['target'].sum())

    # Print names of players who moved with detailed information
    moving_players = df_2020[df_2020['target'] == 1][['player_id', 'short_name', 'power_stamina', 'age', 'league_name']]
    moving_players = moving_players.merge(
        target_players[['player_id', 'league_name', 'power_stamina', 'age']], on='player_id', how='left',
        suffixes=('_fifa_20', '_fifa_24')
    )
    print("Players who moved to top leagues in 2024:")
    # print(moving_players.rename(columns={
    #     'short_name': 'short_name',
    #     'power_stamina_fifa_20': 'stamina_in_fifa_20',
    #     'age_fifa_20': 'age_in_fifa_20',
    #     'league_name_fifa_20': 'league_name_in_fifa_20',
    #     'league_name_fifa_24': 'league_name_in_fifa_24',
    #     'power_stamina_fifa_24': 'stamina_in_fifa_24',
    #     'age_fifa_24': 'age_in_fifa_24'
    # }).drop(columns=['player_id']).to_string(index=False))  # Drop player_id for cleaner output

    return df_2020


def train_and_evaluate_models(df):
    # Select features and target
    feature_cols = ['power_stamina', 'power_strength', 'movement_agility',
                    'movement_acceleration', 'movement_sprint_speed', 'injury_free']
    X = df[feature_cols]
    y = df['target']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models = {
        'Logistic Regression': LogisticRegression(),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Support Vector Machine': SVC(kernel='rbf', probability=True)
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

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
