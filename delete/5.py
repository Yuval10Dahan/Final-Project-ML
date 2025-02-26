import pandas as pd
import numpy as np
import time
from itertools import product
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve
from sklearn.metrics import mean_absolute_error, mean_squared_error


def load_data(file_path):
    df = pd.read_csv(file_path, low_memory=False)

    columns_needed = ['short_name', 'long_name', 'power_stamina', 'power_strength',
                      'movement_agility', 'movement_acceleration', 'movement_sprint_speed',
                      'player_traits', 'fifa_version', 'age', 'league_name', 'player_id', 'mentality_aggression']
    print("Total dataset size after dropping NaN values:", len(df))

    df['injury_free'] = df['player_traits'].astype(str).apply(
        lambda x: 1 if "Injury Free" in x or "Injury Prone" not in x else 0
    )
    df.drop(columns=['player_traits'], inplace=True)

    df_2020 = df[(df['fifa_version'] == 20) &
                 (df['age'] >= 21) &
                 (df['power_stamina'] >= 60) &
                 (df['injury_free'] == 1)].copy()

    target_players = df[(df['fifa_version'] == 24) &
                        (df['league_name'].isin(['La Liga', 'Premier League', 'Bundesliga', 'Ligue 1', 'Serie A']))]

    target_player_ids = set(target_players['player_id'])
    print("Total players in high-intensity leagues (2024):", len(target_player_ids))

    df_2020.loc[:, 'target'] = df_2020['player_id'].apply(lambda x: 1 if x in target_player_ids else 0)
    print("Players labeled as moving (target=1):", df_2020['target'].sum())

    return df_2020


def train_and_evaluate_models(df):
    feature_cols = ['power_stamina', 'power_strength', 'movement_agility',
                    'movement_acceleration', 'movement_sprint_speed', 'injury_free', 'mentality_aggression']
    X = df[feature_cols]
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    smote = SMOTE(sampling_strategy=0.6, random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model_params = {
        'Logistic Regression': [
            {'max_iter': [500, 1000], 'C': [0.1, 1, 10]}
        ],
        'Random Forest': [
            {'n_estimators': [100, 200, 300], 'max_depth': [5, 10, 20]}
        ],
        'Support Vector Machine': [
            {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
        ]
    }

    models = {
        'Logistic Regression': LogisticRegression(class_weight='balanced'),
        'Random Forest': RandomForestClassifier(class_weight='balanced', random_state=42),
        'Support Vector Machine': SVC(kernel='rbf', probability=True, class_weight='balanced')
    }

    results = []
    for name, model in models.items():
        for param_set in model_params[name]:
            param_combinations = [dict(zip(param_set, v)) for v in product(*param_set.values())]
            for params in param_combinations:
                model.set_params(**params)
                model.fit(X_train, y_train)
                y_pred_probs = model.predict_proba(X_test)[:, 1]

                precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_probs)
                f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-6)
                best_threshold = thresholds[np.argmax(f1_scores)]
                y_pred = (y_pred_probs > best_threshold).astype(int)

                metrics = {
                    'Model': name,
                    'Parameters': str(params),
                    'Accuracy': accuracy_score(y_test, y_pred),
                    'Precision': precision_score(y_test, y_pred, zero_division=0),
                    'Recall': recall_score(y_test, y_pred, zero_division=0),
                    'F1 Score': f1_score(y_test, y_pred, zero_division=0),
                    'MAE': mean_absolute_error(y_test, y_pred_probs),
                    'MSE': mean_squared_error(y_test, y_pred_probs)
                }
                results.append(metrics)

    return pd.DataFrame(results)


if __name__ == "__main__":
    start_time = time.perf_counter()

    file_path = "male_players_2024-2015.csv"
    df = load_data(file_path)
    results_df = train_and_evaluate_models(df)
    print(results_df)

    end_time = time.perf_counter()
    print(f"Total runtime: {end_time - start_time:.2f} seconds")
