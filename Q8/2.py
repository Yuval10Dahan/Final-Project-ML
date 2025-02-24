import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from scipy.spatial import ConvexHull


def load_data(file_path):
    df = pd.read_csv(file_path, low_memory=False)
    if "league_id" not in df.columns:
        raise ValueError("No 'league_id' column found. Ensure your dataset includes league information.")
    return df


def preprocess_data(df):
    chosen_leagues = [13, 53]  # Premier League = 13, La Liga = 53
    attribute_columns = [
        "pace", "shooting", "passing", "dribbling", "defending", "physic", "attacking_short_passing",
        "skill_dribbling", "skill_long_passing", "skill_ball_control", "movement_acceleration",
        "movement_sprint_speed", "movement_agility", "movement_reactions", "movement_balance", "power_jumping",
        "power_stamina", "power_strength", "power_long_shots", "mentality_aggression", "mentality_interceptions",
        "mentality_positioning", "mentality_vision", "defending_marking_awareness"
    ]

    # Keep only the latest FIFA version for each player_id
    df_sorted = df.sort_values(by=["player_id", "fifa_version"], ascending=[True, False])
    df_latest = df_sorted.drop_duplicates(subset=["player_id"], keep="first")

    # Filter by selected leagues
    df_filtered = df_latest[df_latest["league_id"].isin(chosen_leagues)][["league_id"] + attribute_columns].dropna()

    # Feature Engineering: Interaction Terms
    df_filtered["physicality"] = df_filtered["power_strength"] * df_filtered["power_stamina"]
    attribute_columns.append("physicality")

    # Encode league IDs
    le = LabelEncoder()
    df_filtered["league_encoded"] = le.fit_transform(df_filtered["league_id"])

    # Standardize attributes
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_filtered[attribute_columns])

    return df_filtered, X_scaled, le.classes_, attribute_columns



def apply_pca(X_scaled):
    pca = PCA(n_components=2)
    return pca.fit_transform(X_scaled)


def perform_clustering(X_scaled, df_filtered, num_clusters=5):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    df_filtered["cluster"] = kmeans.fit_predict(X_scaled)
    return df_filtered


def balance_classes(X, y):
    smote = SMOTE(random_state=42)
    return smote.fit_resample(X, y)


def train_models(X_scaled, df_filtered):
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, df_filtered["league_encoded"], test_size=0.2, random_state=42
    )
    X_train, y_train = balance_classes(X_train, y_train)

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(eval_metric='mlogloss', random_state=42),  # Removed use_label_encoder
        "SVM": SVC(kernel='rbf', probability=True, random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = (model, y_test, y_pred)
        print(f"\n{name} Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(classification_report(y_test, y_pred))

    return results



def feature_importance_analysis(models, attribute_columns):
    rf_model = models["RandomForest"][0]
    xgb_model = models["XGBoost"][0]
    rf_importance = pd.DataFrame({"Feature": attribute_columns, "Importance": rf_model.feature_importances_})
    xgb_importance = pd.DataFrame({"Feature": attribute_columns, "Importance": xgb_model.feature_importances_})
    rf_importance = rf_importance.sort_values(by="Importance", ascending=False)
    xgb_importance = xgb_importance.sort_values(by="Importance", ascending=False)
    print("\nTop 10 Features - Random Forest")
    print(rf_importance.head(10))
    print("\nTop 10 Features - XGBoost")
    print(xgb_importance.head(10))


def plot_pca_clusters(X_pca, df_filtered):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df_filtered["cluster"], palette="viridis", alpha=0.7)

    for i in range(df_filtered["cluster"].nunique()):
        cluster_points = X_pca[df_filtered["cluster"] == i]
        hull = ConvexHull(cluster_points)
        for simplex in hull.simplices:
            plt.plot(cluster_points[simplex, 0], cluster_points[simplex, 1], 'k-')

    plt.title("PCA Projection with Clustering and Convex Hulls")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Cluster")
    plt.show()


def main():
    file_path = "male_players_2024-2015.csv"
    df = load_data(file_path)
    df_filtered, X_scaled, class_names, attribute_columns = preprocess_data(df)
    X_pca = apply_pca(X_scaled)
    df_filtered = perform_clustering(X_scaled, df_filtered)

    print(df_filtered["league_id"].value_counts())
    models = train_models(X_scaled, df_filtered)
    feature_importance_analysis(models, attribute_columns)
    plot_pca_clusters(X_pca, df_filtered)


if __name__ == "__main__":
    main()
