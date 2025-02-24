import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(file_path):
    """Load dataset from CSV and check for necessary columns."""
    df = pd.read_csv(file_path, low_memory=False)

    if "league_name" not in df.columns:
        raise ValueError("No 'league_name' column found. Ensure your dataset has league information.")

    return df


def preprocess_data(df):
    """Preprocess dataset: select attributes, encode leagues, and scale features."""
    attribute_columns = [
        "pace", "shooting", "passing", "dribbling", "defending", "physic",
        "attacking_finishing", "attacking_heading_accuracy", "attacking_short_passing", "attacking_volleys",
        "skill_dribbling", "skill_curve", "skill_fk_accuracy", "skill_long_passing", "skill_ball_control",
        "movement_acceleration", "movement_sprint_speed", "movement_agility", "movement_balance",
        "power_shot_power", "power_jumping", "power_stamina", "power_strength", "power_long_shots",
        "mentality_aggression", "mentality_interceptions", "mentality_positioning", "mentality_vision",
        "mentality_composure", "defending_marking_awareness"
    ]

    df_filtered = df[["league_name"] + attribute_columns].dropna()

    # Encode league names
    le = LabelEncoder()
    df_filtered["league_encoded"] = le.fit_transform(df_filtered["league_name"])

    # Standardize attributes
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_filtered[attribute_columns])

    return df_filtered, X_scaled, le.classes_, attribute_columns


def apply_pca(X_scaled):
    """Apply PCA for dimensionality reduction."""
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca


def perform_clustering(X_scaled, df_filtered):
    """Apply K-Means clustering and assign cluster labels."""
    kmeans = KMeans(n_clusters=3, random_state=42)
    df_filtered["cluster"] = kmeans.fit_predict(X_scaled)
    return df_filtered


def train_models(X_scaled, df_filtered):
    """Train Decision Tree and Logistic Regression models to classify leagues."""
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, df_filtered["league_encoded"], test_size=0.2, random_state=42
    )

    # Decision Tree
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Logistic Regression
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(X_train, y_train)
    y_pred = log_reg.predict(X_test)

    return clf, log_reg, y_test, y_pred


def evaluate_models(y_test, y_pred, class_names):
    """Evaluate models using classification metrics."""
    report = classification_report(y_test, y_pred, target_names=class_names, zero_division=1)
    accuracy = accuracy_score(y_test, y_pred)

    print("\nClassification Report:")
    print(report)
    print("\nLogistic Regression Accuracy:", accuracy)


def feature_importance_analysis(clf, attribute_columns):
    """Analyze feature importance from Decision Tree."""
    feature_importances = pd.DataFrame({
        "Feature": attribute_columns,
        "Importance": clf.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    print("\nTop 10 Important Features for League Classification:")
    print(feature_importances.head(10))


def plot_pca_clusters(X_pca, df_filtered):
    """Plot PCA projection with clustering results."""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df_filtered["cluster"], palette="viridis", alpha=0.7)
    plt.title("PCA Projection with Clustering")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Cluster")
    plt.show()


def main():
    file_path = "male_players_2024-2015.csv"

    # Load and preprocess data
    df = load_data(file_path)
    df_filtered, X_scaled, class_names, attribute_columns = preprocess_data(df)

    # PCA and clustering
    X_pca = apply_pca(X_scaled)
    df_filtered = perform_clustering(X_scaled, df_filtered)

    print(df_filtered["league_name"].value_counts())

    # Train models
    clf, log_reg, y_test, y_pred = train_models(X_scaled, df_filtered)

    # Evaluate models
    evaluate_models(y_test, y_pred, class_names)

    # Feature importance
    feature_importance_analysis(clf, attribute_columns)

    # Plot PCA Clusters
    plot_pca_clusters(X_pca, df_filtered)


if __name__ == "__main__":
    main()
