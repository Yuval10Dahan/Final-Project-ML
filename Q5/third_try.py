import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import time

def load_data(file_path):
    # Load the dataset
    data_frame = pd.read_csv(file_path, low_memory=False)

    # Select only numerical columns from the dataset
    numerical_cols = data_frame.select_dtypes(include=[np.number]).columns.tolist()

    # Remove the target column "overall" from the feature set
    numerical_cols.remove("overall")

    # Remove biased features
    biased_features = ["player_id", "fifa_version"]
    numerical_cols = [col for col in numerical_cols if col not in biased_features]
    X = data_frame[numerical_cols].fillna(0)

    # Extract the target variable
    y = data_frame["overall"]

    return X, y


def remove_correlated_features(X, threshold=0.85):
    # Compute the absolute correlation matrix of the dataset
    corr_matrix = X.corr().abs()

    # Store highly correlated features
    correlated_features = set()

    # Iterate over the upper triangle of the correlation matrix
    for i in range(len(corr_matrix.columns)):
        for j in range(i):  # Only check values below the diagonal to avoid redundancy

            # If correlation is greater than the threshold, mark the feature for removal
            if corr_matrix.iloc[i, j] > threshold:
                correlated_features.add(corr_matrix.columns[i])

    # Drop the identified correlated features from the dataset
    return X.drop(columns=correlated_features)


def split_data(X, y, test_size=0.2, random_state=42):
    # Split the data into train and test sets (20% of the data for testing)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def compute_feature_importance(X_train, y_train):
    # Initialize a Random Forest Regressor with 100 trees
    random_forest = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train the model using the training data
    random_forest.fit(X_train, y_train)

    # Store feature names and their corresponding importance scores
    feature_importances = pd.DataFrame({"Feature": X_train.columns, "Importance": random_forest.feature_importances_})

    return feature_importances.sort_values(by="Importance", ascending=False)


def perform_pca(X, n_components=6):
    # Normalize feature values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit PCA on the scaled data and transform it to the new lower-dimensional space
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    # Plot explained variance
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, n_components + 1), np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Explained Variance by PCA Components')
    plt.show()

    return X_pca, pca


if __name__ == "__main__":
    start_time = time.perf_counter()  # Start timing

    file_path = "male_players_2024-2015.csv"
    X, y = load_data(file_path)
    X_reduced = remove_correlated_features(X)
    X_train, X_test, y_train, y_test = split_data(X_reduced, y)

    feature_importances = compute_feature_importance(X_train, y_train)
    print("Top 10 Important Features (After Refinement):")
    print(feature_importances.head(10))

    # Perform PCA on the dataset
    perform_pca(X_reduced)

    end_time = time.perf_counter()
    print(f"Total runtime: {end_time - start_time:.2f} seconds")
