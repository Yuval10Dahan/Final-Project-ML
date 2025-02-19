import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = "male_players_2024-2015.csv"
df = pd.read_csv(file_path, low_memory=False)

# Selecting numerical attributes (excluding categorical and irrelevant ones)
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numerical_cols.remove("overall")  # Target variable

# Remove biased features and duplicates
biased_features = ["player_id", "fifa_version", "club_team_id"]
numerical_cols = list(set(numerical_cols) - set(biased_features))

# Ensure no duplicate feature names exist
numerical_cols = list(dict.fromkeys(numerical_cols))

# Extract features and target
X = df[numerical_cols].fillna(0)  # Fill missing values
y = df["overall"]

# Identify highly correlated features (e.g., financial attributes)
corr_matrix = X.corr().abs()
high_correlation_threshold = 0.85
correlated_features = set()
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if corr_matrix.iloc[i, j] > high_correlation_threshold:
            correlated_features.add(corr_matrix.columns[i])

# Remove correlated features
X_reduced = X.drop(columns=correlated_features)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

# Feature importance using Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Get feature importances
feature_importances = pd.DataFrame({"Feature": X_reduced.columns, "Importance": rf.feature_importances_})
feature_importances = feature_importances.sort_values(by="Importance", ascending=False)

# Display top 10 important features after refining
print("Top 10 Important Features (After Final Refinement):")
print(feature_importances.head(10))

# PCA for dimensionality reduction
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_reduced)
pca = PCA(n_components=5)  # Reduce to top 5 principal components
X_pca = pca.fit_transform(X_scaled)

# Explained variance ratio
plt.figure(figsize=(8,5))
plt.plot(range(1, 6), np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by PCA Components')
plt.show()
