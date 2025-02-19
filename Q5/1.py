import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import time

start_time = time.perf_counter()  # Start timing

# Load the dataset
file_path = "male_players_2024-2015.csv"
df = pd.read_csv(file_path, low_memory=False)

# Selecting numerical attributes (excluding categorical and irrelevant ones)
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numerical_cols.remove("overall")  # Target variable

# Extract features and target
X = df[numerical_cols].fillna(0)  # Fill missing values
y = df["overall"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature importance using Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Get feature importances
feature_importances = pd.DataFrame({"Feature": X.columns, "Importance": rf.feature_importances_})
feature_importances = feature_importances.sort_values(by="Importance", ascending=False)

# Display top 10 important features
print("Top 10 Important Features:")
print(feature_importances.head(10))

# PCA for dimensionality reduction
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=10)  # Reduce to 10 principal components
X_pca = pca.fit_transform(X_scaled)

# Explained variance ratio
plt.figure(figsize=(8,5))
plt.plot(range(1, 11), np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by PCA Components')
plt.show()


end_time = time.perf_counter()
print(f"Total runtime: {end_time - start_time:.2f} seconds")
