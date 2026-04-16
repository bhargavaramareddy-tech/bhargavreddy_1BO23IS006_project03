import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# -------------------- LOAD DATA --------------------
df = pd.read_csv("data.csv")

print("Dataset Preview:")
print(df.head())

print("\nColumns in dataset:")
print(df.columns)

print("\nShape before cleaning:", df.shape)

# -------------------- DATA PREPROCESSING --------------------

# Convert categorical 'Sex' to numeric
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Fill missing values (IMPORTANT)
df = df.fillna(df.mean(numeric_only=True))

print("\nShape after cleaning:", df.shape)

# -------------------- FEATURES & TARGET --------------------

X = df[['Sex', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']]
y = df['Calories']

print("\nX shape:", X.shape)
print("y shape:", y.shape)

# -------------------- TRAIN TEST SPLIT --------------------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# -------------------- SCALING --------------------

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------- PCA --------------------

pca = PCA(n_components=3)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# -------------------- MODEL --------------------

model = LinearRegression()
model.fit(X_train, y_train)

# -------------------- PREDICTION --------------------

y_pred = model.predict(X_test)

# -------------------- EVALUATION --------------------

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("Mean Squared Error:", mse)
print("R2 Score:", r2)

# -------------------- VISUALIZATION --------------------

# Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()

# Calories Distribution
df['Calories'].plot(kind='hist')
plt.title("Calories Distribution")
plt.xlabel("Calories Burned")
plt.show()