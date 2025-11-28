import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
import joblib
import os

# Load dataset
df = pd.read_csv("data/housing.csv")

# Clean dataset
df = df.dropna()

# Split into features and target
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Pipeline: Scaling + Ridge Regression
model = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", Ridge(alpha=1.0))
])

model.fit(X_train, y_train)

# Create model directory if missing
os.makedirs("model", exist_ok=True)

print("Saving model now...")
joblib.dump(model, "house_price_model.pkl")
print("Model saved successfully!")
