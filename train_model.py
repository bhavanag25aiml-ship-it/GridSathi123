
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import joblib
from pathlib import Path

# Load practice dataset
df = pd.read_csv("data/training_electricity.csv")
df["datetime"] = pd.to_datetime(df["datetime"])

# Create time features
df["hour"] = df["datetime"].dt.hour
df["day_of_week"] = df["datetime"].dt.dayofweek
df["month"] = df["datetime"].dt.month

features = ["hour", "day_of_week", "month", "temperature"]

X = df[features]
y = df["consumption_kwh"]

# Split chronologically (80% train, 20% test)
split = int(len(df) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]
y_train = y.iloc[:split]
y_test = y.iloc[split:]

# Train XGBoost model
model = XGBRegressor(
    n_estimators=150,
    max_depth=4,
    learning_rate=0.05,
    random_state=42
)

model.fit(X_train, y_train)

# Predict electricity consumption
predictions = model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("GridSathi AI Model Training Completed!")
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("MAE:", round(mae, 3))
print("RMSE:", round(rmse, 3))

# Save trained model
Path("models").mkdir(exist_ok=True)
joblib.dump(model, "models/energy_model.pkl")

print("Model saved successfully!")
