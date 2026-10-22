
import pandas as pd

# Read electricity data
df = pd.read_csv("data/electricity.csv")

# Convert date and time
df["datetime"] = pd.to_datetime(df["datetime"])

# Create features for machine learning
df["hour"] = df["datetime"].dt.hour
df["day_of_week"] = df["datetime"].dt.dayofweek
df["month"] = df["datetime"].dt.month

# Check missing values
print("Missing Values:")
print(df.isnull().sum())

# Display data
print("\nProcessed GridSathi Data:")
print(df.head())

# Save processed data
df.to_csv("data/processed_electricity.csv", index=False)

print("\nGridSathi Data Preparation Successful!")
