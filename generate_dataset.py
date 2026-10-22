
import pandas as pd
import numpy as np

# Generate reproducible sample data
rng = np.random.default_rng(42)

# Hourly timestamps for 90 days
dates = pd.date_range(
    start="2025-01-01",
    periods=90 * 24,
    freq="h"
)

hours = dates.hour.to_numpy()
days = dates.dayofweek.to_numpy()

# Simulated temperature
temperature = (
    25
    + 5 * np.sin(2 * np.pi * (hours - 8) / 24)
    + rng.normal(0, 1.5, len(dates))
)

# Simulated electricity consumption
consumption = (
    1.5
    + 1.8 * ((hours >= 18) & (hours <= 22))
    + 0.7 * ((hours >= 7) & (hours <= 9))
    + 0.3 * (days >= 5)
    + 0.04 * np.maximum(temperature - 25, 0)
    + rng.normal(0, 0.2, len(dates))
)

consumption = np.maximum(consumption, 0.1)

df = pd.DataFrame({
    "datetime": dates,
    "consumption_kwh": consumption.round(2),
    "temperature": temperature.round(2)
})

df.to_csv("data/training_electricity.csv", index=False)

print("GridSathi practice dataset created!")
print("Total rows:", len(df))
print(df.head())
