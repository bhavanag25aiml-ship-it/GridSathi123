
import pandas as pd
import matplotlib.pyplot as plt

# Load electricity data
df = pd.read_csv("data/electricity.csv")

# Convert date and time
df["datetime"] = pd.to_datetime(df["datetime"])

# Draw electricity consumption graph
plt.figure(figsize=(10, 5))

plt.plot(
    df["datetime"],
    df["consumption_kwh"],
    marker="o",
    color="blue"
)

plt.title("GridSathi - Electricity Consumption")
plt.xlabel("Time")
plt.ylabel("Electricity Consumption (kWh)")

plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()

plt.show()
