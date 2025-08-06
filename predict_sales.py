import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

# Sample data based on the user-provided CSV
data = {
    "Date": ["2025-07-01", "2025-07-01", "2025-07-02", "2025-07-03", "2025-07-03", "2025-07-04"],
    "Product": ["Pencil", "Notebook", "Eraser", "Pencil", "Notebook", "Eraser"],
    "Units_Sold": [50, 20, 30, 40, 10, 50],
    "Unit_Price": [5, 20, 3, 5, 20, 3]
}

df = pd.DataFrame(data)
df["Date"] = pd.to_datetime(df["Date"])
df["Total_Revenue"] = df["Units_Sold"] * df["Unit_Price"]

# Group by Date for daily total revenue
daily_revenue = df.groupby("Date").agg({
    "Units_Sold": "sum",
    "Total_Revenue": "sum"
}).reset_index()

# Prepare data for Linear Regression
daily_revenue["Day_Num"] = (daily_revenue["Date"] - daily_revenue["Date"].min()).dt.days
X = daily_revenue[["Day_Num"]]
y = daily_revenue["Total_Revenue"]

# Train linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict for the next 3 days
future_days = np.array([[day] for day in range(X["Day_Num"].max() + 1, X["Day_Num"].max() + 4)])
future_predictions = model.predict(future_days)

# Create a DataFrame for predictions
future_dates = [daily_revenue["Date"].max() + pd.Timedelta(days=i) for i in range(1, 4)]
pred_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted_Total_Revenue": future_predictions.round(2)
})



# Save files for Tableau
df.to_csv("sales_data.csv", index=False)
print("sales_data.csv saved")
pred_df.to_csv("future_predictions.csv", index=False)
print("future_predictions.csv saved.")