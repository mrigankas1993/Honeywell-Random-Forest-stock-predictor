import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit

# Load the JSON data (replace 'honey_stock.json' with your actual JSON file path)
with open('honey_stock.json', 'r') as f:
    data = json.load(f)

# Convert the loaded data into a DataFrame with correct columns
df = pd.DataFrame(data, columns=["Date", "Open", "Close", "Volume", "RSI_14", "Std_Dev_20", "Skewness_20", "MACD"])

# Save the 'Date' column in a separate variable
dates = df['Date'].copy()

# Drop 'Date', 'Std_Dev_20', and 'Skewness_20' columns
df = df.drop(['Date', 'Std_Dev_20', 'Skewness_20'], axis=1)

# Apply log transformation to 'Volume' and 'Close'
df['Volume'] = np.log(df['Volume'] + 1)  # Adding 1 to avoid log(0)
df['Log_Close'] = np.log(df['Close'])  # Target variable

# Add the Price Shift feature (difference between consecutive closing prices)
df['Price_Shift'] = df['Close'].diff().fillna(0)

# Define function for sequence generation
def create_sequences(data, labels, window_size=60, prediction_days=3):
    X, y = [], []
    for i in range(window_size, len(data) - prediction_days):
        X.append(data[i - window_size:i])  # Use 60 days of data
        y.append(np.mean(labels[i:i + prediction_days]))  # Predict the 3-day average
    return np.array(X), np.array(y)

# Features (excluding 'Log_Close') and the target label 'Log_Close'
features = df.drop('Log_Close', axis=1).values
target = df['Log_Close'].values

# Generate sequences with 60-day window and 3-day average as target
X, y = create_sequences(features, target, window_size=60, prediction_days=3)

# Split the data chronologically into training and testing sets
train_size = int(len(X) * 0.7)  # Use 70% of the data for training
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# Apply Min-Max Scaling
scaler = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

# Flatten the 3D data into 2D for Random Forest
X_train_flattened = X_train_scaled.reshape(X_train_scaled.shape[0], -1)
X_test_flattened = X_test_scaled.reshape(X_test_scaled.shape[0], -1)

# Initialize the Random Forest Regressor
model = RandomForestRegressor(
    n_estimators=50,  # Number of trees in the forest
    max_depth=4,  # Maximum depth of each tree
    min_samples_split=8,  # Minimum samples required to split a node
    min_samples_leaf=9,  # Minimum samples required at a leaf node
    max_features=0.8,  # Proportion of features considered for splitting
    random_state=70
)

# Train the model on the training set
model.fit(X_train_flattened, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_flattened)

# Inverse the log transformation on predictions and actual values to return them to the original scale
y_pred_original = np.exp(y_pred)
y_test_original = np.exp(y_test)

# Predict the 3-day average price using the last 60 days of training data
last_60_days = X_train_scaled[-1]  # Extract the last sequence used in training

# Reshape to match (1, 240) as expected by RandomForest (60 days × number of features)
last_60_days_flattened = last_60_days.reshape(1, -1)  # 1 sample with 240 features

# Predict the next 3 days' average log close price
next_3_day_avg_log_pred = model.predict(last_60_days_flattened)

# Inverse the log transformation to get the actual price
next_3_day_avg_pred_original = np.exp(next_3_day_avg_log_pred[0])

print(f"Predicted 3-Day Average Closing Price: {next_3_day_avg_pred_original}")

# Evaluate the model performance
mse = mean_squared_error(y_test_original, y_pred_original)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_original, y_pred_original)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R² Score: {r2}")

# Extract the relevant dates for the test set
# Ensure that test_dates matches the length of y_pred_original
# Ensure that test_dates matches the length of y_pred_original
# Convert the 'Date' column to datetime format and remove the time part
test_dates = pd.to_datetime(dates.iloc[train_size + 60:train_size + 60 + len(y_pred_original)]).dt.date
print(test_dates)

# Plotting the Actual vs Predicted Prices (3-Day Average) with Dates on the x-axis
plt.figure(figsize=(12, 6))
plt.plot(test_dates, y_test_original, label='Actual Price', color='blue')
plt.plot(test_dates, y_pred_original, label='Predicted Price', color='red')
plt.title('Actual vs Predicted Stock Prices (3-Day Average Closing Price for Honeywell Stock Price)')
plt.xlabel('Date')
plt.ylabel('3-Day Average Closing Price')
plt.legend()

# Display every 50th date on the x-axis for better readability
plt.xticks(test_dates[::50], rotation=45)  # Adjust step as needed to control spacing

plt.tight_layout()  # Adjust layout to prevent label clipping
plt.show()

