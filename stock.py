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

# Drop 'Date', 'Std_Dev_20', and 'Skewness_20' columns
df = df.drop(['Date', 'Std_Dev_20', 'Skewness_20'], axis=1)

# Apply log transformation to 'Volume' and 'Close'
df['Volume'] = np.log(df['Volume'] + 1)  # Adding 1 to avoid log(0)
df['Log_Close'] = np.log(df['Close'])  # Target variable

# Add the Price Shift feature (difference between consecutive closing prices)
df['Price_Shift'] = df['Close'].diff().fillna(0)

# Define function for sequence generation
def create_sequences(data, labels, window_size=30):
    X, y = [], []
    for i in range(window_size, len(data) - 1):  # We predict the next day's price
        X.append(data[i - window_size:i])  # 30 days of data for each sequence
        y.append(labels[i + 1])  # The close price of the next day
    return np.array(X), np.array(y)

# Features (excluding 'Log_Close') and the target label 'Log_Close'
features = df.drop('Log_Close', axis=1).values
target = df['Log_Close'].values

# Generate the sequences for the model
X, y = create_sequences(features, target)

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

# Initialize the Random Forest Regressor with hyperparameters to prevent overfitting
model = RandomForestRegressor(
    n_estimators=50,  # Number of trees in the forest
    max_depth=4,  # Maximum depth of each tree
    min_samples_split=8,  # Minimum samples required to split a node
    min_samples_leaf=9,  # Minimum samples required at a leaf node
    max_features=0.8,  # Proportion of features considered for splitting
    random_state=70
)

# Cross-validation with TimeSeriesSplit to avoid leakage of future data
tscv = TimeSeriesSplit(n_splits=5)
for train_index, val_index in tscv.split(X_train_flattened):
    X_train_cv, X_val_cv = X_train_flattened[train_index], X_train_flattened[val_index]
    y_train_cv, y_val_cv = y_train[train_index], y_train[val_index]

    # Train the model on the training fold
    model.fit(X_train_cv, y_train_cv)

    # Validate on the validation fold
    y_val_pred = model.predict(X_val_cv)
    val_mse = mean_squared_error(y_val_cv, y_val_pred)
    val_rmse = np.sqrt(val_mse)
    print(f"Validation MSE: {val_mse}, RMSE: {val_rmse}")

# Train the model on the entire training set
model.fit(X_train_flattened, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_flattened)

# Inverse the log transformation on predictions and actual values to return them to the original scale
y_pred_original = np.exp(y_pred)
y_test_original = np.exp(y_test)

# Predict the next day's price using the last 30 days of training data
last_30_days = X_train_scaled[-1]  # Extract the last sequence used in training

# Reshape to match (1, 150) as expected by RandomForest
last_30_days_flattened = last_30_days.reshape(1, -1)  # 1 sample with 150 features

# Predict the next day's log close price
next_day_log_pred = model.predict(last_30_days_flattened)

# Inverse the log transformation to get the actual price
next_day_pred_original = np.exp(next_day_log_pred[0])

print(f"Predicted Next Day's Closing Price: {next_day_pred_original}")

# Evaluate the model performance
mse = mean_squared_error(y_test_original, y_pred_original)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_original, y_pred_original)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R² Score: {r2}")

# Plotting the Actual vs Predicted Prices (on original scale)
plt.figure(figsize=(12, 6))
plt.plot(np.arange(len(y_test_original)), y_test_original, label='Actual Price', color='blue')
plt.plot(np.arange(len(y_pred_original)), y_pred_original, label='Predicted Price', color='red')
plt.title('Actual vs Predicted Stock Prices (Next Day\'s Closing Price)')
plt.xlabel('Index')
plt.ylabel('Closing Price')
plt.legend()
plt.show()
