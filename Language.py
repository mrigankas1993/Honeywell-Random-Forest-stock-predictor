
import yfinance as yf
import pandas as pd

# Download stock data for the last 5 years
ticker = 'HON'
data = yf.download(ticker, period="5y")  # Downloads last 5 years of data

# Extract relevant columns: Open, Close, and Volume
data_filtered = data[['Open', 'Close', 'Volume']]

# Calculate RSI (14-day)
delta = data_filtered['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
rsi_14 = 100 - (100 / (1 + rs))

# Add RSI as a new column
data_filtered['RSI_14'] = rsi_14

# Calculate Standard Deviation (20 days)
data_filtered['Std_Dev_20'] = data_filtered['Close'].rolling(window=20).std()

# Calculate Skewness (20 days)
data_filtered['Skewness_20'] = data_filtered['Close'].rolling(window=20).skew()

# Calculate MACD (12-day EMA - 26-day EMA)
ema_12 = data_filtered['Close'].ewm(span=12, adjust=False).mean()
ema_26 = data_filtered['Close'].ewm(span=26, adjust=False).mean()
macd = ema_12 - ema_26

# Add MACD as a new column
data_filtered['MACD'] = macd

# Reset index to include the Date in the JSON
data_filtered = data_filtered.reset_index()

# Save the filtered data to a JSON file with proper formatting
with open('honey_stock.json', 'w') as f:
    # Writing JSON structure with the proper formatting
    f.write('[\n')  # Start of JSON array
    for i, row in data_filtered.iterrows():
        # Writing each row as a JSON object
        json_line = row.to_json(orient='records', date_format='iso')
        f.write(json_line)
        if i < len(data_filtered) - 1:
            f.write(',\n')  # Separate each record with a comma
    f.write('\n]')  # End of JSON array

print("\nData has been saved to the file honey_stock.json'")