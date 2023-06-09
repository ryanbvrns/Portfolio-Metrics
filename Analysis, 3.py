import os
import pandas as pd
import matplotlib.pyplot as plt


# This code is currently not working!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# Specify the folder path where the Excel files are located
folder_path = "Stock Data"

# Prompt the user for the ticker symbols and weights
ticker_symbols = input("Enter the ticker symbols (separated by spaces): ").split()

# Load the data for each ticker symbol
data_frames = []
for ticker_symbol in ticker_symbols:
    file_name = f"{ticker_symbol}_stock_data.xlsx"
    file_path = os.path.join(folder_path, file_name)
    data = pd.read_excel(file_path)
    data_frames.append(data)

# Combine the data frames for all ticker symbols
combined_data = pd.concat(data_frames,axis=1)

# Plotting daily adjusted closing prices
plt.figure(figsize=(10, 6))
for ticker_symbol in ticker_symbols:
    plt.plot(combined_data.index, combined_data[ticker_symbol]['Adj Close'], label=ticker_symbol)

plt.xlabel('Date')
plt.ylabel('Adjusted Closing Price')
plt.title('Daily Adjusted Closing Prices')
plt.legend()
plt.grid(True)
plt.show()

# Plotting log prices
plt.figure(figsize=(10, 6))
for ticker_symbol in ticker_symbols:
    plt.plot(combined_data.index, combined_data[ticker_symbol]['Log Return'], label=ticker_symbol)

plt.xlabel('Date')
plt.ylabel('Log Price')
plt.title('Log Prices')
plt.legend()
plt.grid(True)
plt.show()
