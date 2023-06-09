import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# This code performs a stress test on a portfolio by applying stress factors to historical stock prices.
# It prompts the user to enter ticker symbols and corresponding weights for the assets in the portfolio.
# The code also prompts the user to enter stress factors for each stock.
# The stress factors represent the degree of increase or decrease to be applied to the stock prices.
# The code loads the historical stock data for each ticker symbol from Excel files in a specified folder.
# It then combines the data frames for all ticker symbols and applies the stress factors to the stock prices.
# The stressed portfolio value is calculated as the weighted sum of the stressed stock prices.
# The code also calculates the drawdown, which measures the decline in portfolio value from its peak.
# Finally, the code plots the stressed portfolio value and drawdown using matplotlib.

# Specify the folder path where the Excel files are located
folder_path = "Stock Data"

# Define the ticker symbols and corresponding weights in the portfolio
ticker_symbols = input("Enter the ticker symbols (separated by spaces): ").split()
weights = input("Enter the corresponding weights (separated by spaces): ").split()

# Convert the weights to floats and normalize them to sum up to 1
weights = np.array([float(weight) for weight in weights])
weights /= np.sum(weights)

# Load the data for each ticker symbol
data_frames = []
for ticker_symbol in ticker_symbols:
    file_name = f"{ticker_symbol}_stock_data.xlsx"
    file_path = os.path.join(folder_path, file_name)
    data = pd.read_excel(file_path, index_col=0, parse_dates=True)
    data_frames.append(data)

# Combine the data frames for all ticker symbols
combined_data = pd.concat(data_frames, axis=1)

# Prompt the user to enter stress factors for each stock
stress_factors = []
for ticker_symbol in ticker_symbols:
    stress_factor = float(input(f"Enter the stress factor for {ticker_symbol}: "))
    stress_factors.append(stress_factor)

# Convert the stress factors to a NumPy array
stress_factors = np.array(stress_factors)

# Apply stress factors to the stock prices
stressed_prices = combined_data['Adj Close'] * stress_factors

# Calculate the stressed portfolio value
portfolio_value = np.sum(stressed_prices * weights, axis=1)

# Calculate the drawdown
portfolio_return = portfolio_value.pct_change()
rolling_max = portfolio_value.cummax()
drawdown = (portfolio_value - rolling_max) / rolling_max

# Plot the stressed portfolio value and drawdown
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

ax1.plot(portfolio_value)
ax1.set_ylabel("Portfolio Value")

ax2.plot(drawdown, color='red')
ax2.set_ylabel("Drawdown")

plt.xlabel("Date")
plt.suptitle("Stress Test: Portfolio Value and Drawdown")
plt.show()
