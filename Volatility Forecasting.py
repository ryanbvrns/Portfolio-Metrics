import os
import pandas as pd
import arch
import numpy as np
import matplotlib.pyplot as plt

# This code performs volatility forecasting for a specified ticker symbol using EWMA and GARCH(1,1) models.
# It assumes that historical stock data for the ticker symbol is available in Excel format in a specified folder.
# The code loads the data for the ticker symbol and calculates the log returns.
# It then performs EWMA volatility forecasting and GARCH(1,1) volatility forecasting using the ARCH library.
# The code plots the EWMA and GARCH volatility forecasts using matplotlib.
# Finally, it prints the summary of the GARCH model.

# Specify the folder path where the Excel files are located
folder_path = "Stock Data"

# Define the ticker symbol for which you want to perform volatility forecasting
ticker_symbol = ticker_symbols = input("Enter the ticker symbol: ")

# Load the data for the ticker symbol
file_name = f"{ticker_symbol}_stock_data.xlsx"
file_path = os.path.join(folder_path, file_name)
data = pd.read_excel(file_path, index_col=0, parse_dates=True)

# Calculate log returns
log_returns = np.log(data['Adj Close'] / data['Adj Close'].shift(1)).dropna()

# Perform EWMA volatility forecasting
ewma_volatility = log_returns.ewm(span=30, min_periods=30).std()

# Perform GARCH(1,1) volatility forecasting
garch_model = arch.arch_model(log_returns, vol='Garch', p=1, q=1)
garch_results = garch_model.fit(disp='off')
garch_volatility = garch_results.conditional_volatility

# Plot the EWMA and GARCH volatility forecasts
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(log_returns.index, ewma_volatility, label='EWMA Volatility')
ax.plot(log_returns.index, garch_volatility, label='GARCH Volatility')
ax.set_xlabel('Date')
ax.set_ylabel('Volatility')
ax.set_title(f'Volatility Forecasting for {ticker_symbol}')
ax.legend()
plt.show()

# Print the summary of GARCH model
print(garch_results.summary())
