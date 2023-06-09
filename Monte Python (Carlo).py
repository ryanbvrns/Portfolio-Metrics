import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# This code performs a Monte Carlo simulation to estimate the distribution of portfolio values.
# It assumes that historical stock data for the specified ticker symbols is available in Excel format in a specified folder.
# The code loads the data for each ticker symbol, calculates the log returns, and calculates the average returns and covariance matrix.
# It then generates random returns based on the average returns and covariance matrix using Monte Carlo simulations.
# The code calculates the portfolio value for each set of random returns and stores them in an array.
# It finally plots a histogram of the portfolio values and prints the mean and standard deviation of the portfolio values.


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

# Calculate the log returns for each ticker symbol
log_returns = np.log(combined_data['Adj Close'] / combined_data['Adj Close'].shift(1))

# Calculate the average returns and covariance matrix
average_returns = log_returns.mean()
covariance_matrix = log_returns.cov()

# Define the number of Monte Carlo simulations
num_simulations = 1000

# Perform Monte Carlo simulations
portfolio_values = []
for _ in range(num_simulations):
    # Generate random returns for each ticker symbol
    random_returns = np.random.multivariate_normal(average_returns, covariance_matrix, size=1)

    # Calculate the portfolio value based on random returns
    portfolio_value = np.sum(random_returns * weights)

    # Append the portfolio value to the list
    portfolio_values.append(portfolio_value)

# Convert the portfolio values to a NumPy array
portfolio_values = np.array(portfolio_values)

# Calculate the mean and standard deviation of portfolio values
mean_portfolio_value = np.mean(portfolio_values)
std_portfolio_value = np.std(portfolio_values)

# Plot the histogram of portfolio values
plt.hist(portfolio_values, bins=30, edgecolor='black')
plt.xlabel('Portfolio Value')
plt.ylabel('Frequency')
plt.title('Monte Carlo Simulation: Portfolio Value Distribution')
plt.show()

# Print the mean and standard deviation of portfolio values
print(f"Mean Portfolio Value: {mean_portfolio_value:.2f}")
print(f"Standard Deviation of Portfolio Value: {std_portfolio_value:.2f}")