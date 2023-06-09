import os
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

# First step is download historical data from yFinance using the Yahoo Finace data file
# This code calculates the Value at Risk (VaR) of a portfolio using historical stock data.
# It prompts the user to enter ticker symbols and corresponding weights for the assets in the portfolio.
# The code loads the historical stock data for each ticker symbol from Excel files in a specified folder.
# It then calculates the log returns and covariance matrix of the asset returns.
# The user is prompted to enter the portfolio value and a confidence level.
# Using these inputs, the code calculates the portfolio VaR and individual VaR for each asset in the portfolio.
# It also calculates the undiversified VaR, which represents the total risk of the portfolio without considering diversification.
# The calculated VaR values are displayed along with the numbers used to calculate them.
# The code then performs an optimization using scipy.optimize.minimize to find the optimal weights for the minimum portfolio VaR.
# The optimal weights and the minimum portfolio VaR are displayed as the final result.

# Specify the folder path where the Excel files are located
folder_path = "Stock Data"

# Prompt the user for the ticker symbols and weights
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
    data = pd.read_excel(file_path)
    data_frames.append(data)

# Concatenate the data frames for all ticker symbols
combined_data = pd.concat(data_frames, axis=1)

# Calculate the log returns for each ticker symbol
log_returns = np.log(combined_data['Adj Close'] / combined_data['Adj Close'].shift(1))

# Calculate the covariance matrix of log returns
covariance_matrix = log_returns.cov().to_numpy()

# Calculate the portfolio variance
portfolio_variance = np.dot(np.dot(weights, covariance_matrix), weights)

# Prompt the user for the portfolio value
portfolio_value = float(input("Enter the portfolio value: "))

# Prompt the user for the confidence level
confidence_level = float(input("Enter the confidence level (between 0 and 1): "))

# Calculate the z-score corresponding to the confidence level
z_score = norm.ppf(confidence_level)

# Calculate the portfolio VaR
portfolio_VaR = portfolio_value * np.sqrt(portfolio_variance) * z_score

# Calculate individual VaR for each asset
individual_VaR_values = np.sqrt(np.diag(covariance_matrix)) * z_score * portfolio_value * weights

# Calculate undiversified VaR
undiversified_VaR = np.sum(individual_VaR_values)

# Display the numbers used to calculate the portfolio VaR
print("----- Portfolio VaR Calculation Details -----")
print(f"Ticker Symbols: {ticker_symbols}")
print(f"Weights: {weights}")
print(f"Portfolio Variance: {portfolio_variance:.4f}")
print(f"Portfolio Value: {portfolio_value}")
print(f"Confidence Level: {confidence_level}")
print(f"Z-Score: {z_score:.2f}")
print("----------------------------------------------")

# Display the portfolio VaR
print(f"The Value at Risk (VaR) of the portfolio at {confidence_level * 100:.2f}% confidence level is: {portfolio_VaR:.2f}")

# Display individual VaR values along with the numbers used to calculate them
print("------ Individual VaR ------")
for i, ticker_symbol in enumerate(ticker_symbols):
    individual_covariance = covariance_matrix[i, i]
    individual_weight = weights[i]
    individual_VaR = np.sqrt(individual_covariance) * z_score * portfolio_value * individual_weight
    print(f"Ticker: {ticker_symbol}, VaR: {individual_VaR:.2f}")
    print(f"   Variance: {individual_covariance:.4f}")
    print(f"   Weight: {individual_weight:.4f}")
    print("----------------------------------")

# Display the undiversified VaR
print(f"The Undiversified VaR of the portfolio at {confidence_level * 100:.2f}% confidence level is: {undiversified_VaR:.2f}")

# Define a function that calculates the portfolio VaR given a vector of weights
def portfolio_VaR(weights):
    portfolio_variance = np.dot(np.dot(weights, covariance_matrix), weights)
    portfolio_VaR = portfolio_value * np.sqrt(portfolio_variance) * z_score
    return portfolio_VaR

# Define a constraint that ensures the weights sum up to 1
def sum_constraint(weights):
    return np.sum(weights) - 1

# Define an initial guess for the weights (equal weights)
initial_weights = np.ones(len(ticker_symbols)) / len(ticker_symbols)

# Define the optimization problem using scipy.optimize.minimize
result = minimize(portfolio_VaR, initial_weights, method='SLSQP', constraints={'type': 'eq', 'fun': sum_constraint})

# Display the optimal weights and the minimum portfolio VaR
print("----- Optimal Weights and Minimum Portfolio VaR -----")
print(f"Optimal Weights: {', '.join([f'{w:.4f}' for w in result.x])}")
print(f"Sum of Weights: {np.sum(result.x):.4f}")
print(f"Minimum Portfolio VaR: {result.fun:.2f}")
print("------------------------------------------------------")