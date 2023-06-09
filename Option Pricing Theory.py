from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

# This code performs option pricing simulations using the Black-Scholes model,
# binomial tree model, and Brownian motion. It calculates the prices of European
# options and generates stock price paths based on different models.

# Functions:
# - black_scholes(S, K, r, T, sigma, option_type): Calculates the price of an option
#   using the Black-Scholes model.
# - binomial_tree(S, K, r, T, sigma, option_type, n): Calculates the price of an option
#   using the binomial tree model.
# - brownian_motion(S, r, sigma, T, num_steps): Generates a sequence of stock prices
#   based on Brownian motion.

# Input Parameters:
# - S0: Initial stock price.
# - K: Strike price.
# - r: Risk-free interest rate.
# - T: Time to maturity.
# - sigma: Volatility of the underlying asset.
# - option_type: Type of option ('call' or 'put').
# - n: Number of steps for the binomial tree.
# - num_steps: Number of steps for the Brownian motion.

# Black-Scholes Model
def black_scholes(S, K, r, T, sigma, option_type):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Must be 'call' or 'put'.")

    return price

# Binomial Tree Model
def binomial_tree(S, K, r, T, sigma, option_type, n):
    delta_t = T / n
    u = np.exp(sigma * np.sqrt(delta_t))
    d = 1 / u
    p = (np.exp(r * delta_t) - d) / (u - d)

    stock_prices = np.zeros((n + 1, n + 1))
    option_prices = np.zeros((n + 1, n + 1))

    stock_prices[0, 0] = S

    for i in range(1, n + 1):
        stock_prices[i, 0] = stock_prices[i - 1, 0] * u
        for j in range(1, i + 1):
            stock_prices[i, j] = stock_prices[i - 1, j - 1] * d

    if option_type == 'call':
        option_prices[n] = np.maximum(stock_prices[n] - K, 0)
    elif option_type == 'put':
        option_prices[n] = np.maximum(K - stock_prices[n], 0)
    else:
        raise ValueError("Invalid option type. Must be 'call' or 'put'.")

    for i in range(n - 1, -1, -1):
        for j in range(i + 1):
            option_prices[i, j] = np.exp(-r * delta_t) * (p * option_prices[i + 1, j] +
                                                          (1 - p) * option_prices[i + 1, j + 1])

    return option_prices[0, 0]


# Brownian Motion
def brownian_motion(S, r, sigma, T, num_steps):
    delta_t = T / num_steps
    drift = (r - 0.5 * sigma ** 2) * delta_t
    vol = sigma * np.sqrt(delta_t)
    steps = np.random.normal(0, 1, num_steps)
    path = [S]

    for i in range(num_steps):
        price = path[-1] * np.exp(drift + vol * steps[i])
        path.append(price)

    return path

# Calculate option Greeks
def calculate_option_greeks(S, K, r, T, sigma, option_type):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Calculate Delta
    if option_type == 'call':
        delta = norm.cdf(d1)
    elif option_type == 'put':
        delta = norm.cdf(d1) - 1
    else:
        raise ValueError("Invalid option type. Must be 'call' or 'put'.")

    # Calculate Gamma
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))

    # Calculate Theta
    if option_type == 'call':
        theta = - (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        theta = - (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)
    else:
        raise ValueError("Invalid option type. Must be 'call' or 'put'.")

    # Calculate Vega
    vega = S * norm.pdf(d1) * np.sqrt(T)

    # Calculate Rho
    if option_type == 'call':
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        rho = - K * T * np.exp(-r * T) * norm.cdf(-d2)
    else:
        raise ValueError("Invalid option type. Must be 'call' or 'put'.")

    return delta, gamma, theta, vega, rho


# Example usage
S0 = 100  # Initial stock price
K = 100  # Strike price
r = 0.05  # Risk-free interest rate
T = 1  # Time to maturity
sigma = 0.2  # Volatility
option_type = 'call'  # Option type: 'call' or 'put'
n = 1000  # Number of steps for the binomial tree
num_steps = 252  # Number of steps for the Brownian motion

# Calculate option prices using different models
bs_price = black_scholes(S0, K, r, T, sigma, option_type)
bt_price = binomial_tree(S0, K, r, T, sigma, option_type, n)
bm_prices = brownian_motion(S0, r, sigma, T, num_steps)

# Calculate option Greeks using Black-Scholes model
delta, gamma, theta, vega, rho = calculate_option_greeks(S0, K, r, T, sigma, option_type)

# Print the results
print("Black-Scholes Price:", bs_price)
print("Binomial Tree Price:", bt_price)

# Print the option Greeks
print("Option Greeks:")
print("Delta:", delta)
print("Gamma:", gamma)
print("Theta:", theta)
print("Vega:", vega)
print("Rho:", rho)

# Plot Brownian motion prices
time = np.linspace(0, T, num_steps+1)
plt.plot(time, bm_prices)
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Brownian Motion Prices')
plt.grid(True)
plt.show()