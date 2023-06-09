import os
import yfinance as yf

# DO THIS FIRST !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# This code fetches historical stock data from Yahoo Finance for a specified time frame and saves it to Excel files.
# It prompts the user to enter the start and end dates for the data retrieval.
# The user is also prompted to enter the ticker symbols for the stocks of interest.
# The code creates a folder named "Stock Data" if it doesn't exist already.
# For each ticker symbol, it fetches the historical data from Yahoo Finance using the yfinance library.
# The data is then saved to an Excel file named "{ticker_symbol}_stock_data.xlsx" in the "Stock Data" folder.
# After saving the data, the code displays a message indicating the successful save of the stock data.

# Create a folder named "Stock Data" if it doesn't exist
folder_name = "Stock Data"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# Prompt the user for the time frame
start_date = input("Enter the start date (YYYY-MM-DD): ")
end_date = input("Enter the end date (YYYY-MM-DD): ")

# Prompt the user for the ticker symbols
ticker_symbols = input("Enter the ticker symbols (separated by spaces): ").split()

# Fetch and save data for each ticker symbol
for i, ticker_symbol in enumerate(ticker_symbols):
    # Fetch the data from Yahoo Finance
    data = yf.download(tickers=ticker_symbol, start=start_date, end=end_date)

    # Save the data to an Excel file
    file_name = f"{ticker_symbol}_stock_data.xlsx"
    file_path = os.path.join(folder_name, file_name)
    data.to_excel(file_path)

    print(f"Stock data for {ticker_symbol} saved to {file_path}.")