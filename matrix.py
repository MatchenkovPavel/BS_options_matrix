import math
import numpy as np
import yfinance as yf
from math import log,sqrt
from datetime import date
import requests
import pandas as pd
from scipy.stats import norm



def get_data_from_yf(date_from, date_to):
    ticker = 'BTC-USD'
    # get the data for instrument
    data = yf.download(ticker, date_from, date_to)
    data_frame = pd.DataFrame(data)
    prices = []
    for day in range(1, len(data_frame)):
        prices.append(data_frame.iloc[day]['Close'])
    return prices


def history_volatility_calculation(prices):
    relative_price = []
    for day in range(1, len(prices)):
        relative_price.append((prices[day] / prices[day - 1] - 1))
    mathematical_expectation = np.mean(relative_price)

    squared_deviation = 0
    for price in relative_price:
        squared_deviation += (price - mathematical_expectation) ** 2
    squared_deviation = np.mean(squared_deviation)

    # standard deviation of price changes
    a = sqrt(squared_deviation / (len(prices) - 1))
    volatility = round(a * sqrt(365), 4)
    return volatility


def get_current_price():
    url = 'https://api.coingecko.com/api/v3/coins/bitcoin?tickers=false&market_data=true&community_data=false&developer_data=false&sparkline=false'
    response = requests.get(url).json()
    return (response['market_data']['current_price']['usd'])


def h(E, U, v, t):
    return ((log(U / E) + ((v ** 2) / 2) * t) / (v * sqrt(t)))


def black_scholes_call(E,t):
    return round((U*(math.e**(-r*t))*norm.cdf(h(E,U,v,t)))-E*(math.e**(-r*t))*norm.cdf(h(E,U,v,t)-v*sqrt(t)),3)


def black_scholes_put(E,t):
    return round((-U*(math.e**(-r*t))*norm.cdf(-h(E,U,v,t)))+E*(math.e**(-r*t))*norm.cdf(v*sqrt(t)-h(E,U,v,t)),3)


def create_call_value_matrix(prices):
    # Create an empty array to hold the estimated option values
    call_output_array = np.zeros((len(prices), number_of_days)) #rows-price for the strike , columns - days to expiration

    # For each possible price...
    for i, strike in enumerate(prices):
        # ...and each day from now to expiration
        for d in range(number_of_days):
            # Get the Black-Scholes data for the contract assuming the
            # given price and days until expiration
            call_price = black_scholes_call(strike,(number_of_days - d)/365)
            # create matrix of call prices
            call_output_array[i, d] = round(call_price, 1)

    call_matrix = pd.DataFrame(call_output_array,index=prices)
    return call_matrix

def create_put_value_matrix(prices):
    # Create an empty array to hold the estimated option values
    put_output_array = np.zeros((len(prices), number_of_days)) #rows-price for the strike , columns - days to expiration

    # For each possible share price...
    for i, strike in enumerate(prices):
        # ...and each day from now to expiration
        for d in range(number_of_days):
            # Get the Black-Scholes data for the contract assuming the
            # given price and days until expiration
            put_price = black_scholes_put(strike,(number_of_days - d)/365)
            # create matrix of put prices
            put_output_array[i, d] = round(put_price, 1)

    put_matrix = pd.DataFrame(put_output_array,index=prices)
    return put_matrix


# set the data :
# period for calcul volatility
date_from, date_to = ('2021-12-09', '2022-04-06')

# strike price
E = float(40000)

#strike prices in the matrix
range_of_price = np.arange(E - 10000, E + 15000,5000)

#current_price
U = get_current_price()

# get volatility(implied_volatility) from def
get_prices = get_data_from_yf(date_from,date_to)
v = history_volatility_calculation(get_prices)

# set time and get number_of_days for matrix
t = float((date.fromisoformat('2022-04-29')-date.today()).days / 365)
number_of_days = (date.fromisoformat('2022-04-29')-date.today()).days

#free risk rate
r = 0

call_matrix = create_call_value_matrix(range_of_price)
put_matrix = create_put_value_matrix(range_of_price)
call_matrix.to_csv('Call_matrix.csv')
put_matrix.to_csv('Put_matrix.csv')
#print(call_matrix)
#print(put_matrix)



