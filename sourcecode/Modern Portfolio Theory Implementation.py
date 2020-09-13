import numpy as np
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import datetime
import scipy.optimize as optimization

stocks = ['AAPL', 'WMT', 'TSLA', 'GE', 'AMZN', 'DB']
start_date = '01/01/2010'
end_date = '01/01/2019'

# Downloading data from Yahoo finance
def download_data(stocks):
    data = web.DataReader(stocks, 'yahoo', start_date, end_date)['Adj Close']
    data.columns = stocks
    return data

def calculate_returns(data):
    returns = np.log(data/data.shift(1))
    return returns
	
def initialize_weights():
    weights = np.random.random(len(stocks))
    weights /= np.sum(weights)
    return weights
	
def generate_portfolios(returns, n):
    preturns = []
    pvariances = []
    
    for i in range (n):
        weights = initialize_weights()
        preturns.append(np.sum(returns.mean()*weights)*252) #add to array total return for each portfolio
        pvariances.append(np.sqrt(np.dot(weights.T, np.dot(returns.cov()*252, weights))))
    
    preturns = np.array(preturns)
    pvariances = np.array(pvariances)
    
    return preturns, pvariances
	
def plot_portfolios(returns, variances):
    plt.figure(figsize=(10,6))
    plt.scatter(variances, returns, c = returns/variances, marker = 'o', cmap = 'Spectral_r', edgecolors='black') # Sharpe Ratio S(x)= (Rx - Rf)/SDx, here no Rf so S = R/SD
    plt.grid(True)
    plt.xlabel("Expected Volatility")
    plt.ylabel("Expected Return")
    plt.colorbar(label = "Sharpe Ratio")
    plt.show()
	
def statistics(returns, weights):
    p_return = np.sum(returns.mean()*weights)*252
    p_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov()*252, weights)))
    return np.array([p_return, p_volatility, p_return/p_volatility])
	
def min_func_sharpe(weights, returns):
    return -statistics(returns, weights)[2]
	
def optimize_portfolio(returns, weights):
    # On doit definir 2 contraintes (constraints & bounds)
    constraints = ({"type": "eq", "fun": lambda x: np.sum(x)-1}) # sum of weights is 1
    bounds = tuple((0,1) for x in range(len(stocks))) # the weights can be 1 at most (1 = 100% money invested in asset)
    
    optimum = optimization.minimize(fun = min_func_sharpe, x0 = weights, args = returns, method = "SLSQP", bounds = bounds, constraints = constraints)
    #fun = min_func_sharpe: function we want to minimize
    #x0 = weights: starting point of algorithm, a guess basically before finding best weights for our portfolio
    #args = returns: min_func_sharpe and statistics function need "returns" as an other parameter to "weights"
    #method = "SLSQP": Sequential Least SQuares Programming, good for our purpose
    
    return optimum
	
	
if __name__ == "__main__":
	data = download_data(stocks)
	returns = calculate_returns(data)
	weights = initialize_weights()
	P = generate_portfolios(returns, 10000)
	plot_portfolios(P[0], P[1])
	a = optimize_portfolio(returns, weights)
	print(a)
	
	main()
	

	
