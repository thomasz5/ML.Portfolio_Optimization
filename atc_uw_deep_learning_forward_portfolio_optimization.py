"""
Deep Cognitive Frameworks Enhancing Proactive Portfolio Optimization Strategies since 2011
""


import numpy as np # Efficient numerical operations and array manipulations
import matplotlib.pyplot as plt # Creating visualizations such as plots and charts
import tensorflow as tf # Machine learning tasks, particularly deep learning
from pathlib import Path # Working with file system paths
import pandas as pd #Data manipulation and analysis using dataframes
from datetime import datetime # Dates and times
import os # File and directory operations
from cvxopt import matrix,solvers # Convex optimization tasks
solvers.options['show_progress'] = False # Disabling the progress display for optimization solvers to maintain a clean output


"""### Stock Data Preparation
We will use Microsoft, Johnson & Johnson, JPMorgan Chase & Co.,Exxon Mobil Corp stock prices from 2011 to 2020.
"""


# Loading stock prices data from a NumPy file named "example_stocks.npy" into an array
stock_prices = np.load("./example_stocks.npy")

# Plotting the stock prices
plt.plot(stock_prices[:,0], label = "MSFT")
plt.plot(stock_prices[:,1], label = "JNJ")
plt.plot(stock_prices[:,2], label = "JPM")
plt.plot(stock_prices[:,3], label = "XOM")

#plot readability
plt.legend()
plt.title("stock prices (starting from 2011)")

"""### Return
In the context of investments, return signifies how much the investment changed over preemptive time intervals.

Many methologies for coputing returns exist. The most straightforward one involves determining the percentage change in stock prices by dividing the change in stock price by the price in the prededing time interval.

In finanical economics, there's an alternative metric they use called **log return**, which is used to calculate the percentage change, approximately equal to the above one, when the differences between the changes is very minimal.
"""
# Calculating the logarithmic daily returns
MSFT_return = np.log(stock_prices[1:,0]) - np.log(stock_prices[:-1,0])
JNJ_return = np.log(stock_prices[1:,1]) - np.log(stock_prices[:-1,1])
JPM_return = np.log(stock_prices[1:,2]) - np.log(stock_prices[:-1,2])
XOM_return = np.log(stock_prices[1:,3]) - np.log(stock_prices[:-1,3])

# Calculating the mean return across all stocks
mean_return = np.mean([MSFT_return, JNJ_return, JPM_return, XOM_return],axis = 1)

# Plotting the daily log returns
plt.plot(MSFT_return, label = "MSFT",alpha = 0.5)
plt.plot(JNJ_return, label = "JNJ",alpha = 0.5)
plt.plot(JPM_return, label = "JPM",alpha = 0.5)
plt.plot(XOM_return, label = "XOM",alpha = 0.5)
plt.legend()
plt.title("stock daily log return (starting from 2011)")

"""### Mean (ie. Average)

The average of **the return** shows a value we expect the return would change in every time step.
"""

plt.plot(MSFT_return, label = "MSFT")
plt.plot(np.arange(len(MSFT_return)),[np.mean(MSFT_return) for i in np.arange(len(MSFT_return))], color = "red")

"""### Risk Quantification throgh Variance and Standard Deviation

Variance serves as a value indicating how likely the actual returns deviate from their means.
Standard deviation (Std) is the square root of this variance, measuring the dispersion within the returns.

In traditional financial econometric (at least 30 years ago), the prevalent assumption was that the variance (and std) represented the primary risk associated with investing in stocks.
The higher the variance, the higher probability the actual returns deviate from the expected (average or anticipated) return.

However, this approach is not exclusive, and there are more methods to assess investment risk.
Various advanced techniques and metrics other than simple variance and std offer more nuanced and accurate understanding of financial risk.
"""
# Calculating the standard deviation and variance of stocks
np.std(MSFT_return)
np.var(MSFT_return)

"""### Covariance and Correlation

**Covariance** signifies how the change in one return relates to the change in another,

**Correlation** of two returns is a standardized form of covariance (lying between -1 and 1) that offers a more standard perspective bewteen the two returns.

In the example below, I paired return on the same day of JNJ and MSFT. You can see that when the return of MSFT increase, the return of JNJ also increases

So, the return of MSFT and JNJ are positively correlated

"""
# Creating a scatter plot comparing MSFT and JNJ logarithmic daily returns
plt.scatter(MSFT_return, JNJ_return)
plt.xlabel("MSFT return")
plt.ylabel("JNJ return")
plt.title("JNJ return vs MSFT return")

"""### Computing correlation"""

# Computing the logarithmic daily returns
MSFT_return = np.log(stock_prices[1:,0]) - np.log(stock_prices[:-1,0])
JNJ_return = np.log(stock_prices[1:,1]) - np.log(stock_prices[:-1,1])
JPM_return = np.log(stock_prices[1:,2]) - np.log(stock_prices[:-1,2])
XOM_return = np.log(stock_prices[1:,3]) - np.log(stock_prices[:-1,3])

# Calculating the mean return across all stocks
mean_return = np.mean([MSFT_return, JNJ_return, JPM_return, XOM_return],axis = 1)

# Computing the correlation and covariance matrix for daily returns
np.corrcoef([MSFT_return, JNJ_return, JPM_return, XOM_return])
cov = np.cov([MSFT_return, JNJ_return, JPM_return, XOM_return])
cov

"""## Portfolio Optimization

Now, we have two values **mean** (the expected return, expected profit/loss) and **variance** (risk,bacially).

We want to invest in four stocks mentioned above to **get maximum return with minimum risk**.

We have to allocate (to weight) the investment in each stocks, so we get maximum return but minimum variance.

There are many ways to find such weights, eg. Brute Force, random, genetic algorithm.

I use **Quadratic Programming** in this example
"""

# Invest in each stock evenly (with weights (0.25, 0.25, 0.25, 0.25)
w = np.array([0.25,0.25,0.25,0.25])
portfolio_return = w @ mean_return.T
portfolio_variance = w.reshape([-1,4]) @ cov @ w.reshape([-1,4]).T
print("portfolio return: ", portfolio_return)
print("portfolio variance:", portfolio_variance.squeeze())

# Adjusting the investments: 50% in MSFT, 30% in JNJ, 10% in JPM, and 10% in XOM
w = np.array([0.50,0.30,0.10,0.10])
portfolio_return = w @ mean_return.T
portfolio_variance = w.reshape([-1,4]) @ cov @ w.reshape([-1,4]).T
print("portfolio return: ", portfolio_return)
print("portfolio variance:", portfolio_variance.squeeze())

"""Just random 10000 allocation and find the one that gave highest return but smallest risk"""

# Generating random weights for 10,000 portfolios
random_weight = (np.random.rand( 4 * 10000)).reshape([-1,4])
random_weight = np.divide(random_weight.T,np.sum(random_weight,axis = 1)).T

# Calculating returns and variances for each random portfolio
portfolio_return = random_weight @ mean_return.T
portfolio_variance = []
for i in range(len(random_weight )):
    portfolio_variance.append((random_weight[i,:].reshape([-1,4]) @ cov @ random_weight[i,:].reshape([-1,4]).T).squeeze())
portfolio_variance = np.array(portfolio_variance)

# Plotting the Mean-Variance relationship for random portfolios
plt.scatter(portfolio_variance, portfolio_return,s = 1)
plt.xlabel("variance (risk)")
plt.ylabel("expected return")
plt.title("Mean-Variance")

# Calculating the optimal portfolio weight for a given risk level
risk_level = 0.3
P = matrix( cov,tc = 'd')
q = matrix(-risk_level * mean_return, tc = 'd')
G = matrix(-np.eye(4))
h = matrix(np.zeros(4))
A = matrix(np.ones([4]).reshape([-1,4]), tc = 'd')
b = matrix(1.0, tc = 'd')
sol = solvers.qp(P, q, G, h, A, b)
optimal_weight = np.array(sol['x'])
w = optimal_weight.T
optimal_return = w @ mean_return.T
optimal_variance = w.reshape([-1,4]) @ cov @ w.reshape([-1,4]).T

print("optimal weight:", np.round(w * 100))

# Plotting the Mean-Variance relationship for random portfolios and highlighting the optimal portfolio
plt.scatter(portfolio_variance, portfolio_return,s = 1)
plt.scatter(optimal_variance,optimal_return, color = 'red', s = 10)
plt.xlabel("variance (risk)")
plt.ylabel("expected return")
plt.title("Mean-Variance for random portfolio")

"""**Given an expected return we want to make in an investment, we can allocate the portfolio to get smallest risk. Basically, the optimal portfolio given the return we want is on the boundary of the plot above**


**However, consider the boundary, we can see that if we want more return we must take more risk**
So, how much risk you can endure is also an input to portfolio optmimzation

### Portfolio Optimization Strategy

1. We compute the optimal weights for each stock using 40-days back in the past
2. Every 7 days, we rebalance the portfolio using 40-day data.


No slippage, No transaction, No short


Test on stock data from July 2019 til the end of the year
"""

# Loading and plotting the stock prices starting
test_stock = np.load("./test_stock.npy")
plt.plot(test_stock[:,0], label = "MSFT")
plt.plot(test_stock[:,1], label = "JNJ")
plt.plot(test_stock[:,2], label = "JPM")
plt.plot(test_stock[:,3], label = "XOM")
plt.legend()
plt.title("stock prices (starting from July 2019)")

"""



#### Backward-Looking Portfolio Optimization
We use stock prices in the past to optimize a portfolio."""

# Setting parameters for the portfolio and rebalancing strategy
reb_interval = 7
start_date = 10
start_balance = 1
risk_level = 0.08

# Extracting a subset of stock data for the initial analysis
bw_data = test_stock[:40,:]
MSFT_return = np.log(bw_data [1:,0]) - np.log(bw_data[:-1,0])
JNJ_return = np.log(bw_data [1:,1]) - np.log(bw_data[:-1,1])
JPM_return = np.log(bw_data[1:,2]) - np.log(bw_data[:-1,2])
XOM_return = np.log(bw_data [1:,3]) - np.log(bw_data [:-1,3])
mean_return = np.mean([MSFT_return, JNJ_return, JPM_return, XOM_return],axis = 1)
MSFT_mean,JNJ_mean, JPM_mean, XOM_mean = np.mean([MSFT_return, JNJ_return, JPM_return, XOM_return],axis = 1)

# Plotting the daily log returns
plt.plot(MSFT_return, label = "MSFT",alpha = 0.5)
plt.plot(JNJ_return, label = "JNJ",alpha = 0.5)
plt.plot(JPM_return, label = "JPM",alpha = 0.5)
plt.plot(XOM_return, label = "XOM",alpha = 0.5)
plt.legend()
plt.title("stock daily log return (starting from July 2019)")

# Initializing balance and computing initial statistics for the portfolio
bw_balance = start_balance
balance_log = []
# Compute the initial balance
bw_data = test_stock[:40,:]
MSFT_return = np.log(bw_data [1:,0]) - np.log(bw_data[:-1,0])
JNJ_return = np.log(bw_data [1:,1]) - np.log(bw_data[:-1,1])
JPM_return = np.log(bw_data[1:,2]) - np.log(bw_data[:-1,2])
XOM_return = np.log(bw_data [1:,3]) - np.log(bw_data [:-1,3])
mean_return = np.mean([MSFT_return, JNJ_return, JPM_return, XOM_return],axis = 1)
MSFT_mean,JNJ_mean, JPM_mean, XOM_mean = np.mean([MSFT_return, JNJ_return, JPM_return, XOM_return],axis = 1)
cov = np.cov([MSFT_return, JNJ_return, JPM_return, XOM_return])

# Solving for the optimal portfolio weights using convex optimization
P = matrix( cov,tc = 'd')
q = matrix(-risk_level * mean_return, tc = 'd')
G = matrix(-np.eye(4))
h = matrix(np.zeros(4))
A = matrix(np.ones([4]).reshape([-1,4]), tc = 'd')
b = matrix(1.0, tc = 'd')
sol = solvers.qp(P, q, G, h, A, b)
optimal_weight = np.array(sol['x'])
w = optimal_weight.T
print("Investment in [MSFT, JNJ, JPM, XOM] %:",np.around(bw_balance * w.squeeze() * 100))
print("Sum balance:", np.sum(bw_balance * w.squeeze()))

balance_log.append(bw_balance * w.squeeze())

# Iterating through the remaining days, considering rebalancing at set intervals
for i in range(1,len(test_stock)-40):
    today_balance = balance_log[-1]
    if i % reb_interval == 0:
        w = np.array([0,0,1,0])

        bw_balance = np.sum(balance_log[-1])
        bw_data = test_stock[i:40+i,:]
        MSFT_return = np.log(bw_data [1:,0]) - np.log(bw_data[:-1,0])
        JNJ_return = np.log(bw_data [1:,1]) - np.log(bw_data[:-1,1])
        JPM_return = np.log(bw_data[1:,2]) - np.log(bw_data[:-1,2])
        XOM_return = np.log(bw_data [1:,3]) - np.log(bw_data [:-1,3])
        mean_return = np.mean([MSFT_return, JNJ_return, JPM_return, XOM_return],axis = 1)
        MSFT_mean,JNJ_mean, JPM_mean, XOM_mean = np.mean([MSFT_return, JNJ_return, JPM_return, XOM_return],axis = 1)
        cov = np.cov([MSFT_return, JNJ_return, JPM_return, XOM_return])
        P = matrix(cov,tc = 'd')
        q = matrix(-risk_level * mean_return, tc = 'd')
        G = matrix(-np.eye(4))
        h = matrix(np.zeros(4))
        A = matrix(np.ones([4]).reshape([-1,4]), tc = 'd')
        b = matrix(1.0, tc = 'd')
        sol = solvers.qp(P, q, G, h, A, b)
        optimal_weight = np.array(sol['x'])
        w = optimal_weight.T
        print("rebalance: ", int(i/reb_interval))
        print("Investment in [MSFT, JNJ, JPM, XOM]:",np.round(w * 100))
        print("Sum balance:", np.sum(bw_balance * w.squeeze()))
        today_balance = bw_balance * w.squeeze()

    stock_ret = np.log(test_stock[40+i, :]) - np.log(test_stock[40+i-1, :])
    balance_log.append(today_balance * np.exp(stock_ret))

# Extracting data for the final analysis period
bw_data = test_stock[40:,:]
MSFT_return = np.log(bw_data [1:,0]) - np.log(bw_data[:-1,0])
JNJ_return = np.log(bw_data [1:,1]) - np.log(bw_data[:-1,1])
JPM_return = np.log(bw_data[1:,2]) - np.log(bw_data[:-1,2])
XOM_return = np.log(bw_data [1:,3]) - np.log(bw_data [:-1,3])
portfolio_return = np.log(np.sum(np.array(balance_log),axis = 1)[1:]) - np.log(np.sum(np.array(balance_log),axis = 1)[:-1])

# Plotting cumulative growth for all stocks and portfolio
plt.plot(np.cumprod(np.exp(MSFT_return)),label = "MSFT")
plt.plot(np.cumprod(np.exp(JNJ_return)), label = "JNJ")
plt.plot(np.cumprod(np.exp(JPM_return)), label = "JPM")
plt.plot(np.cumprod(np.exp(XOM_return)), label = "XOM")
plt.plot(np.cumprod(np.exp(portfolio_return)), label = "Portfolio")
plt.legend()
plt.title("Growth when investing one dallars in each asset")

plt.plot(1000000*np.cumprod(np.exp(portfolio_return)), label = "Portfolio")
plt.title("Portfolio if invest \$ 1000000")

"""## Deep Learning

### Problems of Traditional Portfolio Optimization

The method I showed above was invented 70 years ago. There are a lot of problems with that method. But there are many methods invented to fix that problem.

One of the problems of that portfolio that it is backward-looking! That is, we use stock prices in the past and present to optimize the portfolio. In this meeting, we will forecast stock prices using machine learning and optimize portfolios using such the prediction.

Remark: I do not like this method.
### Machine Learning

Basically, we have a stock data in the past, and we want to predict the price of the stock in the future. we feed a computer the stock prices in the past and the stock prices in the future of the past (eg feed a stock price of stock in 2000 as input and stock price in 2001 as an output) Machine learning algorithm will fit the model to predict the stock prices in the future

### data preparation
"""

# Creating a dataset and datasets for training, validation, and testing
def create_dataset(stock_price, window_length = 20, shift = 1,valid_p = 0.1,test_p = 0.1):
    ltrain_index = int((1- test_p - valid_p) * len(stock_price))
    lvalid_index = int((1-test_p) * len(stock_price))
    X_train,Y_train,X_valid, Y_valid, X_test,Y_test = list(), list(), list(), list(), list(), list()
    i = 0
    while(i+window_length  + shift <  ltrain_index):
        X_train.append(stock_price[i:i+window_length])
        Y_train.append(stock_price[i+window_length:i+window_length+shift])
        i += 1

    while(i+window_length  + shift < lvalid_index):
        X_valid.append(stock_price[i:i+window_length])
        Y_valid.append(stock_price[i+window_length:i+window_length+shift])
        i += 1

    while(i+window_length  + shift < len(stock_prices)):
        X_test.append(stock_price[i:i+window_length])
        Y_test.append(stock_price[i+window_length:i+window_length+shift])
        i += 1
    return np.array(X_train), np.array(Y_train), np.array(X_valid), np.array(Y_valid), np.array(X_test), np.array(Y_test)

def create_datasets(stock_prices, window_length = 20, shift = 1, valid_p = 0.1, test_p = 0.1):
    X_train,Y_train,X_valid, Y_valid, X_test,Y_test = list(), list(), list(), list(), list(), list()
    for stock_price in stock_prices:
        x_train, y_train, x_valid, y_valid, x_test, y_test = create_dataset(stock_price, window_length, shift, valid_p, test_p)
        X_train.append(x_train)
        Y_train.append(y_train)
        X_valid.append(x_valid)
        Y_valid.append(y_valid)
        X_test.append(x_test)
        Y_test.append(y_test)
    return np.concatenate(X_train),np.concatenate(Y_train),np.concatenate(X_valid), np.concatenate(Y_valid), np.concatenate(X_test), np.concatenate(Y_test)

X_train,Y_train, X_valid, Y_valid, X_test,Y_test = create_datasets([stock_prices[:,0], stock_prices[:,1], stock_prices[:,2], stock_prices[:,3]], window_length = 40, shift = 20 ,test_p = 0.001)

X_train

# Plotting a sample of the training input and output
index = 700
plt.plot(np.concatenate([X_train[index,:],Y_train[index,:] ]), color = "orange",label = "actual output")
plt.plot(X_train[index,:],color = "blue",label = "Input")
plt.legend()

# Normalizing the data
mean_dat = np.mean(np.concatenate([X_train[index,:],Y_train[index] ]))
std_dat = np.std(np.concatenate([X_train[index,:],Y_train[index] ]))

plt.plot((np.concatenate([X_train[index,:],Y_train[index] ]) -mean_dat)/std_dat , color = "orange",label = "actual output")
plt.plot((X_train[index,:]-mean_dat)/std_dat,color = "blue",label = "Input")
plt.legend()

dat = np.concatenate([X_train,Y_train],axis = 1)
mean = np.mean(dat, axis = 1)
std = np.std(dat, axis = 1)
for i in range(len(X_train)):
    X_train[i] = (X_train[i] - mean[i])/std[i]
    Y_train[i] = (Y_train[i] - mean[i])/std[i]

dat = np.concatenate([X_valid,Y_valid],axis = 1)
mean = np.mean(dat, axis = 1)
std = np.std(dat, axis = 1)
for i in range(len(X_valid)):
    X_valid[i] = (X_valid[i] - mean[i])/std[i]
    Y_valid[i] = (Y_valid[i] - mean[i])/std[i]

X_train = np.expand_dims(X_train,axis =2)
Y_train = np.expand_dims(Y_train,axis =2)
X_valid = np.expand_dims(X_valid,axis =2)
Y_valid = np.expand_dims(Y_valid,axis =2)

"""## Model

I use LSTM to predict stock prices
"""

# Building and training the LSTM model
train_layers = [
                tf.keras.layers.LSTM(64, return_sequences=True),
                tf.keras.layers.LSTM(64, return_sequences=True),
                tf.keras.layers.LSTM(32, return_sequences=False),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(20, activation= None)]

model = tf.keras.Sequential(
    train_layers
)

# We use Adam Optimizater and use mean squared error as loss function here.
model.compile(optimizer='adam',
              loss=tf.keras.losses.MeanSquaredError())

#create check_point
cp_id = 0
while os.path.exists("./training_LSTM_"+str(cp_id)):
    cp_id += 1
checkpoint_path = "./training_LSTM_"+str(cp_id)+"/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
# Early stopping to prevent overfitting
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=120)

# Training the model
history  = model.fit(X_train, Y_train, epochs=3, validation_data = (X_valid, Y_valid),callbacks=[es,cp_callback])

# Loading the best weights
model.load_weights("./training_LSTM_0/cp.ckpt")

"""#### deep learning forward-looking portfolio optimization
we use stock prices in the past to optimize a portfolio
"""

# Setting parameters for the portfolio and rebalancing strategy
reb_interval = 7
start_date = 10
start_balance = 1
risk_level = 0.08

# Predicting stock prices for the initial period and comparing with individual stock values
bw_data = test_stock[:40,:]
MSFT_return = np.log(bw_data [1:,0]) - np.log(bw_data[:-1,0])
JNJ_return = np.log(bw_data [1:,1]) - np.log(bw_data[:-1,1])
JPM_return = np.log(bw_data[1:,2]) - np.log(bw_data[:-1,2])
XOM_return = np.log(bw_data [1:,3]) - np.log(bw_data [:-1,3])
mean_return = np.mean([MSFT_return, JNJ_return, JPM_return, XOM_return],axis = 1)
MSFT_mean,JNJ_mean, JPM_mean, XOM_mean = np.mean([MSFT_return, JNJ_return, JPM_return, XOM_return],axis = 1)

# Mean returns of all chosen individual stocks
np.mean(test_stock[:40,:], axis = 0)

# Function to predict stock prices using the trained LSTM model
def predict(input_prices):
    mean = np.mean(input_prices, axis = 0)
    std = np.std(input_prices, axis = 0)
    stock_prices = (input_prices - mean)/std

    msft_predict = model.predict(stock_prices[:,0].reshape([-1,40,1]),verbose = 0).squeeze()
    jnj_predict = model.predict(stock_prices[:,1].reshape([-1,40,1]),verbose = 0).squeeze()
    jpm_predict = model.predict(stock_prices[:,2].reshape([-1,40,1]),verbose = 0).squeeze()
    xom_predict = model.predict(stock_prices[:,3].reshape([-1,40,1]),verbose = 0).squeeze()

    return np.stack([msft_predict, jnj_predict,jpm_predict, xom_predict],axis = 1) * std + mean

# Predicting stock prices for the initial period
predicted_prices = predict(test_stock[:40])
tp = np.vstack([test_stock[:40], predicted_prices])
plt.plot(tp[:,0], label = "MSFT")
plt.plot(tp[:,1], label = "JNJ")
plt.plot(tp[:,2], label = "JPM")
plt.plot(tp[:,3], label = "XOM")
plt.axvline(40,color = "red",alpha = 0.2,linestyle = "dotted")
plt.legend()
plt.title("stock prices (starting from July 2019) with predicted after vertical line")

# Initializing portfolio balances
bw_balance = start_balance
fw_balance_log = []

# Compute the initial balance w/predicted stocks prices
bw_data = predict(test_stock[:40,:])

MSFT_return = np.log(bw_data [1:,0]) - np.log(bw_data[:-1,0])
JNJ_return = np.log(bw_data [1:,1]) - np.log(bw_data[:-1,1])
JPM_return = np.log(bw_data[1:,2]) - np.log(bw_data[:-1,2])
XOM_return = np.log(bw_data [1:,3]) - np.log(bw_data [:-1,3])
mean_return = np.mean([MSFT_return, JNJ_return, JPM_return, XOM_return],axis = 1)
MSFT_mean,JNJ_mean, JPM_mean, XOM_mean = np.mean([MSFT_return, JNJ_return, JPM_return, XOM_return],axis = 1)
cov = np.cov([MSFT_return, JNJ_return, JPM_return, XOM_return])

# Using quadratic programming to find optimal portfolio weights
P = matrix( cov,tc = 'd')
q = matrix(-risk_level * mean_return, tc = 'd')
G = matrix(-np.eye(4))
h = matrix(np.zeros(4))
A = matrix(np.ones([4]).reshape([-1,4]), tc = 'd')
b = matrix(1.0, tc = 'd')
sol = solvers.qp(P, q, G, h, A, b)
optimal_weight = np.array(sol['x'])
w = optimal_weight.T
print("Investment in [MSFT, JNJ, JPM, XOM] %:",np.around(bw_balance * w.squeeze() * 100))
print("Sum balance:", np.sum(bw_balance * w.squeeze()))

fw_balance_log.append(bw_balance * w.squeeze())

# Implementing the rebalancing strategy over the test period
for i in range(1,len(test_stock)-40):
    today_balance = fw_balance_log[-1]
    if i % reb_interval == 0:
        bw_balance = np.sum(fw_balance_log[-1])
        bw_data = predict(test_stock[i:40+i,:])
        MSFT_return = np.log(bw_data [1:,0]) - np.log(bw_data[:-1,0])
        JNJ_return = np.log(bw_data [1:,1]) - np.log(bw_data[:-1,1])
        JPM_return = np.log(bw_data[1:,2]) - np.log(bw_data[:-1,2])
        XOM_return = np.log(bw_data [1:,3]) - np.log(bw_data [:-1,3])
        mean_return = np.mean([MSFT_return, JNJ_return, JPM_return, XOM_return],axis = 1)
        MSFT_mean,JNJ_mean, JPM_mean, XOM_mean = np.mean([MSFT_return, JNJ_return, JPM_return, XOM_return],axis = 1)
        cov = np.cov([MSFT_return, JNJ_return, JPM_return, XOM_return])
        P = matrix(cov,tc = 'd')
        q = matrix(-risk_level * mean_return, tc = 'd')
        G = matrix(-np.eye(4))
        h = matrix(np.zeros(4))
        A = matrix(np.ones([4]).reshape([-1,4]), tc = 'd')
        b = matrix(1.0, tc = 'd')
        sol = solvers.qp(P, q, G, h, A, b)
        optimal_weight = np.array(sol['x'])
        w = optimal_weight.T
        print("rebalance: ", int(i/reb_interval))
        print("Investment in [MSFT, JNJ, JPM, XOM]:",np.round(w * 100))
        print("Sum balance:", np.sum(bw_balance * w.squeeze()))
        today_balance = bw_balance * w.squeeze()

    # Computing daily stock returns and updating the portfolio balance
    stock_ret = np.log(test_stock[40+i, :]) - np.log(test_stock[40+i-1, :])
    fw_balance_log.append(today_balance * np.exp(stock_ret))

# Analyzing portfolio performance and comparing with individual stock values for the forward test period
bw_data = test_stock[40:,:]
MSFT_return = np.log(bw_data [1:,0]) - np.log(bw_data[:-1,0])
JNJ_return = np.log(bw_data [1:,1]) - np.log(bw_data[:-1,1])
JPM_return = np.log(bw_data[1:,2]) - np.log(bw_data[:-1,2])
XOM_return = np.log(bw_data [1:,3]) - np.log(bw_data [:-1,3])

# Calculating portfolio returns and plotting results
bl_portfolio_return = np.log(np.sum(np.array(balance_log),axis = 1)[1:]) - np.log(np.sum(np.array(balance_log),axis = 1)[:-1])
fl_portfolio_return = np.log(np.sum(np.array(fw_balance_log),axis = 1)[1:]) - np.log(np.sum(np.array(fw_balance_log),axis = 1)[:-1])

plt.plot(np.cumprod(np.exp(MSFT_return)),label = "MSFT")
plt.plot(np.cumprod(np.exp(JNJ_return)), label = "JNJ")
plt.plot(np.cumprod(np.exp(JPM_return)), label = "JPM")
plt.plot(np.cumprod(np.exp(XOM_return)), label = "XOM")
plt.plot(np.cumprod(np.exp(bl_portfolio_return )), label = "BL Portfolio")
plt.plot(np.cumprod(np.exp(fl_portfolio_return )), label = "FL Portfolio")
plt.legend()
plt.title("Portfolio and individual stock values over time")

plt.plot(1000000*np.cumprod(np.exp(bl_portfolio_return )), label = "Backward Portfolio",color ="purple")
plt.plot(1000000*np.cumprod(np.exp(fl_portfolio_return )), label = "Forward Portfolio (with LSTM prediction)",color= "brown")
plt.legend()
plt.title("Portfolio if invest \$ 1000000")
