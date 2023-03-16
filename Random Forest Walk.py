import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
import scipy
from scipy import stats
import requests as request
from bs4 import BeautifulSoup
from datetime import date
from datetime import datetime
from datetime import timedelta
import math
import time
import csv
import pickle
from scipy.stats import uniform as sp_randFloat
from scipy.stats import randint as sp_randInt
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
from sklearn.model_selection import RepeatedKFold
warnings.simplefilter(action='ignore', category=FutureWarning)

#where to save results
filepath = ('C:/Users/jlbat/Desktop/')
start = datetime.now()
#setting up data
timeperiod = "5y"
listedinterval = "1d"
datafilter_days = 600
maximum_starting_spot = 500
maximum_steps = 10

collectedresults = pd.DataFrame(columns = ['Ticker', 'Default Percentage', 'Close Compared Percentage', 'Full Backtested Percentage', 'Close Pred'])


collectedresults = pd.DataFrame(columns = ['Ticker', 'Default Percentage', 'Close Compared Percentage', 'Full Backtested Percentage', 'Close Pred'])

stocks = request.get('https://finance.yahoo.com/losers/?guccounter=1&guce_referrer=aHR0cHM6Ly93d3cuZ29vZ2xlLmNvbS8&guce_referrer_sig=AQAAABwIVuDpYjgE3pZ0XefRHyyhGMQtW_qeJA8STShxLRzxtXwR5Jh9nsKp2ZwDSrveP7Mi61NlBuGGsNCYLWba8d0QIbMve7OMBtzjzI4bbCA84Q6Cm43jCyN57ZzcM2VT9fBioDxjQ5-bmecxgt3_N-S1JRDU5vjxLpz7wgBuAYB1')
soup = BeautifulSoup(stocks.text, 'lxml')
table1 = soup.find('div', id='fin-scr-res-table')
headers = []
for i in table1.find_all('td'):
    title=i.text
    headers.append(title)

headers
len(headers)
cont = 0
dato = pd.DataFrame(headers)
dato2 = dato[::10]
bist = list(dato2[0])
bist2 = []
bist3 =[]
def datalist(x):
    day = yf.Ticker(x).history(period="1d")
    yearcount = yf.Ticker(x).history(period=timeperiod, interval=listedinterval)
    if len(yearcount) > datafilter_days and int(day["Close"]) >= 2:
        bist3.append(x)
        print(x + ': Appended')
    else:
        print(x + ': Too low')

while cont < len(bist):    
    bist2 = (bist[cont].upper())
    datalist(bist2)
    cont = cont + 1  
    if cont == len(bist):
        print("TOTAL SELECTED:", len(bist3))
        break

tickerlist = bist3
print("\n--------------------------------------------")
cnt = 0

for i in tickerlist:
    ticker = tickerlist[cnt].upper()
    stock = yf.Ticker(ticker)
    stock_hist = stock.history(period=timeperiod, interval=listedinterval)
    
    #moving data to find out difference in prices between two days
    stock_prev = stock_hist.copy()
    stock_prev = stock_prev.shift(1)

    ###finding actual close
    data = stock_hist[["Close"]]
    data = data.rename(columns = {'Close':'Actual_Close'})

    ##setup out target
    data["Target"] = stock_hist.rolling(2).apply(lambda x: x.iloc[1] > x.iloc[0])["Close"]

    ###join the data
    predict = ["Close", "Volume", "Open", "High", "Low"]
    data = data.join(stock_prev[predict]).iloc[1:]

    ##create a model
    model = RandomForestClassifier(n_estimators=2000, min_samples_split=10, random_state=1)

    #create a train and test set
    train = data.iloc[:-100]
    test = data.iloc[-100:]

    model.fit(train[predict], train["Target"])

    #error of predicitons
    preds = model.predict(test[predict])
    preds = pd.Series(preds, index=test.index)
    ps_orig = precision_score(test["Target"], preds)
    print("\nTicker:", ticker, "\n--------------------------------------------", "\nBASIC TRAIN SUCCESS %: ", ps_orig, "\n--------------------------------------------")

    #combine predicitons and test values
    combined = pd.concat({"Target": test["Target"],"Predicitions": preds}, axis=1)

    ################################
    #backtesting
    print("Rows:", len(stock_hist))

    startz = maximum_starting_spot

    print("ML Starting Spot:", startz)

    stepz = maximum_steps

    print("Steps:", stepz, "\n--------------------------------------------")

    predictions = []
    prescore = []

####BACK TEST WITH 20% CORRECT
    def backtest1(data, model, predictors, start=startz, step=stepz):
        predictions = []
        # Loop over the dataset in increments
        for i in range(start, data.shape[0], step):
            # Split into train and test sets
            train = data.iloc[0:i].copy()
            test = data.iloc[i:(i+step)].copy()

            # Fit the random forest model
            model.fit(train[predictors], train["Target"])

            # Make predictions
            preds = model.predict_proba(test[predictors])[:,1]
            preds = pd.Series(preds, index=test.index)
            preds[preds > .2] = 1
            preds[preds<=.2] = 0

            # Combine predictions and test values
            combined = pd.concat({"Target": test["Target"],"Predictions": preds}, axis=1)

            predictions.append(combined)

        return pd.concat(predictions)
    data = data.fillna(0)    
    predictions = backtest1(data, model, predict)


    ####value counts
    tail = predictions.tail(1)
    tail = tail.reset_index()
    tail = tail["Predictions"]
    test_predictions = predictions["Predictions"].value_counts()
    real_changes = predictions["Target"].value_counts()
    ps_default = precision_score(predictions["Target"], predictions["Predictions"])
    print("\nDEFAULT PREDICTIONS\n", "\nEstimates:" , test_predictions, "\nReal Data:", real_changes, "\nBack Tested Prediction Estimator:", ps_default, "\nTodays Prediction:", tail, "\n--------------------------------------------")


#### BACK TEST WITH 51% OR HIGHER
    def backtest2(data, model, predictors, start=startz, step=stepz):
        predictions = []
        # Loop over the dataset in increments
        for i in range(start, data.shape[0], step):
            # Split into train and test sets
            train = data.iloc[0:i].copy()
            test = data.iloc[i:(i+step)].copy()

            # Fit the random forest model
            model.fit(train[predictors], train["Target"])

            # Make predictions
            preds = model.predict_proba(test[predictors])[:,1]
            preds = pd.Series(preds, index=test.index)
            preds[preds > .51] = 1
            preds[preds<=.51] = 0

            # Combine predictions and test values
            combined = pd.concat({"Target": test["Target"],"Predictions": preds}, axis=1)

            predictions.append(combined)

        return pd.concat(predictions)
        

    #rolling means/more specific data
    weekly_mean = data.rolling(7).mean()["Close"]
    quarterly_mean = data.rolling(90).mean()["Close"]
    annual_mean = data.rolling(365).mean()["Close"]
    weekly_trend = data.shift(1).rolling(7).sum()["Target"]
    spy = yf.Ticker('SPY')
    daysss = len(stock_hist)
    dayyys = str(daysss) + "d"

    #JOINING IN THE S&P
    sp = spy.history(period=timeperiod, interval=listedinterval)
    data["weekly_mean"] = weekly_mean / data["Close"]
    data["quarterly_mean"] = quarterly_mean / data["Close"]
    data["annual_mean"] = annual_mean / data["Close"]
    data["annual_weekly_mean"] = data["annual_mean"] / data["weekly_mean"]
    data["annual_quarterly_mean"] = data["annual_mean"] / data["quarterly_mean"]
    data["weekly_trend"] = weekly_trend
    data["open_close_ratio"] = data["Open"] / data["Close"]
    data["high_close_ratio"] = data["High"] / data["Close"]
    data["low_close_ratio"] = data["Low"] / data["Close"]
    sp = sp.rename(columns = {'Close':'SP CLOSE'})
    sp = sp["SP CLOSE"]
    data = data.join(sp).iloc[1:]
    sp_weekly_mean = data.rolling(7).mean()["SP CLOSE"]
    data["sp_weekly_mean"] = sp_weekly_mean
    data = data.fillna(0)
    
    ####### LINEAR REG  
    data2 = data.drop(['Actual_Close', 'Target', 'quarterly_mean', 'annual_mean', 'annual_weekly_mean', 'annual_quarterly_mean', 'weekly_trend'], axis=1)
    data2 = data2.drop(['open_close_ratio', 'high_close_ratio', 'low_close_ratio', 'sp_weekly_mean', 'High', 'Low', 'weekly_mean'], axis=1)

    axisvalues=list(range(1,len(data2.columns)+1))
    def calc_slope(row):
        a = scipy.stats.linregress(row, y=axisvalues)
        return pd.Series(a._asdict())
    
    data2=data2.apply(calc_slope,axis=1)

    data2 = data2.drop(['intercept', 'rvalue', 'pvalue', 'intercept_stderr'], axis=1)
    
    data = data.join(data2)
    
    #ADD IN THE NEW PREDICTORS
    full_predictors = ["Actual_Close", "Close", "Volume", "weekly_mean", "SP CLOSE", "sp_weekly_mean", "stderr"]
    
    #NEW BACKTEST
    predictions = backtest2(data.iloc[365:], model, full_predictors)
    tail = predictions.tail(1)
    tail = tail.reset_index()
    tail = tail["Predictions"]
    test_predictions = predictions["Predictions"].value_counts()
    real_changes = predictions["Target"].value_counts()
    ps = precision_score(predictions["Target"], predictions["Predictions"])
    print("\nALL FACTORS INCLUDED\n", "\nEstimates:" , test_predictions, "\nReal Data:", real_changes, "\nBack Tested Prediction Estimator:", ps, "\nTodays Prediction:", tail, "\n--------------------------------------------")
    collectedresults = collectedresults.append({'Ticker' : ticker, 'Default Percentage' : ps_orig, 'Close Compared Percentage' : ps_default, 'Full Backtested Percentage' : ps, 'Close Pred' : tail}, ignore_index = True)

    cnt = cnt + 1
print("\n--------------------------------------------\n", collectedresults, "\n--------------------------------------------")

print("Full Runtime: ", datetime.now()-start, "\n --------------------------------------------")
collectedresults.to_csv(filepath + str(now_) + '.csv', index=False)
print("Results Printed")