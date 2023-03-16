import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
import scipy
from scipy import stats
import requests as request
from bs4 import BeautifulSoup
from datetime import datetime
import time
import csv
import pickle
from scipy.stats import uniform as sp_randFloat
from scipy.stats import randint as sp_randInt
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



#where to save results
filepath = ('C:/Users/Will/OneDrive/Desktop/')


period = "max"


start = datetime.now()
datafilter_days = 400
### DATA

tickerlist = ['XLE', 'XLB', 'XLI', 'XLY', 'XLP', 'XLV', 'XLF', 'SMH', 'XTL', 'XLU', 'IYR']

print('Tickers: ' + str(tickerlist))
def optimizer():
    print("--------------------------------------------")
    will = 0
    for i in tickerlist:
        startfunction = datetime.now()
        ticker = tickerlist[will].upper()
        stock = yf.Ticker(ticker)
        stock_hist = stock.history(period=period, interval="1wk")
        
        #moving elizabeth to find out difference in prices between two days
        stock_prev = stock_hist.copy()
        stock_prev = stock_prev.shift(1)

        ###finding actual close
        elizabeth = stock_hist[["Close"]]
        elizabeth = elizabeth.rename(columns = {'Close':'Actual_Close'})

        ##setup out target
        elizabeth["Target"] = stock_hist.rolling(2).apply(lambda x: x.iloc[1] > x.iloc[0])["Close"]

        ###join the elizabeth
        predict = ["Close", "Volume", "Open", "High", "Low"]
        elizabeth = elizabeth.join(stock_prev[predict]).iloc[1:]
        


        elizabeth["open_close_ratio"] = elizabeth["Open"] / elizabeth["Close"]
        elizabeth["high_close_ratio"] = elizabeth["High"] / elizabeth["Close"]
        elizabeth["low_close_ratio"] = elizabeth["Low"] / elizabeth["Close"]

        
        ####### LINEAR REG  
        elizabeth2 = elizabeth.drop(['Actual_Close', 'Target'], axis=1)
        elizabeth2 = elizabeth2.drop(['open_close_ratio', 'high_close_ratio', 'low_close_ratio', 'High', 'Low'], axis=1)

        axisvalues=list(range(1,len(elizabeth2.columns)+1))
        
        def calc_slope(row):
            a = scipy.stats.linregress(row, y=axisvalues)
            return pd.Series(a._asdict())
        
        elizabeth2=elizabeth2.apply(calc_slope,axis=1)

        elizabeth2 = elizabeth2.drop(['intercept', 'rvalue', 'pvalue', 'intercept_stderr'], axis=1)
        
        elizabeth = elizabeth.join(elizabeth2)

        ### DROP FACTORS
        #print(elizabeth.columns)

        ### OPTIMIZING PARAMETERS FOR MODEL
        model = ElasticNet(fit_intercept=True)
        
        y = elizabeth['Actual_Close']
        X = elizabeth.drop(['Actual_Close'], axis=1)
        
        #make training set - 25% test data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)


        parameters = {
                          'max_iter'    : sp_randInt(500, 10000),
                          'tol'         : sp_randFloat()

                         }
        print("-------------------------------------------- \nOptimization started for:", ticker)
        
        cross_validation = RepeatedKFold(n_splits=5, n_repeats=10, random_state =1)
        
        randm_src = RandomizedSearchCV(estimator=model, param_distributions = parameters,
                                       cv = cross_validation, n_iter = 75, verbose = 1, n_jobs=-1, random_state=1)
        
        randm_src.fit(X_train, y_train)
        print("Optimization complete for:", ticker)
        print("Model fit for: ", ticker + "\n----------------")
        
        
        y_pred = randm_src.predict(X_test)

        print("Model statistics for:" + ticker + "\n----------------")
        print('Model Score:', randm_src.score(X_test, y_test))
        print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
        print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
        print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred)))

        #save the model
        with open('model_' + ticker + '.pkl', 'wb') as f:
            pickle.dump(randm_src,f)
        print("model", ticker, "saved")
        print("\nRuntime for:", ticker, datetime.now()-startfunction, "\n--------------------------------------------")
        will = will + 1
    global optiruntime
    optiruntime = (datetime.now() - startfunction)

  

def guesser():
    will = 0
    startguesser = datetime.now()
    william = pd.DataFrame(columns = ['Ticker', 'Last Weeks Predicted Close', 'Last Weeks Close', 'Next Weeks Predicted Close', 'Difference'])
    for i in tickerlist:
        ticker = tickerlist[will].upper()
        stock = yf.Ticker(ticker)
        stock_hist = stock.history(period=period, interval="1wk")
        
        #moving elizabeth to find out difference in prices between two days
        stock_prev = stock_hist.copy()
        stock_prev = stock_prev.shift(1)

        ###finding actual close
        elizabeth = stock_hist[["Close"]]
        elizabeth = elizabeth.rename(columns = {'Close':'Actual_Close'})

        ##setup out target
        elizabeth["Target"] = stock_hist.rolling(2).apply(lambda x: x.iloc[1] > x.iloc[0])["Close"]

        ###join the elizabeth
        predict = ["Close", "Volume", "Open", "High", "Low"]
        elizabeth = elizabeth.join(stock_prev[predict]).iloc[1:]
        
        #rolling means/more specific elizabeth
        weekly_mean = elizabeth.rolling(7).mean()["Close"]
        quarterly_mean = elizabeth.rolling(90).mean()["Close"]
        annual_mean = elizabeth.rolling(365).mean()["Close"]
        weekly_trend = elizabeth.shift(1).rolling(7).sum()["Target"]
        spy = yf.Ticker('SPY')
        daysss = len(stock_hist)
        dayyys = str(daysss) + "d"

        #JOINING IN THE S&P

        elizabeth["open_close_ratio"] = elizabeth["Open"] / elizabeth["Close"]
        elizabeth["high_close_ratio"] = elizabeth["High"] / elizabeth["Close"]
        elizabeth["low_close_ratio"] = elizabeth["Low"] / elizabeth["Close"]

        
        ####### LINEAR REG  
        elizabeth2 = elizabeth.drop(['Actual_Close', 'Target'], axis=1)
        elizabeth2 = elizabeth2.drop(['open_close_ratio', 'high_close_ratio', 'low_close_ratio', 'High', 'Low'], axis=1)

        axisvalues=list(range(1,len(elizabeth2.columns)+1))
        
        def calc_slope(row):
            a = scipy.stats.linregress(row, y=axisvalues)
            return pd.Series(a._asdict())
        
        elizabeth2=elizabeth2.apply(calc_slope,axis=1)

        elizabeth2 = elizabeth2.drop(['intercept', 'rvalue', 'pvalue', 'intercept_stderr'], axis=1)
        
        elizabeth = elizabeth.join(elizabeth2)

    #open model
        with open('model_' + ticker + '.pkl', 'rb') as f:
            model = pickle.load(f)
        elizabeth2 = elizabeth
        elizabeth = elizabeth.drop(['Actual_Close'], axis=1)
        y_pred = model.predict(elizabeth.tail(2))
        y_pred_fixed = np.delete(y_pred, 1)
        y_pred_tmrw = model.predict(elizabeth.tail(1))
        

        
        close_prev = elizabeth.copy()
        yest_close_fixed = close_prev.tail(1)["Close"]
        
        william = william.append({'Ticker' : ticker,'Last Weeks Predicted Close' : float(y_pred_fixed), 'Last Weeks Close' : float(yest_close_fixed),   'Next Weeks Predicted Close' : float(y_pred_tmrw), 'Difference' : float(float(y_pred_tmrw) - elizabeth2.tail(1)["Actual_Close"])}, ignore_index = True)
        print(ticker, "close predicted \n--------------------------------------------")
        will = will + 1
        
        

    print(william)
    now_ = time.strftime("%m_%d_%Y")
    william.to_csv(filepath + str(now_) + ' Weekly ETF Prediciton' + '.csv', index=False)
    global guessertime
    guessertime = (datetime.now()-startguesser) 
    print("-------------------------------------------- \nResults Printed and Saved in OneDrive")


optimizer()
guesser()


print("Optimization Runtime: ", optiruntime)
print("Guesser Runtime: ", guessertime)  
print("Full Runtime: ", datetime.now()-start, "\n --------------------------------------------")




