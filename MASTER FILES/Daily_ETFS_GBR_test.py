import os
import time
from datetime import datetime
import csv

import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
import scipy
from scipy import stats
from scipy.stats import uniform as sp_randFloat
from scipy.stats import randint as sp_randInt

from sklearn.metrics import precision_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import requests as request
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#^^^ LIBRARIES AND PACKAGES ^^^#
###############################################################################################
#vvv CODEBASE vvv#

#where to save results
start = datetime.now()
filepath_linux = ('C:/Users/Admin/OneDrive/Coding/Predictions/GBR/')
filepath_windows = ('C:/Users/Will/OneDrive/Coding/Predictions/GBR/')

#basic inputs
datafilter_days = 400
period = "700d"

#tickers
tickerlist = ['XLE',

              ]
#print tickerlist
print('--------------------------------------------\n' 'Tickers: ' + str(tickerlist) + '\n--------------------------------------------\n')

def data_collector():
    ticker = tickerlist[cnt].upper()
    stock = yf.Ticker(ticker)
    global stock_hist
    stock_hist = stock.history(period=period, interval="1d")

    ###moving data to find out difference in prices between two days###
    stock_prev = stock_hist.copy()
    stock_prev = stock_prev.shift(1)

    ###finding actual close###
    data = stock_hist[["Close"]]
    data = data.rename(columns = {'Close':'Actual_Close'})

    ###setup our target###
    data["Target"] = stock_hist.rolling(2).apply(lambda x: x.iloc[1] > x.iloc[0])["Close"]

    ###join the data###
    predict = ["Close", "Volume", "Open", "High", "Low"]
    data_prev = data.join(stock_prev[predict]).iloc[0:]
    data_today = data.join(stock_hist[predict]).iloc[0:]
 
    ###adding in more data
    data_full = moredata(data_prev)
    data_today = moredata(data_today)

    ###removing features
    data_full = data_remove(data_full)
    data_today = data_remove(data_today)

    return data_full, data_today, ticker
    
def moredata(data_alter):
    ###rolling means/more specific data###
    weekly_mean = data_alter.rolling(7).mean()["Close"]
    quarterly_mean = data_alter.rolling(90).mean()["Close"]
    annual_mean = data_alter.rolling(365).mean()["Close"]
    weekly_trend = data_alter.shift(1).rolling(7).sum()["Target"]
    spy = yf.Ticker('SPY')
    daysss = len(stock_hist)
    dayyys = str(daysss) + "d"

    ###JOINING IN THE S&P###
    sp_period = len(data_alter) + 1
    sp = spy.history(period=str(sp_period) + "d", interval="1d")
    data_alter["weekly_mean"] = weekly_mean / data_alter["Close"]
    data_alter["quarterly_mean"] = quarterly_mean / data_alter["Close"]
    data_alter["annual_mean"] = annual_mean / data_alter["Close"]
    data_alter["annual_weekly_mean"] = data_alter["annual_mean"] / data_alter["weekly_mean"]
    data_alter["annual_quarterly_mean"] = data_alter["annual_mean"] / data_alter["quarterly_mean"]
    data_alter["weekly_trend"] = weekly_trend
    data_alter["open_close_ratio"] = data_alter["Open"] / data_alter["Close"]
    data_alter["high_close_ratio"] = data_alter["High"] / data_alter["Close"]
    data_alter["low_close_ratio"] = data_alter["Low"] / data_alter["Close"]
    sp = sp.rename(columns = {'Close':'SP CLOSE'})
    sp = sp["SP CLOSE"]
    data_alter = data_alter.join(sp).iloc[1:]
    sp_weekly_mean = data_alter.rolling(7).mean()["SP CLOSE"]
    data_alter["sp_weekly_mean"] = sp_weekly_mean
    data_alter = data_alter.fillna(0)
    return data_alter

def data_remove(x):
    x = x.drop(['Low'], axis=1)
    return x
    
def optimizer():
    global cnt
    cnt = 0
    for i in tickerlist:
        startfunction = datetime.now()
        data_full, data_today, ticker = data_collector()

        #model and data
        model = GradientBoostingRegressor()
        y = data_full['Actual_Close']
        X = data_full.drop(['Actual_Close'], axis=1)
        
        #make training set - 25% test data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

        #parameters to be tuned
        parameters = {
                          'learning_rate' : sp_randFloat(),
                          'subsample' : sp_randFloat(),
                          'n_estimators' : sp_randInt(100, 500),
                          'min_samples_split' : sp_randInt(2, 6),
                          'min_samples_leaf' : sp_randInt(1, 3),
                          'max_depth'    : sp_randInt(2, 8)

                         }
        print("-------------------------------------------- \nOptimization for:", ticker)  
        randm_src = RandomizedSearchCV(estimator=model, param_distributions = parameters,
                                       cv = 5, n_iter = 10, verbose = 1, n_jobs=-1, random_state=1)
        
        randm_src.fit(X_train, y_train)
        print("Optimization complete for:", ticker)
        print("Model fit for:", ticker + "\n----------------")
        
        #prediction and model statistics
        y_pred = randm_src.predict(X_test)
        print("Model statistics for:" + ticker + "\n----------------")
        print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
        print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
        print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred)))

        ###SAVE THE MODEL###
        with open('model_' + ticker + '.pkl', 'wb') as f:
            pickle.dump(randm_src,f)
        print("model", ticker, "saved")
        print("\nRuntime for:", ticker, datetime.now()-startfunction, "\n--------------------------------------------")
        cnt = cnt + 1
        
    global optiruntime
    optiruntime = (datetime.now() - startfunction)

def guesser():
    global cnt
    cnt = 0
    startguesser = datetime.now()
    final_df = pd.DataFrame(columns = 
        ['Date and hour',
        'Ticker', 
        'Yesterdays Predicted Close', 
        'Yesterdays Close', 
        'Todays Predicted Close', 
        'Todays Actual Close', 
        'Difference', 
        'Tomorrows Predicted Close', 
        'Sentiment Score'
        ])
    
    for i in tickerlist:
        data, data_today, ticker = data_collector()
    
        ###open model###
        with open('model_' + ticker + '.pkl', 'rb') as f:
            model = pickle.load(f)
            
        #todays predictions
        data_full = data.drop(['Actual_Close'], axis=1)
        y_pred = model.predict(data_full.tail(2))
        y_pred_fixed = np.delete(y_pred, 1)
        y_pred_tmrw = model.predict(data_full.tail(1))
        close_prev = data_full.copy()
        yest_close_fixed = close_prev.tail(1)["Close"]
        
        #tomorrows predictions
        data_today = data_today.drop(['Actual_Close'], axis=1)
        y_pred_tmrw_tmrw = model.predict(data_today.tail(1))
        
        #sentiment analysis
        averagescore = sentiment(ticker)
    
    final_df = final_df.append(
        {'Date and hour' :     time.strftime("%m_%d_%Y_%H"),
        'Ticker' : ticker,
        'Yesterdays Predicted Close' : round(float(y_pred_fixed), 2),
        'Yesterdays Close' : round(float(yest_close_fixed), 2),
        'Todays Predicted Close' : round(float(y_pred_tmrw), 2),
        'Todays Actual Close' : round(float(data.tail(1)["Actual_Close"]), 2),
        'Difference' : round(float(float(y_pred_tmrw) - data.tail(1)["Actual_Close"]), 2),
        'Tomorrows Predicted Close' : round(float(y_pred_tmrw_tmrw), 2),
        'Sentiment Score' : averagescore},
        ignore_index = True)
    print(ticker, "close predicted \n--------------------------------------------")
    cnt = cnt + 1
        
    return final_df, startguesser
    
def sentiment(ticker):
    ###SENTIMENT ANALYSIS###
    web_url = 'https://finviz.com/quote.ashx?t='
    news_tables = {}
    tickers = ticker
    url = web_url + ticker
    req = Request(url=url,headers={"User-Agent": "Chrome"}) 
    response = urlopen(req)    
    html = BeautifulSoup(response,"html.parser")
    news_table = html.find(id='news-table')
    news_tables[ticker.upper()] = news_table 
    snews = news_tables[ticker]
    snews_tr = snews.findAll('tr')
    for x, table_row in enumerate(snews_tr):
        a_text = table_row.a.text
        td_text = table_row.td.text
        if x == 3:
            break
            
    news_list = []
    for file_name, news_table in news_tables.items():
        for i in news_table.findAll('tr'):                
            try:
                text = i.a.get_text() 
            except AttributeError:
                print('')      
                
            datex_scrape = i.td.text.split()
            if len(datex_scrape) == 1:
                timex = datex_scrape[0]                  
            else:
                datex = datex_scrape[0]
                timex = datex_scrape[1]

            tick = file_name.split('_')[0]                
            news_list.append([tick, datex, timex, text])          
    vader = SentimentIntensityAnalyzer()
    columns = ['ticker', 'date', 'time', 'headline']
    news_df = pd.DataFrame(news_list, columns=columns)
    scores = news_df['headline'].apply(vader.polarity_scores).tolist()
    scores_df = pd.DataFrame(scores)
    news_df = news_df.join(scores_df, rsuffix='_right')
    news_df['date'] = pd.to_datetime(news_df.date).dt.date
    mean_scores = news_df.groupby(['ticker','date']).mean()
    mean_scores = mean_scores.unstack()
    mean_scores = mean_scores.xs('compound', axis="columns").transpose()
    mean_scores = mean_scores.tail(3)
    mean_scores = mean_scores.mean()
    averagescore = float(mean_scores)
    return averagescore

def estimator_and_printer():
    now_ = time.strftime("%m_%d_%Y_%H")
    final_df, startguesser = guesser()
    print(final_df)
    
    try:
        final_df.to_csv((filepath_linux) + now_ + ' Daily ETF Prediction' + '.csv', index=False)
    except OSError:
        final_df.to_csv((filepath_windows) + now_ + ' Daily ETF Prediction' + '.csv', index=False)
    except OSError:
        final_df.to_csv(now_ + ' Daily ETF Prediction' + '.csv', index=False)
   
    global guessertime 
    guessertime = (datetime.now()-startguesser) 
    print("-------------------------------------------- \nResults Printed and Saved in OneDrive")


optimizer()
estimator_and_printer()

print("Optimization Runtime: ", optiruntime)
print("Guesser Runtime: ", guessertime)  
print("Full Runtime: ", datetime.now()-start, "\n--------------------------------------------")




