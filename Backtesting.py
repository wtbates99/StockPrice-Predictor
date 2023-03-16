import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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


#start runtime
global start
start = datetime.now()

#where to save results
filepath_to_csv = ('C:/Users/Will/OneDrive/Coding/Predictions/25_LOSERS/')

global period
perioddays = 800
period = "800d"
global interval
interval = "1d"


start = datetime.now()
datafilter_days = perioddays + 5
### DATA
table=pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
df = table[0]
tickerlist = list(df["Symbol"])
#remove symbols that dont work
tickerlist.remove('BRK.B')
tickerlist.remove('BF.B')
print("Amount of Stocks Collected: ", len(tickerlist))
print("\n--------------------------------------------")
stocklist = tickerlist
today = date.today()
long_MA = 200
short_MA = 17
start_date =  today - timedelta(perioddays)
end_date = today
initial_wealth = 1000

counter = 0
collectcounter = 0
today = date.today()


passablestocks = []

def get_stock_data(stock,startdate,enddate,period,interval):
        ticker = stock  
        yf.pdr_override()
        df = yf.download(tickers=stock, start=startdate, end=enddate, interval=interval,period=period)
        df.reset_index(inplace=True) 
        df['date'] = df['Date']
      
        return df
      

def ma_strategy(df,short_MA,long_MA):
    df['long_MA'] = df['Close'].rolling(int(long_MA)).mean()
    df['short_MA'] = df['Close'].rolling(int(short_MA)).mean()
    df['crosszero'] = np.where(df['short_MA'] < df['long_MA'], 1.0, 0.0)
    df['position'] = df['crosszero'].diff()
    df['position'].iloc[-1] = -1
    for i, row in df.iterrows():
        if df.loc[i,'position'] == 1 :
                buy_price = round(df.loc[i,'Close'],2)
                df.loc[i,'buy'] = buy_price
        if df.loc[i,'position'] == -1 :
                sell_price = round(df.loc[i,'Close'],2)
                df.loc[i,'sell'] = sell_price
    return df


def buy_sell_signals(df,stock,start_date,end_date):
    
        totalprofit = 0
        print('Stock: {}'.format(stock))
        print('Period: {} - {}'.format(start_date, end_date))
        print('-'*57)
        print('{:^7}{:^10}{:^15}{:^10}{:^15}'.format('S/N','Buy Date','Buy Price($)','Sell Date','Sell Price($)'))
        print('-'*57)

        for i, row in df.iterrows():
                        if df.loc[i,'position'] == 1 :
                                buy_price = round(df.loc[i,'buy'],2)

                                if buy_price:
                                    buydate = df.loc[i,'Date']
                                    if df.loc[i,'position'] == -1 :
                                        sell_price = round(df.loc[i,'sell'],2)
                                        selldate  = df.loc[i,'Date']
                                        profit = sell_price - buy_price
                                        profit = round(profit,2)
                                        totalprofit = totalprofit + profit
                                        totalprofit = round(totalprofit,2)
                                        print('{:^7}{}{:^15}{}{:^15}'.format(i,buydate,buy_price,selldate,sell_price))
                                else:
                                    print('No Buy Price')

        return df



def backtest(df,stock,startdate,enddate,initial_wealth) :
        # assumptions:
        initial_wealth = int(initial_wealth)
        profitloss = 0 
        position = 0
        total_profit = 0 
        qty = 0
        balance = initial_wealth
        buy_p = 0 # per share 
        total_buy_p = 0
        total_sell_p = 0 
        MA_wealth = initial_wealth # moving average wealth
        LT_wealth = initial_wealth # long-term wealth
        inital_sell = 0 
        df['position'].iloc[-1] = -1
                
    

        print('Stock: {}'.format(stock))
        print('Period: {} - {}'.format(startdate, enddate))
        print('Initial Wealth: {}'.format(initial_wealth))
        print('-'*100)
        print('{:^7}{:^15}{:^10}{:^15}{:^20}{:^20}{:^10}{:^20}{:^20}{:^20}{:^20}'.format('Sr. No','Buy Date','Buy Price($)','Sell Date','Sell Price($)','Investment($)','Qty','total_buy_p','total_sell_p','profitloss','MA_wealth'))
                                                                              
        print('-'*100)
        for i,row in df.iterrows():
            if position == 0:
                if df.loc[i,'position'] == 1:
                    buy_p =round( df.loc[i,'Close'],2)
                    buy_d = df.loc[i,'Date']
                    balance = balance + total_sell_p
                    qty = balance / buy_p
                    qty = math.trunc(qty)
                    total_buy_p = round(buy_p * qty,2)
                    balance = balance - total_buy_p 
                    position = 1       
                else:
                    price = df.loc[i,'Close'] 
                    if qty == 0 and MA_wealth == initial_wealth:
                        df.loc[i,'MA_wealth'] = balance
                    elif qty != 0 and MA_wealth != initial_wealth:
                        MA_wealth = sell_balance
                        df.loc[i,'MA_wealth'] = MA_wealth 
            elif position == 1:
                if df.loc[i,'position'] == -1:
                    sell_p = round(df.loc[i,'Close'],2)
                    sell_d = df.loc[i,'Date']
                  
                    total_sell_p = round(sell_p * qty,2)
                    profitloss = round(total_sell_p - total_buy_p,2)
                    total_profit = round(total_profit + profitloss,2)
                    sell_balance = round(balance + total_profit,2)
                    MA_wealth = round(balance + total_sell_p,2)
                    balance = round(balance,2)
                   
                    print('{:^7}{}{:^15}{}{:^15}{:^15}{:^15}{:^20}{:^20}{:^10}{:^10}'.format(i,buy_d,buy_p,sell_d,sell_p,MA_wealth,qty,total_buy_p,total_sell_p,profitloss,MA_wealth ))
                  
                    sell_balance = balance + total_sell_p
                    position = 0
                else:
                    price = df.loc[i,'Close'] 
                    stockprice = price * qty
                    MA_wealth = balance + stockprice
                    df.loc[i,'MA_wealth'] = MA_wealth
                    # print(MA_wealth)

            # long-term strategy           
        first_date = df['Date'].iloc[0]  
        initial_price = df['Close'].iloc[0]
        qty = LT_wealth/initial_price

        for i,row in df.iterrows():
            df.loc[i,'LT_wealth'] = df.loc[i,'Close'] * qty
                    
        last_date = df['Date'].iloc[-1]
        final_price = df['Close'].iloc[-1]
        
        LT_buy_p = initial_price * qty
        LT_sell_p = final_price * qty
        LT_profitloss = LT_sell_p - initial_wealth
        LT_wealth = initial_wealth + LT_profitloss
        MA_profitloss = MA_wealth - initial_wealth    
        MA_profitloss = round(MA_profitloss,2)
        LT_profitloss = round(LT_profitloss,2)


        print('-'*100)
        print('Short MA Profit/Loss: ${:,}, Long MA Profit/Loss: ${:,}'.format(MA_profitloss,LT_profitloss))
        print('')
        print('Short MA Final Wealth: ${:,.2f}, Long MA Final Wealth: ${:,.2f}'.format(MA_wealth,LT_wealth))
        print('-'*100)

        if MA_profitloss >= 500:
              passablestocks.append(stock)
              print('Passed')
        else:
            print('Not Passed')

        return df


for i in stocklist:
    ticker = stocklist[counter].upper()
    stock = ticker
    if int(len(yf.Ticker(ticker).history(period = 'max', interval = interval))) < datafilter_days:
          print('Not Enough Data')
    else:
          print('Sufficent Dataset')
    df = get_stock_data(stock,start_date,end_date,period,interval)
    df = ma_strategy(df,short_MA,long_MA)
    df = buy_sell_signals(df,stock,start_date,end_date)
    backtest(df,stock,start_date,end_date,initial_wealth)
    counter = counter + 1

print(passablestocks)