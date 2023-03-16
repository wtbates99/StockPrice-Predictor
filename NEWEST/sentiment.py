# Import libraries
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import os
import pandas as pd
import matplotlib.pyplot as plt
# NLTK VADER for sentiment analysis
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


listt = ['tsla']


def testing(ticker):
    web_url = 'https://finviz.com/quote.ashx?t='

    news_tables = {}

    tickers = ticker

    for tick in tickers:
        url = web_url + tick
        req = Request(url=url,headers={"User-Agent": "Chrome"}) 
        response = urlopen(req)    
        html = BeautifulSoup(response,"html.parser")
        news_table = html.find(id='news-table')
        news_tables[tick] = news_table
        
    count = 0
    for tick in tickers:
        ticker = tickers[count].upper()
        snews = news_tables[ticker.lower()]
        snews_tr = snews.findAll('tr')
        for x, table_row in enumerate(snews_tr):
            a_text = table_row.a.text
            td_text = table_row.td.text
            if x == 3:
                break
        count = count + 1
    news_list = []



    for file_name, news_table in news_tables.items():
        for i in news_table.findAll('tr'):
            
            try:
                text = i.a.get_text() 
            except AttributeError:
                print('')

            
            date_scrape = i.td.text.split()

            if len(date_scrape) == 1:
                time = date_scrape[0]
                
            else:
                date = date_scrape[0]
                time = date_scrape[1]

            tick = file_name.split('_')[0]
            
            news_list.append([tick, date, time, text])
            
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

    mean_scores = mean_scores.tail(7)
    mean_scores = mean_scores.mean()
    averagescore = float(mean_scores)
    print(averagescore)

testing(listt)