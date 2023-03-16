import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score
from sklearn.ensemble import GradientBoostingRegressor
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
from itertools import combinations


sample_list =['Actual_Close', 'Target', 'Close', 'Volume', 'Open', 'High', 'Low',
       'weekly_mean', 'quarterly_mean', 'annual_mean', 'annual_weekly_mean',
       'annual_quarterly_mean', 'weekly_trend', 'open_close_ratio',
       'high_close_ratio', 'low_close_ratio', 'SP CLOSE', 'sp_weekly_mean',
       'slope', 'stderr']
for i in range(len(sample_list) + 1):
    for subset in combinations(sample_list, i):
        print (subset)
