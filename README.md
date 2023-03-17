# MLTSK (Machine Learning Trader SciKitlearn)


## Purpose
MLTSK is designed to be a template that can be used to make stock predictions using machine learning through the library scikit-learn. We will collect and clean historical stock data using pandas and yFinance, after which we will apply several scikit-learn classifiers and regression models to discover relationships between stock fundamentals and the subsequent daily change of closing price. 


I do **NOT** recommend live trading based off of predictions from this exact code; however, you can make the changes needed to make this deployable. This is *purely* an educational project, and data may be deceptive -- trade at your **OWN** risk **ALWAYS**!
#### Overview 
The overall workflow to use machine learning to make stocks prediction is as follows:

1. Acquire historical fundamental data – these are the features or predictors
2. Acquire historical stock price data – this is will make up the dependent variable, or label (what we are trying to predict).
3. Preprocess data
4. Use a machine learning model to learn from the data
5. Acquire current fundamental data
6. Generate predictions from current fundamental data


This is a very generalised overview, but in principle this is all you need to build a fundamentals-based ML stock predictor.


## pip dependencies install: 
#### Navigate to file location and then run:
  * WINDOWS: py -m pip install -r requirements.txt 
  * LINUX: python -m pip install -r requirements.txt 

## General Notes
#### Files located in MASTER FILES are usable.
#### Anything with "test" is not stable.  
#### Currently the Extra Tree regression method is best.
  * Models will be saved to file location, or set your own filepath under either filepath_windows or filepath_linux
  * MASTER FILES contain final/test versions of more developed python scripts
  * Newests contains several test files that are related to the most recent crreations
  * Backtesting/ETFS_GradientBoostingRegressor/Random Forest Walk are backups of different types of python scripts



## ML_ClosingPrice_Predictors
#### Located under MASTER FILES  
  * Different forms of regression and ML programs to predict stock price  
  * Any file with _test.py is the unstable version 

## Optimization_Linux
 * Can only be used in a UNIX terminal due to library dependancies
 * Used to find the optimal ML model for data set 

