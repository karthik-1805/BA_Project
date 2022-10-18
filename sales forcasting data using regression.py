import numpy as np
import pandas as pd 
import seaborn as sns
import os
import matplotlib.pyplot as plt 

train = pd.read_csv("K://DS BA//BA//Project//Data (4)//Data//Sales Forecasting//train.csv")
train =  train.fillna(0)
features = pd.read_csv("K://DS BA/BA//Project//Data (4)//Data//Sales Forecasting//features.csv")
features = features.fillna(0)
train['Date'] = pd.to_datetime(train['Date'])
features['Date'] = pd.to_datetime(features['Date'])
adata = pd.merge(features, train, on=["Store", "Date"],how='inner')
adata = adata.fillna(0)
adata

adata.head()

print(adata.head())
adata.info()

adata.describe()

adata.columns

adata.corr()

x = adata[['MarkDown1', 'MarkDown2', 'MarkDown3','MarkDown4', 'MarkDown5', 'CPI']]
y = adata.Weekly_Sales

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(x)

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import linear_model

regr = linear_model.LinearRegression()
regr.fit(X,y)
y_pred = regr.predict(X)
print(X[1:10])

regr.intercept_

regr.coef_

Sales = 15981+732*(adata["MarkDown1"])+241*(adata["MarkDown2"])+849*(adata["MarkDown3"])-30*(adata["MarkDown4"])+819*(adata["MarkDown5"])-533*(adata["CPI"])
Sales

print("R squared: {}".format(r2_score(y_true=y,y_pred=y_pred)))