import pandas as pd
import numpy as np

train = pd.read_csv("K://DS BA//BA//Project//Data (4)//Data//Loan Status Prediction//train.csv")

train = train.fillna(0)
train

x = train[['ApplicantIncome', 'CoapplicantIncome','LoanAmount', 'Credit_History']]
train['dep'] = np.where(train['Loan_Status'] == 'Y',1,0)
y = train['dep']
           

from sklearn.linear_model import LogisticRegression
l = LogisticRegression()
l.fit(x,y)

ypred = l.predict(x)
ypred

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y,ypred)
print (confusion_matrix)

test = pd.read_csv("K://DS BA//BA//Project//Data (4)//Data//Loan Status Prediction//test.csv")
test= test.fillna(0)
test

xtest = test[['ApplicantIncome', 'CoapplicantIncome','LoanAmount', 'Credit_History']]
xtest

ypred1 = l.predict(xtest)
ypred1
