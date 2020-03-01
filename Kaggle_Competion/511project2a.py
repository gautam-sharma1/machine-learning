'''

Linear Regression

Programmed partially or completely by Gautam Sharma and partial code taken from official sklearn documentation

We start by importing appropriate libraries

Wherever 1,2,3,4,5 is written it means it represents number of features
'''
import pandas as pd
import numpy as np
import sklearn
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as pyplot
import math

'''
Importing data from final.csv file provided in .zip file
----------------------------------------------------------------------
'''
data=pd.read_csv("final.csv",sep=",")
data=data[["Temperature AVG","Relative Humidity AVG","Wind Speed Daily AVG","Success Percentage"]]

print (data.head())  # prints first five elements



predict="Success Percentage"
year="Date"

"""
Initializing different size training and testing sets
----------------------------------------------------------------------
"""


X= np.array(data.drop([predict],1))#Only 6 feature

y= np.array(data[predict])

x_train1, x_test1, y_train1, y_test1= sklearn.model_selection.train_test_split(X,y,test_size=0.2,random_state=0)




"""
Starting Linear Regression
----------------------------------------------------------------------
"""

linear1= linear_model.LinearRegression()

#
linear1.fit(x_train1,y_train1)
acc1=linear1.score(x_test1,y_test1)
y_pred1=linear1.predict(x_test1)
y_pred2=linear1.predict(x_train1)

RMSE1=mean_squared_error(y_test1, y_pred1, squared=False)
RMSE2=mean_squared_error(y_train1, y_pred2, squared=False)


#
print("RMSE test: ", RMSE1)
print("RMSE train: ", RMSE2)
print("Accuracy", acc1)
print("Linear coefficients: ",linear1.coef_)


E=[]
#
"""
Calculating Error with number of iterations as sklearn does not allows to calculate error with iteration
----------------------------------------------------------------------
"""
for i  in range(1,np.size(y_pred1)):

   e = (abs(y_test1[i] ** 2 - y_pred1[i] ** 2))
   E.append(math.sqrt(e / i))
fig, ax = pyplot.subplots()
ax.set(xlabel='Iterations', ylabel='Linear Regression Error',
       title='Linear Regression Error convergence')
#

E=np.array(E)
pyplot.plot(E)


pyplot.show()

