import sklearn
import math
from sklearn.preprocessing import PolynomialFeatures
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Importing the libraries
import numpy as np
import matplotlib.pyplot as pyplot
import pandas as pd

# Importing the dataset
data=pd.read_csv("final.csv",sep=",")
data=data[["Temperature AVG","Relative Humidity AVG","Wind Speed Daily AVG","Success Percentage"]]
predict="Success Percentage"
year="Date"

X= np.array(data.drop([predict],1))#Only 6 feature
X= np.array(data.drop([predict],1))#Only 6 feature
'''
Preprocessing the data
'''
X=preprocessing.scale(X)
y= np.array(data[predict])
poly = PolynomialFeatures(degree=2,interaction_only=True)

X = poly.fit_transform(X)


x_train1, x_test1, y_train1, y_test1= sklearn.model_selection.train_test_split(X,y,test_size=0.2,random_state=0)


poly.fit(x_train1, y_train1)
lin2 = LinearRegression()
lin2.fit(x_train1, y_train1)
y_pred1=lin2.predict(x_test1)
y_pred2=lin2.predict(x_train1)
RMSE1=mean_squared_error(y_test1, y_pred1, squared=False)
print("RMSE: ", RMSE1)
RMSE2=mean_squared_error(y_train1, y_pred2, squared=False)
print("RMSE: ", RMSE2)

print(lin2.coef_)
E=[]
#
"""
Calculating Error with number of iterations as sklearn does not allows to calculate error with iteration
----------------------------------------------------------------------
"""
for i in range(1,np.size(y_pred1)):

   e = (abs(y_test1[i] ** 2 - y_pred1[i] ** 2))
   E.append(math.sqrt(e / i))
fig, ax = pyplot.subplots()
ax.set(xlabel='Iterations', ylabel='Polynomial Regression Error',
       title='Error convergence')

E=np.array(E)
pyplot.plot(E)
pyplot.show()
