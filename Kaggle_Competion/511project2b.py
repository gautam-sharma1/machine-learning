'''

Multi Layer Perceptron

Programmed partially or completely by Gautam Sharma and partial code taken from official sklearn documentation

We start by importing appropriate libraries


'''



import pandas as pd
import numpy as np
import sklearn
from sklearn.neural_network import MLPRegressor
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as pyplot

'''
Importing data from project2data.csv file provided in .zip file
----------------------------------------------------------------------
'''
data=pd.read_csv("final.csv",sep=",")
data=data[["Temperature AVG","Relative Humidity AVG","Wind Speed Daily AVG","Success Percentage"]]

print (data.head())  # prints first five elements

predict="Success Percentage"
year="Date"


X= np.array(data.drop([predict],1))
y= np.array(data[predict])


x_train, x_test, y_train, y_test= sklearn.model_selection.train_test_split(X,y,test_size=0.2)

"""
Starting MLP
----------------------------------------------------------------------
"""



MLP= MLPRegressor(hidden_layer_sizes=(8,), activation='relu', solver='adam',  alpha=0.05, batch_size='auto',
                  learning_rate='constant',
                  learning_rate_init=0.01, power_t=0.5, max_iter=2000, shuffle=True, random_state=None, tol=0.0001,
                  verbose=False,
                  warm_start=False, momentum=0.5, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1,
                  beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000)

"""
Training and Testing Phase
----------------------------------------------------------------------
"""


MLP.fit(x_train,y_train)
y_pred= MLP.predict(x_test)
y_pred1= MLP.predict(x_train)
MLP.fit(x_train,y_train)
"""
Calculating RMSE for all 6 cases
----------------------------------------------------------------------
"""

RMSE=mean_squared_error(y_test, y_pred, squared=False)
print("RMSE: ", RMSE)
RMSE1=mean_squared_error(y_train, y_pred1 ,squared=False)
print("RMSE: ", RMSE1)


score= MLP.score( x_test, y_test, sample_weight=None)
print(score)

"""
Plotting loss functions as a function of iterations
----------------------------------------------------------------------
"""
fig, ax = pyplot.subplots()

ax.set(xlabel='Iterations', ylabel='Loss',
       title='Loss function convergence with increasing features')
ax.grid()

pyplot.plot(MLP.loss_curve_)

ax.legend([RMSE])

pyplot.show()
