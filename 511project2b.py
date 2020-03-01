'''

Multi Layer Perceptron

Programmed partially or completely by Gautam Sharma and partial code taken from official sklearn documentation

We start by importing appropriate libraries

Wherever 1,2,3,4,5 is written it means it represents number of features
'''



import pandas as pd
import numpy as np
import sklearn
from sklearn.neural_network import MLPRegressor
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as pyplot


#Including happiness rank leads to high corelation
'''
Importing data from project2data.csv file provided in .zip file
----------------------------------------------------------------------
'''
data=pd.read_csv("project2data.csv",sep=",")
data1=data[["Economy","Happiness Score"]]
data2=data[["Economy","Family","Happiness Score"]]
data3=data[["Economy","Family","Health","Happiness Score"]]
data4=data[["Economy","Family","Health","Freedom","Happiness Score"]]
data5=data[["Economy","Family","Health","Freedom","Happiness Score","Generosity"]]
data6=data[["Economy","Family","Health","Freedom","Happiness Score","Generosity","Dystopia Residual"]]

print (data.head())  # prints first five elements

predict="Happiness Score"
year="Date"

'''
Different data initialized to compare its affect on number of features
----------------------------------------------------------------------
'''
X1= np.array(data1.drop([predict],1))
y1= np.array(data1[predict])

X2= np.array(data2.drop([predict],1))
y2= np.array(data2[predict])

X3= np.array(data3.drop([predict],1))
y3= np.array(data3[predict])

X4= np.array(data4.drop([predict],1))
y4= np.array(data4[predict])

X5= np.array(data5.drop([predict],1))
y5= np.array(data5[predict])

X6= np.array(data6.drop([predict],1))
y6= np.array(data6[predict])

x_train1, x_test1, y_train1, y_test1= sklearn.model_selection.train_test_split(X1,y1,test_size=0.1)
x_train2, x_test2, y_train2, y_test2= sklearn.model_selection.train_test_split(X2,y2,test_size=0.1)
x_train3, x_test3, y_train3, y_test3= sklearn.model_selection.train_test_split(X3,y3,test_size=0.1)
x_train4, x_test4, y_train4, y_test4= sklearn.model_selection.train_test_split(X4,y4,test_size=0.1)
x_train5, x_test5, y_train5, y_test5= sklearn.model_selection.train_test_split(X5,y5,test_size=0.1)
x_train6, x_test6, y_train6, y_test6= sklearn.model_selection.train_test_split(X6,y6,test_size=0.1)

"""
Starting MLP
----------------------------------------------------------------------
"""
linear1= linear_model.LinearRegression()
MLP1= MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=0.0001, batch_size='auto',
                  learning_rate='constant',
                  learning_rate_init=0.001, power_t=0.5, max_iter=2000, shuffle=True, random_state=None, tol=0.0001,
                  verbose=False,
                  warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1,
                  beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000)


MLP2= MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam',  alpha=0.0001, batch_size='auto',
                  learning_rate='constant',
                  learning_rate_init=0.001, power_t=0.5, max_iter=2000, shuffle=True, random_state=None, tol=0.0001,
                  verbose=False,
                  warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1,
                  beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000)
#
MLP3= MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam',  alpha=0.0001, batch_size='auto',
                  learning_rate='constant',
                  learning_rate_init=0.001, power_t=0.5, max_iter=2000, shuffle=True, random_state=None, tol=0.0001,
                  verbose=False,
                  warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1,
                  beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000)

MLP4= MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam',  alpha=0.0001, batch_size='auto',
                  learning_rate='constant',
                  learning_rate_init=0.001, power_t=0.5, max_iter=2000, shuffle=True, random_state=None, tol=0.0001,
                  verbose=False,
                  warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1,
                  beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000)

MLP5= MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam',  alpha=0.0001, batch_size='auto',
                  learning_rate='constant',
                  learning_rate_init=0.001, power_t=0.5, max_iter=2000, shuffle=True, random_state=None, tol=0.0001,
                  verbose=False,
                  warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1,
                  beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000)

MLP6= MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam',  alpha=0.0001, batch_size='auto',
                  learning_rate='constant',
                  learning_rate_init=0.001, power_t=0.5, max_iter=2000, shuffle=True, random_state=None, tol=0.0001,
                  verbose=False,
                  warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1,
                  beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000)

"""
Training and Testing Phase
----------------------------------------------------------------------
"""
MLP1.fit(x_train1,y_train1)
y_pred1= MLP1.predict(x_test1)
MLP1.fit(x_train1,y_train1)

MLP2.fit(x_train2,y_train2)
y_pred2= MLP2.predict(x_test2)
MLP2.fit(x_train2,y_train2)

MLP3.fit(x_train3,y_train3)
y_pred3= MLP3.predict(x_test3)
MLP3.fit(x_train3,y_train3)

MLP4.fit(x_train4,y_train4)
y_pred4= MLP4.predict(x_test4)
MLP4.fit(x_train4,y_train4)

MLP5.fit(x_train5,y_train5)
y_pred5= MLP5.predict(x_test5)
MLP5.fit(x_train5,y_train5)

MLP6.fit(x_train6,y_train6)
y_pred6= MLP6.predict(x_test6)
MLP6.fit(x_train6,y_train6)
"""
Calculating RMSE for all 6 cases
----------------------------------------------------------------------
"""
RMSE1=mean_squared_error(y_test1, y_pred1, squared=False)
print("RMSE: ", RMSE1)
RMSE2=mean_squared_error(y_test2, y_pred2, squared=False)
print("RMSE: ", RMSE2)
RMSE3=mean_squared_error(y_test3, y_pred3, squared=False)
print("RMSE: ", RMSE3)
RMSE4=mean_squared_error(y_test4, y_pred4, squared=False)
print("RMSE: ", RMSE4)
RMSE5=mean_squared_error(y_test5, y_pred5, squared=False)
print("RMSE: ", RMSE5)
RMSE6=mean_squared_error(y_test6, y_pred6, squared=False)
print("RMSE: ", RMSE6)
score= MLP1.score( x_test1, y_test1, sample_weight=None)
print(score)
score= MLP2.score( x_test2, y_test2, sample_weight=None)
print(score)
score= MLP3.score( x_test3, y_test3, sample_weight=None)
print(score)
score= MLP4.score( x_test4, y_test4, sample_weight=None)
print(score)
score= MLP5.score( x_test5, y_test5, sample_weight=None)
print(score)
score= MLP6.score( x_test6, y_test6, sample_weight=None)
print(score)

"""
Plotting loss functions as a function of iterations
----------------------------------------------------------------------
"""
fig, ax = pyplot.subplots()

ax.set(xlabel='Iterations', ylabel='Loss',
       title='Loss function convergence with increasing features')
ax.grid()
pyplot.plot(MLP1.loss_curve_)
pyplot.plot(MLP2.loss_curve_)
pyplot.plot(MLP3.loss_curve_)
pyplot.plot(MLP4.loss_curve_)
pyplot.plot(MLP5.loss_curve_)
pyplot.plot(MLP6.loss_curve_)

ax.legend([RMSE1, RMSE2, RMSE3,RMSE4,RMSE5,RMSE6])

ax.legend(['1 feature','2 feature', '3 feature', '4 feature','5 feature','6 feature'])
fig.savefig("Loss function convergence with increasing features.png")


# p="Happiness Score"
# style.use("ggplot")
# pyplot.scatter(data[p],data["Health"])
# pyplot.xlabel(p)
# pyplot.ylabel("Health")

pyplot.show()
