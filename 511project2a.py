'''

Linear Regression

Programmed partially or completely by Gautam Sharma and partial code taken from official sklearn documentation

We start by importing appropriate libraries

Wherever 1,2,3,4,5 is written it means it represents number of features
'''
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as pyplot
import math

'''
Importing data from project2data.csv file provided in .zip file
----------------------------------------------------------------------
'''
data=pd.read_csv("project2data.csv",sep=",")
data=data[["Economy","Family","Health","Freedom","Happiness Score","Generosity","Dystopia Residual"]]

print (data.head())  # prints first five elements


'''
Different data initialized to compare its affect on number of features
----------------------------------------------------------------------
'''
data1=data[["Economy","Happiness Score"]]
data2=data[["Economy","Family","Happiness Score"]]
data3=data[["Economy","Family","Health","Happiness Score"]]
data4=data[["Economy","Family","Health","Freedom","Happiness Score"]]
data5=data[["Economy","Family","Health","Freedom","Happiness Score","Generosity"]]
data6=data[["Economy","Family","Health","Freedom","Happiness Score","Generosity","Dystopia Residual"]]

print (data.head())  # prints first five elements
"""Assigning Happiness Score to be the output variable"""
predict="Happiness Score"
year="Date"

"""
Initializing different size training and testing sets
----------------------------------------------------------------------
"""

#print(X)
X1= np.array(data1.drop([predict],1)) #Only 1 feature
y1= np.array(data1[predict])

X2= np.array(data2.drop([predict],1))  #Only 2 feature
y2= np.array(data2[predict])

X3= np.array(data3.drop([predict],1))   #Only 3 feature
y3= np.array(data3[predict])

X4= np.array(data4.drop([predict],1)) #Only 4 feature
y4= np.array(data4[predict])

X5= np.array(data5.drop([predict],1))  #Only 5 feature
y5= np.array(data5[predict])

X6= np.array(data6.drop([predict],1))#Only 6 feature
y6= np.array(data6[predict])

x_train1, x_test1, y_train1, y_test1= sklearn.model_selection.train_test_split(X1,y1,test_size=0.1)
x_train2, x_test2, y_train2, y_test2= sklearn.model_selection.train_test_split(X2,y2,test_size=0.1)
x_train3, x_test3, y_train3, y_test3= sklearn.model_selection.train_test_split(X3,y3,test_size=0.1)
x_train4, x_test4, y_train4, y_test4= sklearn.model_selection.train_test_split(X4,y4,test_size=0.1)
x_train5, x_test5, y_train5, y_test5= sklearn.model_selection.train_test_split(X5,y5,test_size=0.1)
x_train6, x_test6, y_train6, y_test6= sklearn.model_selection.train_test_split(X6,y6,test_size=0.1)

predict="Happiness Score"
year="Date"
X= np.array(data.drop([predict],1))
print(X)
y= np.array(data[predict])


"""
Starting Linear Regression
----------------------------------------------------------------------
"""

linear1= linear_model.LinearRegression()
linear2= linear_model.LinearRegression()
linear3= linear_model.LinearRegression()
linear4= linear_model.LinearRegression()
linear5= linear_model.LinearRegression()
linear6= linear_model.LinearRegression()

linear1.fit(x_train1,y_train1)
acc1=linear1.score(x_test1,y_test1)
y_pred1=linear1.predict(x_test1)

linear2.fit(x_train2,y_train2)
acc2=linear2.score(x_test2,y_test2)
y_pred2=linear2.predict(x_test2)

linear3.fit(x_train3,y_train3)
acc3=linear3.score(x_test3,y_test3)
y_pred3=linear3.predict(x_test3)


linear4.fit(x_train4,y_train4)
acc4=linear4.score(x_test4,y_test4)
y_pred4=linear4.predict(x_test4)


linear5.fit(x_train5,y_train5)
acc5=linear5.score(x_test5,y_test5)
y_pred5=linear5.predict(x_test5)


linear6.fit(x_train6,y_train6)
acc6=linear6.score(x_test6,y_test6)
y_pred6=linear6.predict(x_test6)


"""
Calculating Root Mean Square Error
----------------------------------------------------------------------
"""
# print("accuracy: ", acc)
RMSE1=mean_squared_error(y_test1, y_pred1, squared=False)
RMSE2=mean_squared_error(y_test2, y_pred2, squared=False)
RMSE3=mean_squared_error(y_test3, y_pred3, squared=False)
RMSE4=mean_squared_error(y_test4, y_pred4, squared=False)
RMSE5=mean_squared_error(y_test5, y_pred5, squared=False)
RMSE6=mean_squared_error(y_test6, y_pred6, squared=False)

print("RMSE: ", RMSE6)
#E1 is error for 1 features and so on..
E1=[]
E2=[]
E3=[]
E4=[]
E5=[]
E6=[]

"""
Calculating Error with number of iterations as sklearn does not allows to calculate error with iteration
----------------------------------------------------------------------
"""
for i  in range(1,np.size(y_pred1)):
   e1=(abs(y_test1[i]**2-y_pred1[i]**2))
   E1.append(math.sqrt(e1/i))

   e2 = (abs(y_test2[i] ** 2 - y_pred2[i] ** 2))
   E2.append(math.sqrt(e2 / i))

   e3 = (abs(y_test3[i] ** 2 - y_pred3[i] ** 2))
   E3.append(math.sqrt(e3 / i))

   e4 = (abs(y_test4[i] ** 2 - y_pred4[i] ** 2))
   E4.append(math.sqrt(e4 / i))

   e5 = (abs(y_test5[i] ** 2 - y_pred5[i] ** 2))
   E5.append(math.sqrt(e5 / i))

   e6 = (abs(y_test6[i] ** 2 - y_pred6[i] ** 2))
   E6.append(math.sqrt(e6 / i))
fig, ax = pyplot.subplots()
ax.set(xlabel='Iterations', ylabel='Error',
       title='Variation in error with number of features')

E1=np.array(E1)
E2=np.array(E2)
E3=np.array(E3)
E4=np.array(E4)
E5=np.array(E5)
E6=np.array(E6)


pyplot.grid()
pyplot.bar(1,RMSE1)
pyplot.bar(2,RMSE2)
pyplot.bar(3,RMSE3)
pyplot.bar(4,RMSE4)
pyplot.bar(5,RMSE5)
pyplot.bar(6,RMSE6)

pyplot.legend(['1 feature','2 feature','3 feature','4 feature','5 feature','6 feature'])

fig.savefig("Los function.png")

pyplot.show()

"""
Feel free to uncomment below code to plot correlation graphs
----------------------------------------------------------------------
"""

# p="Happiness Score"
# style.use("ggplot")
# pyplot.scatter(data[p],data["Family"])
# pyplot.xlabel(p)
# pyplot.ylabel("Family")
# fig.savefig("Family.png")
# pyplot.show()
#
# style.use("ggplot")
# pyplot.scatter(data[p],data["Freedom"])
# pyplot.xlabel(p)
# pyplot.ylabel("Freedom")
# fig.savefig("Freedom.png")
# pyplot.show()
#
# style.use("ggplot")
# pyplot.scatter(data[p],data["Generosity"])
# pyplot.xlabel(p)
# pyplot.ylabel("Generosity")
# fig.savefig("Generosity.png")
# pyplot.show()
#
#
# pyplot.scatter(data[p],data["Economy"])
# pyplot.xlabel(p)
# pyplot.ylabel("Economy")
# fig.savefig("Economy.png")
# pyplot.show()

