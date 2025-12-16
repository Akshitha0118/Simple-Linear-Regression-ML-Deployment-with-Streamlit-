# import the libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pickle
from scipy.stats import variation
import scipy.stats as stats

# import the dataset
dataset=pd.read_csv(r'C:\Users\ADMIN\Downloads\Salary_Data.csv')


# divide the dataset to x and y [independent and dependent variable] 
x = dataset.iloc[: , :-1].values
y = dataset.iloc[:,-1].values


# split the dataset to train set and test set
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8)


# build the model using LinearRegression algorithm
regressor = LinearRegression()
regressor.fit(x_train,y_train)

# predict the model 
y_pred=regressor.predict(x_test)

# visualize the model
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('salary vs experience')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()


# slope of the model 
m_slope=regressor.coef_
print(m_slope)


# constant or intercept of the model 
c_intercept=regressor.intercept_
print(c_intercept)

## future predictions of 12 & 20 years of experience 
y_12 = m_slope * 12 + c_intercept
print(y_12)

y_20 = m_slope * 20 + c_intercept
print(y_20)


# stats integration to machine learning
# dataset mean value
dataset.mean()
dataset['Salary'].mean()

# dataset median value
dataset.median()
dataset['Salary'].median()

# dataset mode value 
dataset.mode()
dataset['Salary'].mode()

# variance of the dataset
dataset.var()
dataset['YearsExperience'].var()

# standard deviation of salary & yearof experiance 
dataset.std()
dataset['Salary'].std()
dataset['YearsExperience'].std()

# coeefficient of variance 
variation(dataset.values)
variation(dataset['Salary'])

# correlation of dataset
dataset.corr()
dataset['Salary'].corr(dataset['YearsExperience'])

# skewness of the dataset
dataset.skew()
dataset['Salary'].skew()

# standard error
dataset.sem()

# standardization technique z - score
dataset.apply(stats.zscore)
stats.zscore(dataset['Salary'])

## SSR
y_mean = np.mean(y)
SSR =np.sum((y_pred - y_mean)**2)
print(SSR)

## SSE
y =y[0:6]
SSE =np.sum((y-y_pred)**2)
print(SSE)

#SST
mean_total = np.mean(dataset.values)
SST = np.sum((dataset.values-mean_total)**2)
print(SST)

#R2
r_square = 1 -SSR/SST
print(r_square)

# bias score with (x_train,y_train)
bias_score = regressor.score(x_train,y_train)


# variance score with (x_test,y_test)
variance_score =regressor.score(x_test,y_test)


# pickling the file to binary 
filename='Linear_regression_model.pkl'
with open(filename,'wb') as file:
    pickle.dump(regressor,file)
print('model has been as pickled')

# finding the file location 
import os 
os.getcwd()