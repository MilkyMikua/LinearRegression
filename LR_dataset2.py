# Use the linear regression mod with a different dataset
# This is the demonstration of apply LR with another dataset
# Created Sep 29 2023, Last modified: Sep 30 2023


# By using a new dataSet, you may consider modify:
# 1. path, dataset. Locate the dateset and read it.
# 2. x_train, y_train. Change the training examples. You may consider rescale with different Rescaling method by change FeatureScaling.meanNormal(x_train).
# 3. w, b, a, epoch. The initial weight(same dimension as x—_train), bias(a scalar), learning rate(0<a<1), amount of iteration. Since feature rescaled, may choose a slightly large learning rate


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import FeatureScaling

#path = r'C:\Users\h8611\OneDrive\Desktop\MLS Slides\HousePrice\house.csv'  # Location of the dataset and read it
path = r'C:\Users\h8611\OneDrive\Desktop\MLS Slides\HousePrice\stock_data.csv'
dataset = pd.read_csv(path)

x_train = np.array(dataset.iloc[0:1000, 3:6])   # consider the 4th, 5th, 6th columns as features of a single training example. And only first 1000 examples
y_train = np.array(dataset.iloc[0:1000, -1])  # The last column is the target

x_train = FeatureScaling.meanNormal(x_train)  # rescaling features. Rescaling data improves data processing. Notice: in this case, MSE does not converage without rescaling

m = x_train.shape[0]  # amount of training examples. row
n = x_train.shape[1]  # number of features. Column
w = np.zeros(n)  # initial slopes are 0
# w = np.array([1, 1.2, 2, 1.6, 0.5])  # initial slope
b = 0  # initial bias
a = 0.05  # initial learning rate
epoch = 50 + int(m/100)  # Set expected iteration

def f_wb(x, w, b):
    '''
    :param w: coefficient
    :param x: training example
    :param b: constant
    :return: prediction based on given feature x
    '''

    f_wb = np.dot(w, x) + b  # perform dot product since w and x can be a vector
    return f_wb

def costFun(w, b, x, y, m):
    '''
    :param w: slope
    :param b: bias
    :param x: training example feature
    :param y: actual output
    :param m: number of training examples
    :return: mean square error
    '''
    sum = 0
    for i in range(m):
        diff = (f_wb(x[i], w, b) - y[i]) ** 2
        sum = sum + diff

    MSE = sum / (2*m)
    return MSE

def grad_descent(w, b, a, x, y, m):
    dj_dw = np.zeros(n)
    dj_db = 0
    for i in range(m):
        dj_dw_i = (f_wb(x[i], w, b) - y[i]) * x[i]
        dj_db_i = (f_wb(x[i], w, b) - y[i])
        dj_dw = dj_dw_i + dj_dw
        dj_db = dj_db_i + dj_db
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    w = w - a * dj_dw
    b = b - a * dj_db
    return w, b


MSE_record = []
for i in range(epoch):
    w, b = grad_descent(w, b, a, x_train, y_train, m)
    MSE_record.append(costFun(w, b, x_train, y_train, m))

print(MSE_record)
print(" ")
print(w, "Optimized weight w \n")
print(b, "Optimized Bias b")
plt.plot(MSE_record)
plt.show()
