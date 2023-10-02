# Build a simple linear regression model
# Each training exmaple contain ONLY 1 FEATURE.
# Feature rescaling applied.
# Since this model only contain 1 feature, we can visualize the plot of Price(Target) over Area(feature), and the apporiate predictor curve
# Created Sep 29 2023, Last modified: Sep 30 2023
# Yurui Huang

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import FeatureScaling

# Parameters and functions:
# x_train = training examples
# y_train = training targets
# m = Number of training example
# a = learning rate
# b = intercept, bias, b
# w = slope, weight, w
# epoch = number of iterations
# f_wb = Model, a function process a given w, b and features of x. Predict a reasonable price
# cosfFun = Mean Squared Error (MSE) or Loss Function
# grad_descent = the gradient descent function, ∇L


path = r'C:\Users\h8611\OneDrive\Desktop\MLS Slides\HousePrice\house.csv'
dataset = pd.read_csv(path)  # read the dataset

x_train = np.array(dataset.iloc[:, 1])  # read the first column's values as feature
y_train = np.array(dataset.iloc[:, -1])  # read the last column's value as target

# x_train = x_train / 1000  # rescale data. Without this rescaling, the result is off very much since feature and target dont share the same scale.
x_train = FeatureScaling.meanNormal(x_train)
# x_train = np.array([1.0, 2.0, 5, 7, 8, 10, 11])  # custom test sample
# y_train = np.array([3.0, 5.0, 10, 15, 19, 18, 23]) # custom test target
m = x_train.shape[0]  # amount of training examples
w = 1  # initial slope
b = 0  # initial bias
a = 0.1  # initial learning rate
epoch = 500  # 100 iteration


def f_wb(x, w, b):
    '''
    :param w: coefficient
    :param x: training example
    :param b: constant
    :return: prediction based on given feature x
    '''
    # f_wb = np.sum(w * x * x) + b  # compute fx = w * x^2 + b
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
        diff = ( f_wb(x[i], w, b) - y[i] ) **2
        sum = sum + diff

    MSE = sum / (2*m)
    return MSE

def grad_descent(w, b, a, x, y, m):
    dj_dw = 0
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


UsedPars = []  # collect parameters of w and b after each iteration. We may plot it to visualize change of each iteration.
for i in range(epoch):
    grad_Desc_i = grad_descent(w, b, a, x_train, y_train, m)  # The ith updated new parameters of w and b
    w = grad_Desc_i[0]
    b = grad_Desc_i[1]

    # collect parameters of w and b after each iteration
    w_b = []
    w_b.append(w)
    w_b.append(b)
    UsedPars.append(w_b)

print(w, "Optimized weight w")
print(b, "Optimized Bias b")


# Uncomment this block to visualize data points
# Plot the data points
plt.scatter(x_train, y_train, marker='x', color='r')
# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
#plt.show()


uppdomain = max(x_train)
x = np.linspace(0, uppdomain+1, 10)

for i in range(1, 501, 100):
    w = UsedPars[i][0]
    b = UsedPars[i][1]
    plt.plot(x, f_wb(x, w, b), label=str(i) + "th iter")  # Corrected label assignment
    MSE = costFun(w, b, x_train, y_train, m)  # to verify if the error do converage
    print(MSE, "MSE at "+ str(i) + "th iteration")
    plt.legend()

plt.show()


