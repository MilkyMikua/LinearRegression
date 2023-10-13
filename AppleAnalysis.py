# Build a linear regression model
# Each training example can contain multiple feature.
# Feature rescaling apply automatically with FeatureScaling's funtion.
# Created Sep 29 2023, Last modified: Sep 30 2023
# Yurui Huang

# By using a new dataSet, you may consider modify:
# 1. path, dataset. Locate the dateset and read it.
# 2. x_train, y_train. Change the training examples. You may consider rescale with different Rescaling method by change FeatureScaling.meanNormal(x_train).
# 3. w, b, a, epoch. The initial weight, bias, learning rate, amount of iteration. Since feature rescaled, may choose a slightly large learning rate


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import FeatureScaling

path = r'C:\Users\h8611\OneDrive\Desktop\MLS Slides\HousePrice\Stocks\AAPL5Year.csv'  # Location of the dataset and read it
dataset = pd.read_csv(path)

x_train = np.array(dataset.iloc[:, 0])   # consider the first 6 columns as features of a single training example
x_trainOriginal = x_train  # We need to feature scale x_train, keep this originalData for future use to plot
y_train = np.array(dataset.iloc[:, 5])  # The last column is the target
print(np.var(y_train), "std")

mean = np.mean(x_train, axis=0)
diff = np.max(x_train) - np.min(x_train)
print(mean, "mean")
print(diff, "np.max(x_train) - np.min(x_train)")
x_train = FeatureScaling.meanNormal(x_train)  # rescaling features. Rescaling data improves data processing. Notice: in this case, MSE does not converage without rescaling

m = x_train.shape[0]  # amount of training examples. row
#n = x_train.shape[1]  # number of features. Column
n=1
w = np.zeros(n)  # initial slopes are 0
# w = np.array([1, 1.2, 2, 1.6, 0.5])  # initial slope
b = 0  # initial bias
a = 0.05  # initial learning rate
epoch = 500 + int(m/100)  # Set expected iteration

def f_wb(x, w, b):
    '''
    :param w: coefficient
    :param x: training example
    :param b: constant
    :return: prediction based on given feature x
    '''

    #x = 132.822*x*x + 222.755*x +117.117 # from lagrange interpolation
    #x = x*x + x
    x = x * x *x + x * x + x
    f_wb = np.dot(w, x) + b  # perform dot product since w and x can be a vector
    # The optimal mudel should be w*132.822*x*x + w*222.755*x +117.117 + b
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
print(MSE_record[-1], "The final mean square error")
print(w, "Optimized weight w")
print(b, "Optimized Bias b")
plt.title("MSE")
plt.plot(MSE_record)
plt.show()


# [784.21393079] The final mean square error
# [38.15978989] Optimized weight w
# [103.26010511] Optimized Bias b

# Uncomment this block to visualize data points
# Plot the data points
plt.scatter(x_trainOriginal, y_train, marker='x', color='r')
# Set the title
plt.title("Apple Stock Prices")
# Set the y-axis label
plt.ylabel('Price in dollar')
# Set the x-axis label
plt.xlabel('at specific day. (day = (day - mean) / (max-min))')
#plt.show()



uppdomain = max(x_train)
botDomain = min(x_train)
x = np.linspace(botDomain, uppdomain+1, 10)

def rescale(x, w, b, mean, diff):
    x = (x-mean) / diff
    # x = 132.822*x*x + 222.755*x +117.117 # from lagrange interpolation
    x = x*x + x
    f_wb = np.dot(w, x) + b  # perform dot product since w and x can be a vector
    # The optimal mudel should be w*132.822*x*x + w*222.755*x +117.117 + b
    return f_wb



y_pred = []
for i in x_trainOriginal:
    y_hat = rescale(i, w, b, mean, diff)
    y_pred.append(y_hat)
plt.plot(y_pred)

plt.show()

# diff = y_pred - y_train
# print(diff)

