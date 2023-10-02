import math

import numpy as np
from matplotlib import pyplot as plt
import math

w = 1
b = 2
def f_wb(x, w, b):

    f_wb = w * np.sqrt(x) + b  # compute fx = w * sqrt(x) + b
    # f_wb = np.dot(w, x) + b  # perform dot product since w and x can be a vector
    return f_wb

x = np.linspace(-10, 10, 100)

plt.plot(x, f_wb(x, w, b), color='red')

plt.show()