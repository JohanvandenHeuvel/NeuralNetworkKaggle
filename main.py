# import lib
import numpy as np

# sigmod function
def nonlinear(x, deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

# input dataset
X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

# output dataset
Y = np.array([[0, 1, 1, 0]]).T

# seed random numbers 0 or 1
np.random.seed(1)

# init random weights
syn0 = 2*np.random.random((3, 1)) - 1

for iter in range(10000):
    l0 = X  # input layer
    l1 = nonlinear(np.dot(l0, syn0))  # hidden layer = dot product of l0 and syn
    l1_error = Y - l1  # calculate amount of error
    l1_delta = l1_error * nonlinear(l1, True)  # ?
    syn0 += np.dot(l0.T, l1_delta)  # update weights

print("trained network:")
print(l1)