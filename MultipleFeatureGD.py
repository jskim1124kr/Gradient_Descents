import numpy as np
from numpy.linalg import inv
import pandas as pd
import matplotlib.pyplot as plt


steps = 1000
learning_late = 0.01

def mean_normalization(X):
    mean = np.mean(X, axis=0)
    stdev = np.std(X, axis=0)
    X_normalization = (X - mean) / stdev

    return X_normalization

data = pd.read_csv('data.csv')

X = data.as_matrix(columns=['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10'])

X = mean_normalization(X)
Y = data['y'].values /100000

bias_feature = np.zeros(len(X)).reshape(len(X), 1)
bias_feature[:] = 1
X = np.hstack((bias_feature,X))

N,D = X.shape
print(X.shape)
print(Y.shape)




def cost_function(X,theta,Y):
    return np.sum(np.square(np.matmul(X, theta) - Y)) / (2 * len(Y))


costs = []
def multi_Gradient_Descent(X,Y,lr,steps):
    theta = np.zeros(D)
    for i in range(steps):
        gradient = (np.matmul(X.T, np.matmul(X, theta) - Y))/N
        theta = theta - lr * gradient
        cost = cost_function(X, theta, Y)
        if i % 20 == 0:
            print("steps [ " + str(i) +" ]  /  cost [ " + str(cost) + " ]")
            costs.append(cost)

    return [theta,cost,costs]


def multi_Gradient_Descent_reg(X,Y,lr,steps,lambda_):
    theta = np.zeros(D)
    m = len(X)
    reg_param = (1 - lr * (lambda_ / m))
    for i in range(steps):
        gradient = (np.matmul(X.T, np.matmul(X, theta) - Y))/N
        theta = np.multiply(theta,reg_param) - lr * gradient
        cost = cost_function(X, theta, Y)
        if i % 20 == 0:
            print("steps [ " + str(i) +" ]  /  cost [ " + str(cost) + " ]")
            costs.append(cost)

    return [theta,cost,costs]

# theta,cost,c_list = multi_Gradient_Descent(X,Y,learning_late,steps)
theta,cost,c_list = multi_Gradient_Descent_reg(X,Y,learning_late,steps,10.0)
print()
print("Gradient_Descent-reg - theta : ", theta)
print("Gradient_Descent-reg - cost : ", cost)
print()


x_axis = []
for i in range(0,1000,20):
    x_axis.append(i)

plt.plot(x_axis, c_list, 'r')
plt.xlabel('Number of Epochs')
plt.ylabel('cost')
plt.show()

def normal_equation(X,Y):
    return np.matmul(np.matmul(inv(np.matmul(X.T,X)),X.T),Y)


def normal_equation_reg(X,Y,lambda_):
    L = np.eye(D)
    L[0,0] = 0
    return np.matmul(np.matmul(inv(np.matmul(X.T,X)+(lambda_* L)),X.T),Y)

# theta2 = normal_equation(X,Y)
theta2 = normal_equation_reg(X,Y,10.0)
cost = np.sum(np.square(np.matmul(X, theta2) - Y)) / (2 * len(Y))

print("Normal_equation - theta : ", theta2)
print("Normal_equation - cost : ", cost)




