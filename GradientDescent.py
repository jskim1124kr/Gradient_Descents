from numpy import *
from matplotlib import pyplot as plt


data = genfromtxt('data.csv',delimiter=',')


m = len(data)


def load_data(data):
    x_data = []
    y_data = []
    for i in range(m):
        x = data[i,0]
        y = data[i,1]
        x_data.append(x)
        y_data.append(y)
    return x_data, y_data

def plot_line(y, data_points):
    x_values = [i for i in range(int(min(data_points))-1, int(max(data_points))+2)]
    y_values = [y(x) for x in x_values]
    plt.plot(x_values, y_values, 'r')


x,y = load_data(data)

learning_late = 0.01
steps = 1000
W = 0
b = 0
H = lambda x : (W*x) + b



def cost_function(x,y):
    for i in range(m):
        theta0 = 0
        theta1 = 0
        theta0 += H(x[i]) - y[i]
        theta1 += (H(x[i]) - y[i]) * x[i]
    return theta0/m , theta1/m


for i in range(steps):
    c1,c2 = cost_function(x,y)
    W = W - (learning_late * c1)
    b = b - (learning_late * c2)

print('W : {} / b : {}'.format(W, b))
plot_line(H, x)
plt.plot(x, y, 'bo')
plt.show()