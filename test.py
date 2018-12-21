import numpy as np
import matplotlib as plt
def step_function(x):
    y = x > 0
    return y.astype(np.int)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    return np.maximum(0,x)

def identity_function(x):
    return x

def init_network():
    network = {}
    network['w1'] = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
    network['b1'] = np.array([0.1,0.2,0.3])
    network['w2']= np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
    network['b2']= np.array([0.1,0.2])
    network['w3'] = np.array([[0.1,0.3],[0.2,0.4]])
    network['b3'] = np.array([0.1,0.2])
    return network

def forward(network,x):
    w1,w2,w3 = network['w1'],network['w2'],network['w3']
    b1,b2,b3 = network['b1'],network['b2'],network['b3']
    A1=np.dot(x,w1)+b1
    Z1 = sigmoid(A1)
    A2=np.dot(Z1,w2)+b2
    Z2 = sigmoid(A2)
    A3= np.dot(Z2,w3)+b3
    y= identity_function(A3)

    return y

network = init_network()
X = np.array([1.0,0.5])
y= forward(network,X)
print(y)
