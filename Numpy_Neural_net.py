import numpy as np
def init_net():
    return {
        'W1': np.random.randn(9,64) * 0.1,
        'b1': np.zeros(64),
        'W2': np.random.randn(64,64) * 0.1,
        'b2': np.zeros(64),
        'W3': np.random.randn(64,9) * 0.1,
        'b3': np.zeros(9),
    }

def softmax(x):
    ex = np.exp(x - np.max(x, axis=1, keepdims=True))
    return ex / ex.sum(axis=1, keepdims=True)

def forward(net, x):
    z1 = x.dot(net['W1']) + net['b1']
    a1 = np.maximum(z1, 0)
    z2 = a1.dot(net['W2']) + net['b2']
    a2 = np.maximum(z2, 0)
    z3 = a2.dot(net['W3']) + net['b3']
    out = softmax(z3)
    return a1, a2, out

def backward(net, x, y, a1, a2, out, lr=0.001):
    m = y.shape[0]
    dz3 = (out - y) / m
    dW3 = a2.T.dot(dz3)
    db3 = dz3.sum(axis=0)
    da2 = dz3.dot(net['W3'].T)
    dz2 = da2 * (a2 > 0)
    dW2 = a1.T.dot(dz2)
    db2 = dz2.sum(axis=0)
    da1 = dz2.dot(net['W2'].T)
    dz1 = da1 * (a1 > 0)
    dW1 = x.T.dot(dz1)
    db1 = dz1.sum(axis=0)

    for k, grad in zip(['W1','b1','W2','b2','W3','b3'],
                       [dW1,db1,dW2,db2,dW3,db3]):
        net[k] -= lr * grad
