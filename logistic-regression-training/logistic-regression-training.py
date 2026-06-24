import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    samples,features = X.shape;
    w = np.zeros(features)
    b=0.0
    
    for i in range(0,steps):

        z = np.dot(X,w)+b

        pred = _sigmoid(z)
        
        dw = np.dot(X.T,(pred-y))*(1/samples)
        db = np.sum(pred-y)*(1/samples)

        w=w-lr*dw
        b=b-lr*db

    return w,b
    pass