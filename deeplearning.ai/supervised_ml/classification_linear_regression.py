'''
fw,b(X)  where w and X are vectors
z = w ⋅ x + b
where
g(z) = 1 / (1+e**-z)
'''

import numpy as np

def sigmoid(z):
#Compute the sigmoid of z
#z (ndarray): A scalar, numpy array of any size.
# Returns: g (ndarray): sigmoid(z), with the same shape as z
         
    g = 1/(1+np.exp(-z))
    return g