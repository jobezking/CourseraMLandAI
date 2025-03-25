# _s(i) notation means individual i instance in summation i.e. for i=3, for i=m-2 etc.
'''    
x1 (size in feet squared)	| 	x2 (number of bedrooms) | x3 number of floors   | y (price in $1000s)
1	2100			        |	4                            |       1               | 400
2	1400			        |	2                            |       2               | 230
3	 850			        |	3                            |       1               | 315	
previous general algorithm for one variable: fw,b(x) = wx + b. 
general algorithm for multiple variables (using above to illustrate):
fw,b(x) = w1x1 + w2x2 + w3x3 + b        more generally: fw,b(x) = w1x1 + w2x2 + w3x3 + ... wnxn + b
the reason is that each x is going to need its own w to describe the slope of its line. 
# So the algorithm for linear regression of multiple variable model will need to represent w and x as vectors (or arrays):
#  fw_s(i)b(x_s(i)) = w_s(i) ⋅ x_s(i) + b = w[i] ⋅ x[i] + b
'''
import numpy as np
import math
import copy

def m_f_wb(x, w, b):                      #f_wb computes wx[i] + b for a single x[i]
# x is x[i] in list or array
# w is w[i] in list or array
# b is a float
    _m_f_wb = np.dot(x, w) + b            # computes dot product, or vector multiplication
    return _m_f_wb

'''
cost function algorithm for multiple variables using Mean Squared Error MSE

              m-1
J(w,b) = 1/2m Σ (fwb(x_s(i)) - y_s(i))**2
              i=0
           
              m-1
J(w,b) = 1/2m Σ (w ⋅ x_s(i) + b - y_s(i))**2
              i=0
'''
def m_compute_cost(x, y, w, b, m):
# x is list or array; input data with m values
# y is list or array; target data with m values
# m is list or array
# w is float, model parameter
# b is float, model parameter
# m is integer
    cost_sum = 0.0
    for i in range(m):
        _m_f_wb_xi = m_f_wb(x[i], w, b)     #compute fw,b(x)
        cost_i = (_m_f_wb_xi - y[i])**2     # compute the individual interior value
        cost_sum = cost_sum + cost_i        # sum all the interior values 
    cost = (1 / (2 * m)) * cost_sum         # multiply summed interior values by outer value
    return cost

'''
gradient for multiple variables means j = 0..n - 1

                   m-1
dJ(w,b) / dw = 1/m Σ (fwb(x_s(i)) - y_s(i)) * x
                   i = 0

                   m-1
dJ(w,b) / db = 1/m Σ (fwb(x_s(i)) - y_s(i))
                   i = 0

which equates to

                   m-1
dJ(w,b) / dw = 1/m Σ (wx_s(i) + b - y_s(i)) * x
                   i = 0

                   m-1
dJ(w,b) / db = 1/m Σ (wx_s(i) + b - y_s(i))
                   i = 0
'''
def m_compute_gradient(x, y, w, b):
# x: (m,n) matrix with m values and n features; data
# y: (m) array with m values; target values
# w: (n) array, model parameters with n features
# b: int, model parameter
# m: integer, number of rows

   m,n = x.shape   #x is a m x n matrix. this function returns the m and n count
   dj_dw = np.zeros((n,))    # The gradient of the cost w.r.t. the parameters w. 
   dj_db = 0.0               # The gradient of the cost w.r.t. the parameter b.

   for i in range(m):                             
        err = (np.dot(x[i], w) + b) - y[i]   # cannot call m_f_wb() or compute_cost() because a j loop has to be done for the n variables
        for j in range(n):                         
            dj_dw[j] = dj_dw[j] + err * x[i, j]    
        dj_db = dj_db + err                        
    
   dj_dw = dj_dw / m                                
   dj_db = dj_db / m                                
        
   return dj_db, dj_dw
'''
gradient descent to find ideal values for w and b using iteration
{ 
w = w - α * dJ(w,b) / dw   
b = b - α * dJ(w,b) / dw 
} for j = 0..n - 1'
'''

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
    """
    Performs batch gradient descent to learn w and b. Updates w and b by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      X (ndarray (m,n))   : Data, m examples with n features
      y (ndarray (m,))    : target values
      w_in (ndarray (n,)) : initial model parameters  
      b_in (scalar)       : initial model parameter
      cost_function       : function to compute cost
      gradient_function   : function to compute the gradient
      alpha (float)       : Learning rate
      num_iters (int)     : number of iterations to run gradient descent
      
    Returns:
      w (ndarray (n,)) : Updated values of parameters 
      b (scalar)       : Updated value of parameter 
      """
    
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    #p_history = []   # list of w,b pairs not used this time because J_history will already be a matrix
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in
    
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db,dj_dw = gradient_function(X, y, w, b)    

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw                
        b = b - alpha * dj_db                
      
        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append( cost_function(X, y, w, b))

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")
        
    return w, b, J_history #return final w,b and J history for graphing