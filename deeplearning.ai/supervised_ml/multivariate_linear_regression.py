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
gradient for multiple variables means 

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

gradient descent to find ideal values for w and b using iteration
{ 
w = w - α * dJ(w,b) / dw   
b = b - α * dJ(w,b) / dw 
} for j = 0..n - 1
'''