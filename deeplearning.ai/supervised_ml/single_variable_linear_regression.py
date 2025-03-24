# _s(i) notation means individual i instance in summation i.e. for i=3, for i=m-2 etc.
'''    
# algorithm for linear regression of one model variable: fwb(x_s(i)) = wx_s(i) + b = wx[i] + b
algorithm is the same as slope intercept equation y = ax + b
y = f(w,b) = wx + b
'''
def f_wb(w, x, b):                      #f_wb computes wx[i] + b for a single x[i]
# x is x[i] in list or array
# w and b are floats
    f_wb = (w * x) + b
    return f_wb

'''
cost function algorithm for one variable using Mean Squared Error MSE

              m-1
J(w,b) = 1/2m Σ (fwb(x_s(i)) - y_s(i))**2
              i=0
           

              m-1
J(w,b) = 1/2m Σ (wx_s(i) + b - y_s(i))**2
              i=0
'''

def compute_cost(x, y, w, b, m):
# x is list or array; input data with m values
# y is list or array; target data with m values
# m is list or array
# w is float, model parameter
# b is float, model parameter
# m is integer
    cost = 0.0
    for i in range(m):
        f_wb_xi = f_wb(w, x[i], b)
        cost_i = 1 / (2*m) * (f_wb_xi - y[i])**2
        cost = cost + cost_i

    return cost

'''
gradient descent, which must be calculated simultaneously
d is actually ∂ for partial derivative

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

def compute_gradient(w, x, y, b, m):
# x is list or array; input data with m values
# y is list or array; target data with m values
# m is list or array
# w is float, model parameter
# b is float, model parameter
# m is integer
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        f_wb_xi = f_wb(w, x, b)
        dj_dw_i = (f_wb_xi - y[i]) * x
        dj_db_i = f_wb_xi - y[i]
        dj_dw += dj_dw_i
        dj_db += dj_db_i

    return dj_dw, dj_db

'''
gradient descent to find ideal values for w and b using iteration

w = w - α * dJ(w,b) / dw 
b = b - α * dJ(w,b) / dw 

            m-1
w = w - α/m Σ (wx_s(i) + b - y_s(i)) * x
            i = 0

            m-1
b = b - α/m Σ (wx_s(i) + b - y_s(i))

'''

def gradient_descent(x, y, w_init, b_init, alpha, num_iters):
# x is list or array; input data with m values
# y is list or array; target data with m values
# w_input is float for initial value of model parameter w
# b_input is float for initial value of model parameter b
# alpha (α) is float representing learning rate
# num_iters is integer for number of iterations to run gradient descent   
    import math
    J_history = []   # list of J(w,b) values (costs). Only to be used for plotting visualization
    p_history = []   # list of w,b pairs. Used only for plotting/graphing visualization
    b = b_init       # start with initial value of b. Subsequent values will be modified via gradient descent
    w = w_init

    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(x, y, w, b)    # calculate gradient. Note that you are passing arrays for x and y
        w = w - alpha * dj_db                          # tune w using gradient descent formula w = w - α * dJ(w,b) / dw 
        b = b - alpha * dj_db                          # tune b using gradient descent formula b = b - α * dJ(w,b) / dw 

        #the rest is storing data points for visualization
        if i < 100000:                                 # prevent resource exhaustion
            J_history.append( compute_cost(x, y, w , b) ) # get cost for current w,b gradient values
            p_history.append([w,b])                       # current w,b gradient descent values
        '''    
        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")
        '''
    return w, b, J_history, p_history           #return w and J,w history for graphing

compute_cost(10, 5, 4, 3, 8)
compute_gradient(10, 5, 4, 3, 8)