'''
_s(i) notation means individual i instance in summation i.e. for i=3, for i=m-2 etc.

formula for linear regression for one variable: fwb(x_s(i)) = wx_s(i) + b

cost function algorithm for one variable using Mean Squared Error MSE

              m-1
J(w,b) = 1/2m Σ (fwb(x_s(i)) - y_s(i))**2
              i=0
           

              m-1
J(w,b) = 1/2m Σ (wx_s(i) + b - y_s(i))**2
              i=0

'''         

def f_wb(w, x, b):
    f_wb = (w * x) + b
    return f_wb

def compute_cost(x, y, w, b, m):
    cost = 0.0
    for i in range(m):
        f_wb_i = f_wb(w, x, b)
        cost_i = 1 / (2*m) * (f_wb_i - y)**2
        cost = cost + cost_i

    return cost

compute_cost(10, 5, 4, 3, 8)