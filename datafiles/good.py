
import os
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import inv


def load_points_from_file(filename):
    points = pd.read_csv(filename, header=None)
    return points[0].values, points[1].values

########################################################################################################################
# Least Squares Methods

def least_squares(x_1, y_1):
    y = np.matrix(y_1).reshape(20, 1)
    x = np.matrix(x_1).reshape(20, 1)

    # Creating X and EXP matrix
    ones = np.ones((20, 1))
    squared = np.power(x, 2)
    cubed = np.power(x, 3)
    x = np.concatenate((ones, x, squared, cubed), axis=1)

    matrix = x.transpose() * x
    als = inv(matrix) * x.transpose() * y

    a_1 = als.item(0)
    b_1 = als.item(1)
    c_1 = als.item(2)
    d_1 = als. item(3)


    error_temp = 0
    for i in range(0, 20):
        error_temp += ((d_1 * (x_1[i])**3) + (c_1 * (x_1[i]**2) + b_1 * x_1[i] + a_1) - y_1[i]) ** 2

    return a_1, b_1, c_1, d_1, error_temp

def least_squares_linear(x_1, y_1):
    y = np.matrix(y_1).reshape(20, 1)
    x = np.matrix(x_1).reshape(20, 1)

    # Creating X and EXP matrix
    ones = np.ones((20, 1))
    x = np.concatenate((ones, x), axis=1)
    matrix = x.transpose() * x
    als = inv(matrix) * x.transpose() * y

    # basically: v = np.linalg.inv(x_e.T.dot(x_e)).dot(x_e.T).dot(y)

    a_1 = als.item(0)
    b_1 = als.item(1)

    error_temp = 0
    for i in range(0, 20):
        error_temp += ((b_1 * x_1[i] + a_1) - y_1[i]) ** 2

    return a_1, b_1, error_temp


def least_squares_sine(x_1, y_1):
    y = np.matrix(y_1).reshape(20, 1)
    x = np.matrix(x_1).reshape(20, 1)

    # Creating X and EXP matrix
    ones = np.ones((20, 1))
    sine = np.sin(x)
    x = np.concatenate((ones, sine), axis=1)

    matrix = x.transpose() * x
    als = inv(matrix) * x.transpose() * y

    a_1 = als.item(0)
    b_1 = als.item(1)


    error_temp = 0
    for i in range(0, 20):
        error_temp += ((np.sin(x_1[i]) * b_1 + a_1) - y_1[i]) ** 2

    return a_1, b_1, error_temp


########################################################################################################################
# MAIN FUNCTION


x, y = load_points_from_file(sys.argv[1])
assert len(x) == len(y)
assert len(x) % 20 == 0
num_segments = len(x) // 20

error = 0

if len(sys.argv) > 2:
    if sys.argv[2] == '--plot':

        colour = 'b'
        plt.set_cmap('Dark2')
        plt.scatter(x, y, c=colour)

        for i in range(num_segments):
            m = x[i * 20:(i * 20) + 20]
            n = y[i * 20:(i * 20) + 20]
            a_3, b_3, error_3 = least_squares_linear(m, n)
            a, b, c, d, error_1 = least_squares(m, n) # poly
            a_2, b_2, error_2 = least_squares_sine(m, n)

            if error_1 > error_2:  # Sine function
                error += error_2
                plt.plot(m, (b_2 * np.sin(m)) + a_2, 'r')
            else:

                if ((error_3 - error_1) / error_3) < 0.2:   # If relative error is less than 0.1%, use linear, the difference is noise!
                    error += error_3
                    plt.plot(m, (b_3 * m) + a_3, 'r')
                else:
                    error += error_1
                    plt.plot(m, d * (m ** 3) + c * (m ** 2) + b * m + a, 'r')

        plt.show()
        print(error)

else:   #   separate function for no-plotting, reduces time for large inputs
    for i in range(num_segments):
        m = x[i * 20:(i * 20) + 20]
        n = y[i * 20:(i * 20) + 20]
        a_3, b_3, error_3 = least_squares_linear(m, n)
        a, b, c, d, error_1 = least_squares(m, n)
        a_2, b_2, error_2 = least_squares_sine(m, n)

        if error_1 > error_2:  # Sine function
            error += error_2
        else:
            if (error_3 - error_1) / (error_1 + error_3) < 0.2:  # If relative error is less than 0.1%, use linear, the difference is noise!
                error += error_3

            else:
                error += error_1

    print(error)


# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------


import os
import sys
import math
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def load_points_from_file(filename):
    """Loads 2d points from a csv called filename
    Args:
        filename : Path to .csv file
    Returns:
        (xs, ys) where xs and ys are a numpy array of the co-ordinates.
    """
    points = pd.read_csv(filename, header=None)
    return points[0].values, points[1].values


def view_data_segments(xs, ys):
    """Visualises the input file with each segment plotted in a different colour.
    Args:
        xs : List/array-like of x co-ordinates.
        ys : List/array-like of y co-ordinates.
    Returns:
        None
    """
    assert len(xs) == len(ys)
    assert len(xs) % 20 == 0
    len_data = len(xs)
    num_segments = len_data // 20
    colour = np.concatenate([[i] * 20 for i in range(num_segments)])
    plt.set_cmap('Dark2')
    plt.scatter(xs, ys, c=colour, marker='x')
    plt.show()

def calcerror(Y,ys):
    error=0
    for i in range(len(ys)):
        error+=(Y[i]-ys[i])**2
    return error

def getVandermondeMatrix(xs,p):
    X=np.zeros([len(xs),p])
    X[:,0]=1
    for i in range(p):
        X[:,i]=list(map(lambda x : x**i, xs))
    return X

def getSineMatrix(xs):
    X=np.zeros([len(xs),2])
    X[:,0]=1
    for i in range(len(xs)):
        X[i,1]=math.sin(xs[i])
    return X

def applypolynomial(A, x):
    result=0
    for i in range(len(A)):
        result+=A[i]*x**i
    return result

def findbestfit(xs,ys,startpoint, p=4):
    X=getVandermondeMatrix(xs,p)
    A=np.linalg.inv(X.T.dot(X)).dot(X.T).dot(ys)
    #Tried linalg.inv but hit numerical instability so let's Gauss
    #A=np.linalg.solve(X.T.dot(X), X.T.dot(ys))
    Space=np.linspace(startpoint, xs[19], 100)
    Y=[applypolynomial(A,x) for x in Space]
    Xs=getSineMatrix(xs)
    As=np.linalg.solve(Xs.T.dot(Xs), Xs.T.dot(ys))
    Ys=As[0]+As[1]*np.sin(Space)
    sinerror=calcerror(As[0]+As[1]*np.sin(xs),ys)
    Xl=getVandermondeMatrix(xs,2)
    Al=np.linalg.inv(Xl.T.dot(Xl)).dot(Xl.T).dot(ys)
    Yl=Al[0]+Al[1]*Space
    linerror=calcerror(Al[0]+Al[1]*xs,ys)
    polerror=calcerror([applypolynomial(A,x) for x in xs],ys)

    if linerror<1.2*sinerror and (linerror<1.2*polerror or abs(A[3])<0.01):
        plt.plot(Space,Yl, c="r")
        #print('line')
        return linerror
    if sinerror<1.1*polerror:
        plt.plot(Space,Ys, c="r")
        #print('sine')
        return sinerror
    plt.plot(Space,Y, c="r")
    #print('pol')
    return polerror

xs, ys = load_points_from_file(sys.argv[1])
totalerror=0
for i in range(int(len(xs)/20)):
    temp=0
    if i!=0:
        temp=20*i-1
    totalerror+=findbestfit(xs[20*i:20*i+20],ys[20*i:20*i+20],xs[temp],4)
print(totalerror)

if len(sys.argv)==3 and sys.argv[2]=='--plot':
    view_data_segments(xs,ys) 



# #-------------------------------------------------------------------------
# #-------------------------------------------------------------------------
# #-------------------------------------------------------------------------




import sys
import numpy as np
try:
    from matplotlib import pyplot as plt
except ImportError:
    pass
    # import warnings
    # warnings.warn("--plot will not work")

import utilities


def least_squares_calc(x_ls, y_ls):
    return np.linalg.inv(x_ls.T.dot(x_ls)).dot(x_ls.T.dot(y_ls))


def least_squares_poly(x, y, n):
    """
    Will carry out least squares for a polynomial of degree n
    """
    x_ls = np.vander(x, increasing=True, N=n + 1)
    y_ls = y.T
    return least_squares_calc(x_ls, y_ls)


def least_squares_sin(x, y):
    """
    Will carry out least squares for an exponential
    """
    x_ls = np.vstack((np.ones(x.shape), np.cos(x))).T
    y_ls = y.T
    return least_squares_calc(x_ls, y_ls)


def func_fitter(x, y):
    """
    Finds the best fitting function
    Linear, Polynomial, Unknown?(Maybe Exponential)
    """
    # Temp just use degree 1 polynomials
    lin_A = least_squares_poly(x, y, 1)
    poly_A = least_squares_poly(x, y, 3)  # Do some tests to ensure it is best modelled by cubic
    sin_A = least_squares_sin(x, y)

    # Currently calculate each one and take the lowest error option.
    lin_func = poly_func(lin_A)
    lin_y = apply_funcs(x, [lin_func])
    lin_error = error(y, lin_y)

    polynomial_func = poly_func(poly_A)
    polynomial_y = apply_funcs(x, [polynomial_func])
    polynomial_error = error(y, polynomial_y)

    sin_func = sine_func(sin_A)
    sin_y = apply_funcs(x, [sin_func])
    sin_error = error(y, sin_y)

    if lin_error > sin_error and polynomial_error > sin_error:
        return sin_func
    elif polynomial_error > lin_error and sin_error > lin_error:
        return lin_func
    elif lin_error > polynomial_error and sin_error > polynomial_error:
        if abs(poly_A[2]) < 0.15 and abs(poly_A[3]) < 0.15 \
                or (polynomial_error / lin_error) >= 0.9:
            return lin_func
        return polynomial_func


def error(y1, y2):
    """
    Calculates the error between the actual y and predicted y
    """
    # error = 0
    # for x, y in zip(y1, y2):
    #     error += (x - y) ** 2
    # return error
    return np.sum(((y1) - y2) ** 2)


def poly_func(A):
    """
    Creates a partially evaluated function for a polynomial/linear
    """
    def func(xi):
        y = 0
        for i in range(len(A)):
            y += A[i] * (xi**i)
        return y
    return func


def sine_func(A):
    """
    Creates a partially evaluated function for an exponential
    """
    def func(xi):
        return A[0] + A[1] * np.cos(xi)
    return func


def get_model(x, y):
    """
    Given x and y it will genereate a model for the data points
    """
    funcs = []
    segments = len(x) // 20
    for i in range(segments):
        func = func_fitter(x[i * 20: i * 20 + 20], y[i * 20: i * 20 + 20])
        funcs.append(func)
    return funcs


def apply_funcs(x, funcs):
    """
    Will apply the model to x to generate a predicted y value
    """
    y = np.ones(x.shape)
    for i, func in enumerate(funcs):
        y[i * 20: i * 20 + 20] = func(x[i * 20: i * 20 + 20])
    return y


def main(args):
    args_len = len(args)
    if args_len in [1, 2]:
        x, y = utilities.load_points_from_file(args[0])
        model_funcs = get_model(x, y)
        model_y = apply_funcs(x, model_funcs)
        model_error = error(model_y, y)
        print(model_error)

    if args_len == 2:
        if args[1] == '--plot':
            # Print out graph
            segments = len(x) // 20
            for i in range(segments):
                segment_x = x[i * 20: i * 20 + 20]
                min_x = segment_x.min()
                max_x = segment_x.max()
                x_plot = np.linspace(min_x, max_x, 100)
                y_plot = model_funcs[i](x_plot)
                plt.plot(x_plot, y_plot)
            utilities.view_data_segments(x, y)


if __name__ == '__main__':
    args = sys.argv[1:]
    main(args)