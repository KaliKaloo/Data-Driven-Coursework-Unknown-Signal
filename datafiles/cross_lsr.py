import os
import random

import sys
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
    plt.scatter(xs, ys, c=colour)
    plt.show()


def linear_least_squares(amount, xx, yy):
    y = np.matrix(yy).reshape(amount, 1)
    x = np.matrix(xx).reshape(amount, 1)

    # extend the first column with 1s
    ones = np.ones(x.shape)
    xs = np.column_stack((ones, x))

    v = np.linalg.inv(xs.T.dot(xs)).dot(xs.T).dot(y)

    a = v.item(0)
    b = v.item(1)

    calc_y = (b* xx )+ a
    return calc_y, v


def poly_least_squares(amount, xx, yy):
    y = np.matrix(yy).reshape(amount, 1)
    x = np.matrix(xx).reshape(amount, 1)
    ones = np.ones(x.shape)
    squared = np.power(x, 2)
    cubed = np.power(x, 3)
    # quartic = np.power(x, 4)
    # quintic = np.power(x, 5)

    # xs = np.column_stack((ones, x,squared))
    xs = np.column_stack((ones, x,squared, cubed))
    # xs = np.column_stack((ones, x, squared, cubed, quartic))
    # xs = np.column_stack((ones, x,squared, cubed, quartic, quintic))

    v = np.linalg.inv(xs.T.dot(xs)).dot(xs.T).dot(y)
    a = v.item(0) 
    b = v.item(1) 
    c = v.item(2) 
    d = v.item(3) 
    # e = v.item(4) 
    # f = v.item(5) 

    # calc_y = (c*(xx**2)) + (b*(xx)) + a
    calc_y = (d * (xx**3))+ (c*(xx**2)) + (b*(xx)) + a
    # calc_y = (e * (xx**4))+ (d * (xx**3))+ (c*(xx**2)) + (b*(xx)) + a
    # calc_y = (f * (xx**5)) + (e * (xx**4))+ (d * (xx**3))+ (c*(xx**2)) + (b*(xx)) + a
    return calc_y, v


# unkown in sine
def unknown_least_squares(amount, xx, yy):
    y = np.matrix(yy).reshape(amount, 1)
    x = np.matrix(xx).reshape(amount, 1)
    one = np.ones(x.shape)
    unknown = np.sin(x)
    xs = np.column_stack((one, unknown))
    v = np.linalg.inv(xs.T.dot(xs)).dot(xs.T).dot(y)

    a = v.item(0)
    b = v.item(1)

    calc_y = (b* np.sin(xx)) + a
    return calc_y, v


def calc_error(y, calc_y ):
    return np.sum(((calc_y) - y) ** 2)


# to pick function type
def cross_validate(seg_num, xs, ys):
    x_training, y_training = np.empty(19), np.empty(19)
    x_validation, y_validation = np.empty(1), np.empty(1)
    total_linear_Error = []
    total_poly_Error = []
    total_sin_Error = []
    leave_out = 0   
    
    for i in xs: 
        j=0
        for i in range(20): 
            if i != leave_out:
                x_training[j] = xs[i]
                y_training[j] = ys[i]
                j+=1
        x_validation = xs[leave_out]
        y_validation = ys[leave_out]

    #-------------------------------------
        linear_Y_train, linear_params = linear_least_squares(19, x_training, y_training)
        poly_Y_train, poly_params  = poly_least_squares(19, x_training, y_training)
        sin_Y_train, sin_params  = unknown_least_squares(19, x_training, y_training)

        linear_Y_validate = (linear_params.item(1)* x_validation ) + linear_params.item(0)
        poly_Y_validate = (poly_params.item(3) * (x_validation**3))+ (poly_params.item(2)*(x_validation**2)) + (poly_params.item(1)*(x_validation)) + poly_params.item(0)
        sin_Y_validate = (sin_params.item(1)* np.sin(x_validation)) + sin_params.item(0)

        total_linear_Error.append(calc_error( y_validation, linear_Y_validate))
        total_poly_Error.append(calc_error( y_validation, poly_Y_validate))
        total_sin_Error.append(calc_error( y_validation, sin_Y_validate))

        leave_out += 1   
    #-------------------------------------

    smallest_error_linear = sum(total_linear_Error)/20
    smallest_error_poly = sum(total_poly_Error)/20
    smallest_error_sin = sum(total_sin_Error)/20

    min_total_error = min (smallest_error_linear, smallest_error_poly, smallest_error_sin)
    
    chosen_function = ""
    if min_total_error == smallest_error_linear:
        chosen_function = "linear"
    elif min_total_error == smallest_error_poly:
        chosen_function="poly"
    else:
        chosen_function="sine"

    return chosen_function


def main ():
    data = load_points_from_file(sys.argv[1])
    x_coordinates = np.array_split(data[0], len(data[0])/20)
    y_coordinates = np.array_split(data[1], len(data[1])/20)
    segments = len(x_coordinates)
    totalError = 0

    for i in range(segments):#
        # pass x and y coordinates for segment to cross validate and choose function type
        chosen_function = cross_validate(i, x_coordinates[i], y_coordinates[i]);
        
        print(chosen_function)
        
        if (chosen_function == "linear"):
            linear_Y, linear_params = linear_least_squares(20, x_coordinates[i], y_coordinates[i])
            totalError += calc_error(y_coordinates[i], linear_Y )
            plt.plot(x_coordinates[i], linear_Y, c="r" )

        elif (chosen_function =="poly"):
            poly_Y, poly_params  = poly_least_squares(20, x_coordinates[i], y_coordinates[i])
            totalError += calc_error(y_coordinates[i], poly_Y)
            plt.plot(x_coordinates[i], poly_Y, c="r" )

        else:
            sin_Y, sin_params  = unknown_least_squares(20, x_coordinates[i], y_coordinates[i])
            totalError += calc_error(y_coordinates[i], sin_Y)
            plt.plot(x_coordinates[i], sin_Y, c="r" )
            
    print(totalError)

    if len(sys.argv)==3 and sys.argv[2] == "--plot":
        view_data_segments(data[0], data[1])

main()


