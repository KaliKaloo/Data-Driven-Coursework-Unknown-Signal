import os
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


def linear_least_squares(xx, yy):
    y = np.matrix(yy).reshape(20, 1)
    x = np.matrix(xx).reshape(20, 1)
    ones = np.ones(x.shape)

    xs = np.column_stack((ones, x))
    coefficients = least_squares_formula(xs, y)

    a = coefficients.item(0)
    b = coefficients.item(1)

    calc_y = (b* xx )+ a
    return calc_y


def poly_least_squares(xx, yy):
    y = np.matrix(yy).reshape(20, 1)
    x = np.matrix(xx).reshape(20, 1)
    ones = np.ones(x.shape)
    twos = np.power(x, 2)
    threes = np.power(x, 3)

    xs = np.column_stack((ones, x,twos, threes))
    coefficients = least_squares_formula(xs, y)

    a = coefficients.item(0) 
    b = coefficients.item(1) 
    c = coefficients.item(2) 
    d = coefficients.item(3) 

    calc_y = (d * (xx**3))+ (c*(xx**2)) + (b*(xx)) + a
    return calc_y


# unkown in sine
def unknown_least_squares(xx, yy):
    y = np.matrix(yy).reshape(20, 1)
    x = np.matrix(xx).reshape(20, 1)
    ones = np.ones(x.shape)
    unknown = np.sin(x)

    xs = np.column_stack((ones, unknown))
    coefficients = least_squares_formula(xs, y)

    a = coefficients.item(0)
    b = coefficients.item(1)

    calc_y = (b* np.sin(xx)) + a
    return calc_y


def least_squares_formula(x, y):
    return np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)


def calc_error(y, calc_y ):
    return np.sum(((calc_y) - y) ** 2)


def main ():
    data = load_points_from_file(sys.argv[1])
    x_coordinates = np.array_split(data[0], len(data[0])/20)
    y_coordinates = np.array_split(data[1], len(data[1])/20)
    segments = len(x_coordinates)
    totalError = 0

    for i in range(segments):
        linear_Y  = linear_least_squares(x_coordinates[i], y_coordinates[i])
        poly_Y  = poly_least_squares(x_coordinates[i], y_coordinates[i])
        sin_Y  = unknown_least_squares(x_coordinates[i], y_coordinates[i])

        linear_error = calc_error(y_coordinates[i], linear_Y )
        poly_error = calc_error(y_coordinates[i], poly_Y)
        unknown_error = calc_error(y_coordinates[i], sin_Y)

        if unknown_error < poly_error:
            plt.plot(x_coordinates[i], sin_Y, c="g" )
            totalError += unknown_error

        else:
            if((linear_error - poly_error) / linear_error) < 0.15:
                plt.plot(x_coordinates[i], linear_Y, c="r" )
                totalError += linear_error

            else:
                plt.plot(x_coordinates[i], poly_Y, c="b" )
                totalError += poly_error

    print(totalError)

    if len(sys.argv)==3 and sys.argv[2] == "--plot":
        view_data_segments(data[0], data[1])


main()