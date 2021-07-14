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

    # extend the first column with 1s
    ones = np.ones(x.shape)
    xs = np.column_stack((ones, x))

    v = np.linalg.inv(xs.T.dot(xs)).dot(xs.T).dot(y)

    a = v.item(0)
    b = v.item(1)

    calc_y = (b* xx )+ a
    return calc_y

def poly_least_squares(xx, yy):
    y = np.matrix(yy).reshape(20, 1)
    x = np.matrix(xx).reshape(20, 1)
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

    return calc_y

# unkown in sine
def unknown_least_squares(xx, yy):

    y = np.matrix(yy).reshape(20, 1)
    x = np.matrix(xx).reshape(20, 1)
    one = np.ones(x.shape)
    unknown = np.sin(x)
    xs = np.column_stack((one, unknown))
    v = np.linalg.inv(xs.T.dot(xs)).dot(xs.T).dot(y)

    a = v.item(0)
    b = v.item(1)

    calc_y = (b* np.sin(xx)) + a
    return calc_y


def calc_error(y, calc_y ):
    return np.sum(((calc_y) - y) ** 2)


def main ():
    data = load_points_from_file(sys.argv[1])
    x_coordinates = np.array_split(data[0], len(data[0])/20)
    y_coordinates = np.array_split(data[1], len(data[1])/20)
    segments = len(x_coordinates)
    totalError = 0

    for i in range(segments):
        # print(x_coordinates[i],"\n")

        linear_Y  = linear_least_squares(x_coordinates[i], y_coordinates[i])
        poly_Y  = poly_least_squares(x_coordinates[i], y_coordinates[i])
        sin_Y  = unknown_least_squares(x_coordinates[i], y_coordinates[i])

        linear_error = calc_error(y_coordinates[i], linear_Y )
        poly_error = calc_error(y_coordinates[i], poly_Y)
        unknown_error = calc_error(y_coordinates[i], sin_Y)

        if poly_error > unknown_error:
            totalError += unknown_error
            plt.plot(x_coordinates[i], sin_Y, c="r" )
            print("sine")

        else:
            if((linear_error - poly_error) / linear_error) < 0.15:
            # if(linear_error < poly_error):
                print("linear")   
                totalError += linear_error
                plt.plot(x_coordinates[i], linear_Y, c="r" )

            else:
                print("poly")
                totalError += poly_error
                plt.plot(x_coordinates[i], poly_Y, c="r" )

    print(totalError)

    if len(sys.argv)==3 and sys.argv[2] == "--plot":
        view_data_segments(data[0], data[1])

main()


