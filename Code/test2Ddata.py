#!/usr/bin/env python

""" made up data
 Created: 11/16/2020
"""

__author__ = "Mike Hagenow"

import numpy as np
import matplotlib.pyplot as plt
from LSQ import lsq
from lSVM import lsvm

# Fabricated 2D data to test the LSQ and SVM methods

def test():

    X_plus = np.array([[1.0, 3.0, 1.0],
                       [2.0, 4.0, 1.0],
                       [3.0, 5.0, 1.0],
                       [2.5, 5.5, 1.0],
                       [1.5,3.5, 1.0]])

    X_minus = np.array([[1.0+1*0.5, 3.0-1*0.5, 1.0],
                       [2.0+1*0.5, 4.0-1*0.5, 1.0],
                       [3.0+1*0.5, 5.0-1*0.5, 1.0],
                       [2.5, -1.0, 1.0],
                       [1.5, 1.0, 1.0]])

    X = np.vstack((X_plus, X_minus))
    y = np.vstack((np.ones((np.shape(X_plus)[0], 1)), -np.ones((np.shape(X_minus)[0], 1))))

    w = lsq(X,y,0.0)
    w2 = lsvm(X,y,0.0)

    print("w:",w)
    print("w2:",w2)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.scatter(X_plus[:,0],X_plus[:,1],color='r')
    ax1.scatter(X_minus[:,0],X_minus[:,1],color='b')
    ax1.plot([1.0, 3.0],[-w[2]/w[1]-w[0]/w[1]*1.0, -w[2]/w[1]-w[0]/w[1]*3.0], color='g')
    ax1.plot([1.0, 3.0],[-w2[2]/w2[1]-w2[0]/w2[1]*1.0, -w2[2]/w2[1]-w2[0]/w2[1]*3.0], color='y')
    ax1.axis("equal")

    X_plus = np.array([[1.0, 7.0, 1.0],
                       [2.0, 5.0, 1.0],
                       [3.0, 3.0, 1.0],
                       [-1, 5.5, 1.0],
                       [0.0, 2.0, 1.0]])

    X_minus = np.array([[1.0 + 4 * 0.5, 7.0 + 2.0 * 0.5, 1.0],
                        [2.0 + 4 * 0.5, 5.0 + 2.0 * 0.5, 1.0],
                        [3.0 + 4 * 0.5, 3.0 + 2.0 * 0.5, 1.0],
                        [4, 8.0, 1.0],
                        [5.0, 10.0, 1.0]])

    X = np.vstack((X_plus, X_minus))
    y = np.vstack((np.ones((np.shape(X_plus)[0], 1)), -np.ones((np.shape(X_minus)[0], 1))))

    w = lsq(X, y, 0.0)
    w2 = lsvm(X, y, 0.0)

    print("w:", w)
    print("w2:", w2)

    ax2.scatter(X_plus[:, 0], X_plus[:, 1], color='r')
    ax2.scatter(X_minus[:, 0], X_minus[:, 1], color='b')
    ax2.plot([1.0, 3.0], [-w[2] / w[1] - w[0] / w[1] * 1.0, -w[2] / w[1] - w[0] / w[1] * 3.0], color='g')
    ax2.plot([1.0, 3.0], [-w2[2] / w2[1] - w2[0] / w2[1] * 1.0, -w2[2] / w2[1] - w2[0] / w2[1] * 3.0], color='y')
    ax2.axis("equal")

    plt.show()

if __name__ == "__main__":
    test()




