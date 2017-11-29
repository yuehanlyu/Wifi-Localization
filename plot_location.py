from __future__ import print_function, absolute_import, division
import math
import pandas as pd
import matplotlib.patches as mpatches
from cmath import phase
import numpy as np
import warnings
from numpy import linalg as LA
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
import matplotlib.pyplot as pp
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore", category=DeprecationWarning)


if __name__ == '__main__':

    theta_left = 32.0 + 90
    theta_mid = 14 + 90
    theta_right = -42 + 90

    rad_left = theta_left * math.pi / 180
    rad_mid = theta_mid * math.pi / 180
    rad_right = theta_right * math.pi / 180

    k_left = math.tan(rad_left)
    k_mid = math.tan(rad_mid)
    k_right = math.tan(rad_right)

    x = np.zeros(3, dtype=float)
    y = np.zeros(3, dtype=float)

    x[0] = -2.23 * k_right / (k_right - k_mid)
    y[0] = 2.33 * k_right * k_mid / (k_mid - k_right)
    x[1] = 2.23 * k_left / (k_left - k_mid)
    y[1] = -2.33 * k_left * k_mid / (k_mid - k_left)

    x[2] = 2.33 * (k_left + k_right) / (k_left - k_right)
    y[2] = (x[2] - 2.33) * k_left

    print (x)
    print (y)
    X = [-2.33, 0, 2.33]
    Y = [0, 0, 0]
    Tar_x = [0]
    Tar_y = [2.54]
    pp.scatter(X, Y, color='blue',label="Signal receiver")
    pp.scatter(x, y, color='salmon', label = "Candidate location")
    pp.scatter(Tar_x, Tar_y, color='green', label = "True location")
    pp.plot([X[0],x[2]], [Y[0],y[2]],'b--')
    pp.plot([X[1], x[1]], [Y[1], y[1]], 'b--')
    pp.plot([X[2], x[1]], [Y[2], y[1]], 'b--')
    pp.title("Localization using only CSI values (manually clustering)")
    pp.xlabel("x (in meters)")
    pp.ylabel("y (in meters)")
    pp.legend(loc = 0)

    pp.axis(ymin=0, ymax=max(Tar_y[0],max(y)))
    pp.show()
