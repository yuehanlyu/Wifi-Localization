from __future__ import print_function, absolute_import, division
import math
import pandas as pd
from cmath import phase
import numpy as np
import warnings
from numpy import linalg as LA
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
import matplotlib.pyplot as pp
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore", category=DeprecationWarning)

def dbinv(x):
    ret = 10**(x/10)
    return ret

def db(x):
    try: # if input is a series
        ret = 10.*np.log10(list(x))
        return ret
    except: # if input is a number
        ret = 10.*np.log10(x)
        return ret

def rssi(file_data):
    rssi_mag = dbinv(file_data["rssi_a"])+dbinv(file_data["rssi_b"])+dbinv(file_data["rssi_c"])
    rssi_ = db(rssi_mag) - 44 - file_data['agc']
    return rssi_

def rssi2dist(rssi_,A,n):
    dist = 10**((A-rssi_)/(10*n))
    return dist