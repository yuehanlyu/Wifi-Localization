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


def spotfi_algorithm_1_package_one(csi_matrix):
    R = abs(csi_matrix)
    phase_matrix = np.vstack((np.unwrap(map(phase,csi_matrix[0,:])),np.unwrap(map(phase,csi_matrix[1,:])),np.unwrap(map(phase,csi_matrix[2,:]))))
    fit_X = np.concatenate((np.linspace(1,30,30),np.linspace(1,30,30),np.linspace(1,30,30)))
    fit_Y = np.concatenate((np.unwrap(map(phase,csi_matrix[0,:])),np.unwrap(map(phase,csi_matrix[1,:])),np.unwrap(map(phase,csi_matrix[2,:]))))
    tau_offset = np.polyfit(fit_X,fit_Y,1)[0]
    C = np.zeros((3,30),dtype=np.complex_)
    for m in range(phase_matrix.shape[0]):
        for n in range(phase_matrix.shape[1]):
            C[m,n] = np.exp(complex(0,phase_matrix[m,n] - (n)*tau_offset))
    # csi_matrix_clean = np.multiply(R,C)
    return C, tau_offset

def spotfi_algorithm_1(csi_matrix,C):
    R = abs(csi_matrix)
    csi_matrix_clean = np.multiply(R,C)
    return csi_matrix_clean


def smooth_csi(csi):
    smoothed_csi = np.zeros((30, 32), dtype=np.complex_)
    # Antenna 1 (values go in the upper left quadrant)
    m = 0
    for ii in range(0, 15):
        n = 0
        for j in range(ii, ii+16):
            smoothed_csi[m, n] = csi[0, j]
            n = n + 1
        m = m + 1

    # Antenna 2

    # # Bottom left of smoothed csi matrix
    for ii in range(0, 15):
        n = 0
        for j in range(ii, ii+16):
            smoothed_csi[m, n] = csi[1, j] # 2 + sqrt(-1) * j;
            n = n + 1
        m = m + 1

    # Top right of smoothed csi matrix
    m = 0;
    for ii in range(0, 15):
        n = 16
        for j in range(ii, ii+16):
            smoothed_csi[m, n] = csi[1, j]  #2 + sqrt(-1) * j;
            n = n + 1
        m = m + 1

    # Antenna 3 (values go in the lower right quadrant)
    for ii in range(0, 15):
        n = 16
        for j in range(ii, ii+16):
            smoothed_csi[m, n] = csi[2, j]   #3 + sqrt(-1) * j;
            n = n + 1
        m = m + 1

    return smoothed_csi

def compute_steering_vector(theta, tau, freq, sub_freq_delta, antenna_distance):
    steering_vector = np.zeros(30,dtype=np.complex_)
    k = 0
    base_element = 1
    for ii in np.linspace(0,1,2):
        for jj in np.linspace(0,14,15):
            steering_vector[k] = base_element * pow(omega_tof_phase(tau, sub_freq_delta),(jj-1))
            k = k+1
        base_element = base_element *  phi_aoa_phase(theta, freq, antenna_distance)
    return steering_vector

def omega_tof_phase(tau, sub_freq_delta):
    time_phase = np.exp(-1j * 2 * math.pi * sub_freq_delta * tau)
    return time_phase
def phi_aoa_phase(theta, frequency, d):
    c = 3.0 * pow(10,8)
    # Convert to radians
    theta = theta / 180 * math.pi
    angle_phase = np.exp(-1j * 2 * math.pi * d * math.sin(theta) * (frequency / c))
    return angle_phase

def detect_peaks(image):
    neighborhood = generate_binary_structure(2,2)
    local_max = maximum_filter(image, footprint=neighborhood)==image
    background = (image==0)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)
    detected_peaks = local_max ^ eroded_background
    return detected_peaks

def aoa_tof_music(x, antenna_distance, frequency, sub_freq_delta, theta_range, tau_range):
    R = np.dot(x,x.conj().T)
    w, v = LA.eig(R)
    w = np.real(w)
    w = w/max(w)

    idx = (-w).argsort()[::-1]
    w = w[idx]
    v = v[:,idx]

    start_index = len(w)-2
    end_index = start_index - 10
    decrease_ratios = np.zeros(start_index - end_index + 1)

    k=0
    for ii in range(28, 17, -1):
        temp_decrease_ratio = w[ii + 1] / w[ii]
        decrease_ratios[k] = temp_decrease_ratio
        k = k + 1

    max_decrease_ratio_index = np.argmax(decrease_ratios)
    index_in_eigenvalues = len(w) - max_decrease_ratio_index
    num_computed_paths = len(w) - index_in_eigenvalues + 1

    # Estimate noise subspace
    column_indices = range(0, (len(w) - num_computed_paths))
    eigenvectors = v[:, list(column_indices)]

    # Peak search
    # theta_range = np.linspace(-90,90,91)
    # tau_range = np.linspace(0,3000 * pow(10,-9),61)
    Pmusic = np.zeros((len(theta_range), len(tau_range)))

    for ii in range(0, len(theta_range)):
        for jj in range(0, len(tau_range)):
            steering_vector = compute_steering_vector(theta_range[ii], tau_range[jj],frequency, sub_freq_delta, antenna_distance)
            PP = np.dot(np.dot(steering_vector.conj().T,eigenvectors),np.dot(eigenvectors.conj().T,steering_vector))
            Pmusic[ii,jj] = 10*math.log(np.abs(1/PP),10)
    detected_peaks = detect_peaks(Pmusic)
    maximum_idx_array = np.zeros(2)
    for i in range(detected_peaks.shape[0]):  #i: idx of theta
        for j in range(detected_peaks.shape[1]): #j: idx of tau
            if detected_peaks[i,j]==True:
                maximum_idx_array = np.vstack((maximum_idx_array,np.array([i,j])))
    maximum_idx_array=maximum_idx_array[1:,]
    return maximum_idx_array

def csi_plot(theta1, theta2, d):
    rad1 = theta1*math.pi/180
    rad2 = theta2*math.pi/180

    if theta1 == 0:
        x = 1
        k2 = math.tan(0.5*math.pi+rad2)
        y = -k2*d
    elif theta2 == 0:
        x = 1 + d
        k1 = math.tan(0.5*math.pi+rad1)
        y = k1*d
    else:
        k1 = math.tan(0.5*math.pi+rad1)
        k2 = math.tan(0.5*math.pi+rad2)
        x = d*k2/(k2-k1)+1
        y = d*k1*k2/(k2-k1)

    X = [1,1+d]
    Y = [0,0]
    pp.scatter(X,Y,color = 'deepskyblue')
    pp.scatter(x,y,color = 'salmon')
    pp.axis(ymin=0, ymax=1.2*y)
    pp.text(x,y+1,'({0},{1})'.format(x,y))
    return x,y