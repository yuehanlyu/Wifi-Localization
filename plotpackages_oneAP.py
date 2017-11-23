# coding: utf-8
from __future__ import print_function, absolute_import, division
import pandas as pd
import scipy.cluster.hierarchy as sch
import load_csi_data
import numpy as np
import spotfi_algorithms
import math
from tqdm import tqdm
import warnings
import matplotlib.pyplot as pp
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore", category=DeprecationWarning)

if __name__ == '__main__':
    file_data = load_csi_data.read_bf_file('./sample_data/csi_mid_3.dat')
    antenna_distance = 0.15
    frequency = 2.412 * pow(10, 9)
    sub_freq_delta = 3125
    theta_range = np.linspace(-90,90,91)
    tau_range = np.linspace(0,3000 * pow(10,-9),61)

    result_aoas = np.array([0.0])

    csi_entry = file_data.loc[0]
    csi = load_csi_data.get_scale_csi(csi_entry)
    csi_matrix = csi[0, :, :]
    C, tau_offset = spotfi_algorithms.spotfi_algorithm_1_package_one(csi_matrix)

    all_maximum_idx_array = np.zeros(2)
    #for i in tqdm(range(file_data.shape[0])):
    for i in tqdm(range(20)):
        csi_entry = file_data.loc[i]
        csi = load_csi_data.get_scale_csi(csi_entry)
        # run algorithm 1 for the first package
        csi_matrix = csi[0, :, :]
        csi_matrix_clean = spotfi_algorithms.spotfi_algorithm_1(csi_matrix, C)
        # return the smoothed_csi matrix
        smoothed_csi = spotfi_algorithms.smooth_csi(csi_matrix_clean)
        maximum_idx_array = spotfi_algorithms.aoa_tof_music(smoothed_csi, antenna_distance, frequency,
                                                            sub_freq_delta, theta_range, tau_range)
        all_maximum_idx_array = np.vstack((all_maximum_idx_array, maximum_idx_array))

    theta = np.linspace(-90, 90, 91)
    tau = np.linspace(0, 3000 * pow(10, -9), 61)

    candicate_aoa_tof = np.zeros(2)
    for i in range(all_maximum_idx_array.shape[0]):
        c_candicate = np.array((theta[int(all_maximum_idx_array[i, 0])], tau[int(all_maximum_idx_array[i, 1])]))
        candicate_aoa_tof = np.vstack((candicate_aoa_tof, c_candicate))
    candicate_aoa_tof = candicate_aoa_tof[1:, ]

    raw_package_results = pd.DataFrame(candicate_aoa_tof, columns=['aoa', 'tof'])

    pp.scatter(raw_package_results['aoa'], raw_package_results['tof'], c="g", alpha=0.5, marker=r'$\clubsuit$',
               label="peak values")
    pp.ylim(ymax=raw_package_results['tof'].max(), ymin=0)
    # pp.ylim()
    pp.xlabel("aoa")
    pp.ylabel("tof")
    pp.legend(loc=2)
    pp.show()

    disMat = sch.distance.pdist(candicate_aoa_tof, 'euclidean')
    Z = sch.linkage(disMat, method='ward')
    # P=sch.dendrogram(Z)

    raw_package_results['cluster'] = sch.fcluster(Z, t=1.115, criterion='inconsistent', depth=2)
    clusters = sorted(raw_package_results['cluster'].unique())

    data_likelihood = pd.DataFrame(columns=['cluster', 'cnt', 'aoa_mean', 'tof_mean', 'aoa_variance', 'tof_variance'])
    for i in clusters:
        data_likelihood.loc[i - 1, 'cluster'] = i
        data_likelihood.loc[i - 1, 'cnt'] = sum(raw_package_results['cluster'] == i)
        data_likelihood.loc[i - 1, 'aoa_mean'] = (
        raw_package_results['aoa'][raw_package_results['cluster'] == i]).mean()
        data_likelihood.loc[i - 1, 'tof_mean'] = (
        raw_package_results['tof'][raw_package_results['cluster'] == i]).mean()
        data_likelihood.loc[i - 1, 'aoa_variance'] = (
        raw_package_results['aoa'][raw_package_results['cluster'] == i]).var()
        data_likelihood.loc[i - 1, 'tof_variance'] = (
        raw_package_results['tof'][raw_package_results['cluster'] == i]).var()

    weight_num_cluster_points = 0.01
    weight_aoa_variance = -0.004
    weight_tof_variance = -0.0016
    weight_tof_mean = -0.0000
    constant_offset = -1

    data_likelihood['likelihood'] = (weight_num_cluster_points * data_likelihood['cnt'] \
                                     + weight_aoa_variance * data_likelihood['aoa_variance'] \
                                     + weight_tof_variance * data_likelihood['tof_variance'] \
                                     + weight_tof_mean * data_likelihood['tof_mean']).apply(lambda x: math.exp(x))

    data_likelihood['likelihood'].fillna(0, inplace=True)
    # cheating
    data_likelihood['likelihood'][data_likelihood['aoa_mean'] == -90] = 0

    print (data_likelihood)