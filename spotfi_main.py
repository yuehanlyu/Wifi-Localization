# coding: utf-8
from __future__ import print_function, absolute_import, division
import pandas as pd
import scipy.cluster.hierarchy as sch
import algo_load_csi_data
import numpy as np
import algo_spotfi
import algo_rssi
import math
from tqdm import tqdm
import warnings
import matplotlib.pyplot as pp
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore", category=DeprecationWarning)

if __name__ == '__main__':
    antenna_distance = 0.15
    frequency = 2.417 * pow(10, 9)
    sub_freq_delta = 3125

    ap_coordinates = np.array([[-2.33,0],[0,0],[2.33,0]])
    theta_range = np.linspace(-90,90,91)
    tau_range = np.linspace(0,3000 * pow(10,-9),61)

    for test_cnt in [3]: # turn of test

        result_aoas = np.array([0.0])
        result_rssi = np.array([0.0])
        result_likelihood = np.array([0.0])

        for i in [1,2,3]:  #1: left 2:mid 3:right

            file_data = algo_load_csi_data.read_bf_file('./sample_data/csi_1123_' + str(i) + '_' + str(test_cnt) + '.dat')
            rssi_ = algo_rssi.rssi(file_data).mean()

            csi_entry = file_data.loc[0]
            csi = algo_load_csi_data.get_scale_csi(csi_entry)
            csi_matrix = csi[0, :, :]
            C, tau_offset = algo_spotfi.spotfi_algorithm_1_package_one(csi_matrix)

            all_maximum_idx_array = np.zeros(2)
            for p_i in tqdm(range(10)):
                csi_entry = file_data.loc[p_i]
                csi = algo_load_csi_data.get_scale_csi(csi_entry)
                # run algorithm 1 for the first package
                csi_matrix = csi[0, :, :]
                csi_matrix_clean = algo_spotfi.spotfi_algorithm_1(csi_matrix, C)
                # return the smoothed_csi matrix
                smoothed_csi = algo_spotfi.smooth_csi(csi_matrix_clean)
                maximum_idx_array = algo_spotfi.aoa_tof_music(smoothed_csi, antenna_distance, frequency,
                                                              sub_freq_delta, theta_range, tau_range)
                all_maximum_idx_array = np.vstack((all_maximum_idx_array, maximum_idx_array))

            candicate_aoa_tof = np.zeros(2)
            for c_i in range(all_maximum_idx_array.shape[0]):
                c_candicate = np.array((theta_range[int(all_maximum_idx_array[c_i, 0])], tau_range[int(all_maximum_idx_array[c_i, 1])]))
                candicate_aoa_tof = np.vstack((candicate_aoa_tof, c_candicate))
            candicate_aoa_tof = candicate_aoa_tof[1:, ]

            raw_package_results = pd.DataFrame(candicate_aoa_tof, columns=['aoa', 'tof'])

            disMat = sch.distance.pdist(candicate_aoa_tof, 'euclidean')
            Z = sch.linkage(disMat, method='ward')
            # P=sch.dendrogram(Z)

            raw_package_results['cluster'] = sch.fcluster(Z, t=1.115, criterion='inconsistent', depth=2)
            clusters = sorted(raw_package_results['cluster'].unique())

            data_likelihood = pd.DataFrame(
                columns=['cluster', 'cnt', 'aoa_mean', 'tof_mean', 'aoa_variance', 'tof_variance'])
            for clu_i in clusters:
                data_likelihood.loc[clu_i - 1, 'cluster'] = clu_i
                data_likelihood.loc[clu_i - 1, 'cnt'] = sum(raw_package_results['cluster'] == clu_i)
                data_likelihood.loc[clu_i - 1, 'aoa_mean'] = (
                raw_package_results['aoa'][raw_package_results['cluster'] == clu_i]).mean()
                data_likelihood.loc[clu_i - 1, 'tof_mean'] = (
                raw_package_results['tof'][raw_package_results['cluster'] == clu_i]).mean()
                data_likelihood.loc[clu_i - 1, 'aoa_variance'] = (
                raw_package_results['aoa'][raw_package_results['cluster'] == clu_i]).var()
                data_likelihood.loc[clu_i - 1, 'tof_variance'] = (
                raw_package_results['tof'][raw_package_results['cluster'] == clu_i]).var()

            weight_num_cluster_points = 0.0001
            weight_aoa_variance = -0.0004
            weight_tof_variance = -0.0016
            weight_tof_mean = -0.0000
            constant_offset = -1

            data_likelihood['likelihood'] = (
            weight_num_cluster_points * data_likelihood['cnt'] + weight_aoa_variance * data_likelihood[
                'aoa_variance'] + weight_tof_variance * data_likelihood['tof_variance'] + weight_tof_mean * data_likelihood[
                'tof_mean']).apply(lambda x: math.exp(x))

            data_likelihood['likelihood'].fillna(0, inplace=True)
            # cheating
            data_likelihood['likelihood'][data_likelihood['aoa_mean'] == -90] = 0
            data_likelihood['likelihood'][data_likelihood['aoa_mean'] == 90] = 0

            result_aoa_1 = data_likelihood['aoa_mean'][data_likelihood['likelihood'] == max(data_likelihood['likelihood'])]
            result_likelihood_1 = data_likelihood['likelihood'][data_likelihood['likelihood'] == max(data_likelihood['likelihood'])]

            result_aoas = np.concatenate((result_aoas, np.array([result_aoa_1.iloc[0]])))
            result_rssi = np.concatenate((result_rssi, np.array([rssi_])))
            result_likelihood = np.concatenate((result_likelihood, np.array([result_likelihood_1.iloc[0]])))

        result_aoas = result_aoas[1:]
        result_rssi = result_rssi[1:]
        result_likelihood = result_likelihood[1:]

        theta_left = result_aoas[0]
        theta_mid = result_aoas[1]
        theta_right = result_aoas[2]
        algo_spotfi.csi_plot(theta_left, theta_mid, theta_right)

        result_aoas = result_aoas[::-1]

        # 一个naive的遍历优化
        num_AP = 3
        min_obj = float(np.Inf)
        final_coor_idx = 0
        candidate_coordinates = np.array([[2.0, 2.0], [1.0, 1.0], [-2.0, 2.0]])
        ap_coordinates = np.array([[-2.33, 0], [0, 0], [2.33, 0]])
        for k in range(candidate_coordinates.shape[0]):  # traverse all candidate coordinates
            obj = 0.0
            for i in range(num_AP):  # sum up errors over all APs
                d = algo_spotfi.ecldDist(ap_coordinates[i],candidate_coordinates[k])
                theta_ = algo_spotfi.coord2aoa(ap_coordinates[i], candidate_coordinates[k])
                obj = obj + result_likelihood[i] * (
                (result_rssi[i] - algo_spotfi.dist2rssi(d, -30, 2.1)) ** 2 + (result_aoas[i] - theta_) ** 2)
            if obj < min_obj:
                final_coor_idx = k
                min_obj = obj
        print(candidate_coordinates[final_coor_idx, :])