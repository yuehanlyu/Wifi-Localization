
# coding: utf-8
# Authorized to zhaoxin03@ppdai.com

from __future__ import print_function, absolute_import, division
import pandas as pd
import numpy as np
# import scipy.io as sio
import math
import os
import matplotlib


def read_bfee(inBytes,i,df_data):
    df_data.loc[i,'cell'] = i + 1 # 第几个包
    # NIC网卡1MHz时钟
    df_data.loc[i,'timestamp_low'] = inBytes[0] + (inBytes[1] << 8) + (inBytes[2] << 16) + (inBytes[3] << 24)
    # 驱动记录并发送到用户控件的波束测量值的总数
    df_data.loc[i,'bfee_count'] = inBytes[4] + (inBytes[5] << 8)

    df_data.loc[i,'Nrx'] = inBytes[8] # 接收端天线数量
    df_data.loc[i,'Ntx'] = inBytes[9] # 发送端天线数量
    # 接收端NIC测量出的RSSI值
    df_data.loc[i,'rssi_a'] = inBytes[10]
    df_data.loc[i,'rssi_b'] = inBytes[11]
    df_data.loc[i,'rssi_c'] = inBytes[12]

    df_data.loc[i,'noise'] = inBytes[13].astype(np.int8)
    df_data.loc[i,'agc'] = inBytes[14]
    df_data.loc[i,'rate'] = inBytes[18] + (inBytes[19] << 8) # 发包频率

    # perm 展示NIC如何将3个接收天线的信号排列在3个RF链上
    # [1,2,3] 表示天线A被发送到RF链A，天线B--> RF链B，天线C--> RF链C
    # [1,3,2] 表示天线A被发送到RF链A，天线B--> RF链C，天线C--> RF链B
    perm = [1,1,1]
    antenna_sel = inBytes[15]
    perm[0] = ((antenna_sel) & 0x3) + 1
    perm[1] = ((antenna_sel >> 2) & 0x3) + 1
    perm[2] = ((antenna_sel >> 4) & 0x3) + 1
    df_data.loc[i,'perm'] = perm

    # csi
    leng = inBytes[16] + (inBytes[17] << 8)
    calc_len = int((30 * (inBytes[8] * inBytes[9] * 8 * 2 + 3) + 7) / 8)
    index = 0
    payload = inBytes[20:]
    csi = np.empty([2,3,30],dtype = complex)
    if leng == calc_len:
        for k in range(30):
            index += 3
            remainder = index % 8
            a = []
            for j in range(inBytes[8]*inBytes[9]):
                tmp_r = ((payload[int(index/8)] >> remainder) | (payload[int(index/8+1)] << (8-remainder))).astype(np.int8)
                tmp_i = ((payload[int(index/8+1)] >> remainder) | (payload[int(index/8+2)] << (8-remainder))).astype(np.int8)
                a.append(complex(tmp_r,tmp_i))
                index += 16
                j +
