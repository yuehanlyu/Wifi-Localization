
# coding: utf-8
# Authorized to Zhaoxin03@ppdai.com

from __future__ import print_function, absolute_import, division
import pandas as pd
import numpy as np
# import scipy.io as sio
import math
import os



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
                j +=1

            csi[:,0,k] = a[:2]
            csi[:,1,k] = a[4:]
            csi[:,2,k] = a[2:4]
            k += 1
    df_data.loc[i,'csi'] = csi
    return df_data



def read_bf_file(filename):
    cur = 0
    count = 0
    broken_perm = 0
    data = pd.DataFrame(columns=['cell','timestamp_low','bfee_count','Nrx','Ntx','rssi_a','rssi_b','rssi_c','noise','agc','perm','rate','csi'])
    triangle = [1,3,6]

    f = open(filename,'rb')
    data_len = os.path.getsize(filename)
    #print(data_len)
    while cur < (data_len - 3):
        field_len = int(f.read(2).encode('hex'),16)
        code = int(f.read(1).encode('hex'),16)
        cur = cur + 3


        if code == 187:
            bytes = np.fromfile(f,np.uint8,count = field_len - 1)
            cur = cur + field_len - 1
            if len(bytes) != field_len - 1:
                f.close()
        else:
            f.seek(field_len - 1,1)
            cur = cur + field_len - 1

        if code == 187:
            data = read_bfee(inBytes=bytes,i=count,df_data = data)
            perm = data.loc[count,'perm']
            Nrx = data.loc[count,'Nrx']

            if sum(perm) == triangle[Nrx-1]:
                count = count + 1
    f.close()
    return data



def dbinv(x):
    ret = 10**(x/10)
    return ret



def get_total_rss(dic):
    rssi_mag = 0
    if dic['rssi_a'] != 0:
        rssi_mag = rssi_mag + dbinv(dic['rssi_a'])
    if dic['rssi_b'] != 0:
        rssi_mag = rssi_mag + dbinv(dic['rssi_b'])
    if dic['rssi_c'] != 0:
        rssi_mag = rssi_mag + dbinv(dic['rssi_c'])


    ret = 10*math.log10(rssi_mag) - 44 - dic['agc']

    return ret



def get_scale_csi(dic):
    csi = dic['csi']
    csi_conj = csi.conjugate()
    csi_sq = np.multiply(csi,csi_conj).real
    csi_pwr = sum(sum(sum(csi_sq[:])))
    rssi_pwr = dbinv(get_total_rss(dic))
    scale = rssi_pwr / (csi_pwr / 30)

    if dic['noise'] == -127:
        noise_db = -92
    else:
        noise_db = dic['noise']

    thermal_noise_pwr = dbinv(noise_db)
    quant_error_pwr = scale * (dic['Nrx'] * dic['Ntx'])
    total_noise_pwr = thermal_noise_pwr + quant_error_pwr;

    ret = csi * math.sqrt(scale / total_noise_pwr)

    if dic['Ntx'] == 2:
        ret = ret * math.sqrt(2)
    elif dic['Ntx'] == 3:
        ret = ret * math.sqrt(dbinv(4.5))

    return ret
