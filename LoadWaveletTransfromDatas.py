# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import pywt

dataset_path = "./Datasets/"


# 传入data必须为一维数组
def wpt1D(data, wavelet='db1', mode='symmetric', maxlevel=4):

    # 一维小波包分解
    wp = pywt.WaveletPacket(data=data, wavelet=wavelet, mode=mode, maxlevel=maxlevel)
    
    # 列出所有树节点路径
    paths = [node.path for node in wp.get_level(4)]
    
    # 合并所有四级子节点
    df_wpt = pd.DataFrame()
    for a in paths:
        df_wpt[a] = wp[a].data
        
    return df_wpt


# # 导入示例数据测试(DF结构，索引为Time，列为幅值)
# df = pd.read_pickle('ut_example.pkl')
# data1 = df[70:90.44]['example'].values
# data2 = df[140:160.44]['example'].values
#
# # 截取两批512个点，合并成32X32二维数组
# ut_wpt1 = wpt1D(data1).values
# ut_wpt2 = wpt1D(data2).values
# ut_wpt12 = pd.concat([wpt1D(data1), wpt1D(data2)], axis=1).values
# ut_wpt_all = wpt1D(df['example'].values).values


def Load_Ut_Datas():
    x = []
    y = []
    for file in os.listdir(dataset_path):
        yi = os.path.splitext(file)[0]
        xi = pd.read_csv(dataset_path+file, header=None)
        xi.drop(0, axis=0, inplace=True)
        xi.drop(0, axis=1, inplace=True)
        for i in range(0, 100):
            y.append(yi)
            x.append(wpt1D(xi.iloc[:, i].values[1750:2261]).values)  # [1750:2261]
    print(np.array(x).shape)
    print(np.array(y).shape)
    return np.array(x), np.array(y)


if __name__ == '__main__':
    Load_Ut_Datas()
