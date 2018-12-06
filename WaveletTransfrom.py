# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 15:02:19 2018

@author: SCUTYJ
"""

import numpy as np
import pandas as pd
import pywt


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


# 导入示例数据测试(DF结构，索引为Time，列为幅值)
df = pd.read_pickle('ut_example.pkl')
data1 = df[70:90.44]['example'].values
data2 = df[140:160.44]['example'].values

# 截取两批512个点，合并成32X32二维数组
ut_wpt1 = wpt1D(data1).values
ut_wpt2 = wpt1D(data2).values
ut_wpt12 = pd.concat([wpt1D(data1), wpt1D(data2)], axis=1).values
ut_wpt_all = wpt1D(df['example'].values).values
