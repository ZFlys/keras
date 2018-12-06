# coding=utf-8
import os
import numpy as np
import pandas as pd
import pywt


dataset_path = "./Datasets/"
time = np.linspace(0, 163.8, 4096)

xi = pd.read_csv(dataset_path + '0.csv', header=None)
xi.drop(0, axis=0, inplace=True)
xi.drop(0, axis=1, inplace=True)

# print(np.array(list(zip(time, xi.iloc[:, 0].values))).shape)
print(list(zip(time, xi.iloc[:, 0].values)))
wpt = pywt.WaveletPacket2D(list(zip(time, xi.iloc[:, 0].values)), 'db1', level=3)

# print(wpt)
print('levels:', len(wpt)-1)
print(wpt)
print(pywt.waverec2(wpt, 'db1').shape)
