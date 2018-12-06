#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pywt
import pywt.data

dataset_path = "./Datasets/"
time = np.linspace(0, 163.8, 4096)

xi = pd.read_csv(dataset_path + '0.csv', header=None)
xi.drop(0, axis=0, inplace=True)
xi.drop(0, axis=1, inplace=True)

wpt = np.array(xi.iloc[:, 0].values)


def plot_signal_decomp(data, w, title):
    """Decompose and plot a signal S.
    S = An + Dn + Dn-1 + ... + D1
    """

    a = data
    ca = []     # 近似分量
    cd = []     # 细节分量
    for i in range(5):
        (a, d) = pywt.dwt(a, 'db1', mode='sym')     # 进行5阶离散小波变换
        ca.append(a)
        cd.append(d)

    rec_a = []
    rec_d = []

    for i, coeff in enumerate(ca):
        coeff_list = [coeff, None] + [None] * i
        rec_a.append(pywt.waverec(coeff_list, w))       # 重构

    for i, coeff in enumerate(cd):
        coeff_list = [None, coeff] + [None] * i
        if i == 3:
            print(len(coeff))
            print(len(coeff_list))
        rec_d.append(pywt.waverec(coeff_list, w))

    fig = plt.figure()
    ax_main = fig.add_subplot(len(rec_a) + 1, 1, 1)
    ax_main.set_title(title)
    ax_main.plot(data)
    ax_main.set_xlim(0, len(data) - 1)

    for i, y in enumerate(rec_a):
        ax = fig.add_subplot(len(rec_a) + 1, 2, 3 + i * 2)
        ax.plot(y, 'r')
        ax.set_xlim(0, len(y) - 1)
        ax.set_ylabel("A%d" % (i + 1))

    for i, y in enumerate(rec_d):
        ax = fig.add_subplot(len(rec_d) + 1, 2, 4 + i * 2)
        ax.plot(y, 'g')
        ax.set_xlim(0, len(y) - 1)
        ax.set_ylabel("D%d" % (i + 1))


# plot_signal_decomp(data1, 'coif5', "DWT: Signal irregularity")
# plot_signal_decomp(data2, 'sym5',
# "DWT: Frequency and phase change - Symmlets5")
plot_signal_decomp(wpt, 'sym5', "DWT: Ecg sample - Symmlets5")


plt.show()
