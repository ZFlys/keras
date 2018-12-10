# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt
import os
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

#传入data必须为一维数组
def wpt1D(data,wavelet='db1',mode='symmetric',maxlevel=2):
    wp = pywt.WaveletPacket(data=data, wavelet=wavelet,mode=mode,maxlevel = maxlevel)     
    paths = [node.path for node in wp.get_level(maxlevel)]    
    num = int(data.shape[0]/(2**maxlevel))
    df_wpt = pd.DataFrame(index=range(num))
    for a in paths:
        df_wpt[a] = wp[a].data        
    return df_wpt

def plot_wpt(df1, df2, df3, maxlevel, title):
    
    df_wpt1 = wpt1D(df1[70:90.44][0],'db1','symmetric',maxlevel)
    df_wpt2 = wpt1D(df2[70:90.44][0],'db1','symmetric',maxlevel)
    df_wpt3 = wpt1D(df3[70:90.44][0],'db1','symmetric',maxlevel)    
    
    plt.figure()
    for i in  range(1,2**maxlevel+1):
        plt.subplot(4,4,i)
        plt.plot(df_wpt1.index, df_wpt1.iloc[:,i-1])
        plt.plot(df_wpt2.index, df_wpt2.iloc[:,i-1])  
        plt.plot(df_wpt3.index, df_wpt3.iloc[:,i-1])
        #plt.ylim((-35,35))
        plt.title('{}'.format(df_wpt1.columns[i-1]))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, 
    wspace=0.6, hspace=0.8)
    plt.suptitle(title)
    plt.show()
    
df0 = pd.read_csv('WPT/HDPE_WOOD_0.csv',engine='python',header=None)
df1 = pd.read_csv('WPT/HDPE_WOOD_1.csv',engine='python',header=None)
df2 = pd.read_csv('WPT/HDPE_WOOD_2.csv',engine='python',header=None)
df00 = pd.read_csv('WPT/HDPE_WOOD_00.csv',engine='python',header=None)
df00.index = df0.index = df1.index = df2.index = np.linspace(0,163.8,4096)

#原始信号比较
HDPE = pd.concat([df0,df1,df2],axis=1)[70:90.44]
#HDPE.columns = ['HDPE基体','完整配方','白杨木','杂木粉']
HDPE.columns = ['完整配方','白杨木','杂木粉']
HDPE.plot()
plt.title('原始信号')
plt.show()
plot_wpt(df0,df1,df2,4,'原始信号小波包分解')

#信号相减后小波包分解
# =============================================================================
# df_x_0 = pd.concat([df0-df00,df1-df00,df2-df00],axis=1)
# df_x_0.columns = ['完整配方','白杨木','杂木粉']
# df_x_0.plot()
# plt.title('信号相减')
# plt.show()
# =============================================================================
plot_wpt(df0-df00,df1-df00,df2-df00,4,'信号相减后小波包分解')


#小波包分解后相减
df_wpt00 = wpt1D(df00[70:90.44][0],'db3','symmetric',4)
df_wpt0 = wpt1D(df0[70:90.44][0],'db1','symmetric',4)
df_wpt1 = wpt1D(df1[70:90.44][0],'db1','symmetric',4)
df_wpt2 = wpt1D(df2[70:90.44][0],'db1','symmetric',4)   
plt.figure()
for i in  range(1,17):
    plt.subplot(4,4,i)
    plt.plot(df_wpt0.index, (df_wpt0-df_wpt00).iloc[:,i-1])
    plt.plot(df_wpt1.index, (df_wpt1-df_wpt00).iloc[:,i-1])  
    plt.plot(df_wpt2.index, (df_wpt2-df_wpt00).iloc[:,i-1])
    plt.title('{}'.format(df_wpt1.columns[i-1]))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, 
wspace=0.6, hspace=0.8)
plt.suptitle('小波包分解后相减')
plt.show()





