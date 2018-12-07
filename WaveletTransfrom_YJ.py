# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 15:02:19 2018

@author: SCUTYJ
"""

import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt
import os
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

#传入data必须为一维数组
def wpt1D(data,wavelet='db1',mode='symmetric',maxlevel=2):

    #一维小波包分解
    wp = pywt.WaveletPacket(data=data, wavelet=wavelet,mode=mode,maxlevel = maxlevel)  
    
    #列出所有树节点路径
    paths = [node.path for node in wp.get_level(maxlevel)]
    
    #合并所有四级子节点
    num = int(data.shape[0]/(2**maxlevel))
    df_wpt = pd.DataFrame(index=range(num))
    for a in paths:
        df_wpt[a] = wp[a].data
        
    return df_wpt


# df_yes = pd.read_table(r'D:\Jupyter Notebook\Research_Ultrasound\质量检测\yes_no\yes.zwav',
#                        engine='python',sep='\t').set_index('TIME(us)')
# df_no = pd.read_table(r'D:\Jupyter Notebook\Research_Ultrasound\质量检测\yes_no\no.zwav',
#                     engine='python',sep='\t').set_index('TIME(us)')
# df = pd.read_pickle('ut_example.pkl')
# df.columns = ['DATA']
    
df1 = pd.read_csv('WPT/HDPE_WOOD_0.csv',engine='python',header=None)
df2 = pd.read_csv('WPT/HDPE_WOOD_1.csv',engine='python',header=None)
df3 = pd.read_csv('WPT/HDPE_WOOD_2.csv',engine='python',header=None)
df1.index = df2.index = df3.index = np.linspace(0,163.8,4096)


def plot_wpt(df1, df2, df3, maxlevel, title):
    
    df_wpt1 = wpt1D(df1[70:90.44][0],'db1','symmetric',maxlevel)
    df_wpt2 = wpt1D(df2[70:90.44][0],'db1','symmetric',maxlevel)
    df_wpt3 = wpt1D(df3[70:90.44][0],'db1','symmetric',maxlevel)
    
    if maxlevel == 1:
        plt.figure()
        for i in  range(1,2**maxlevel+1):
            plt.subplot(2,1,i)
            plt.plot(df_wpt1.index, df_wpt1.iloc[:,i-1])
            plt.plot(df_wpt2.index, df_wpt2.iloc[:,i-1])  
            plt.plot(df_wpt3.index, df_wpt3.iloc[:,i-1])            #plt.ylim((-35,35))
            plt.title('{}'.format(df_wpt1.columns[i-1]))
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, 
        wspace=0.2, hspace=0.4)
        plt.suptitle(title)
        plt.show()
    
    if maxlevel == 2:
        plt.figure()
        for i in  range(1,2**maxlevel+1):
            plt.subplot(2,2,i)
            plt.plot(df_wpt1.index, df_wpt1.iloc[:,i-1])
            plt.plot(df_wpt2.index, df_wpt2.iloc[:,i-1])  
            plt.plot(df_wpt3.index, df_wpt3.iloc[:,i-1])            #plt.ylim((-35,35))
            plt.title('{}'.format(df_wpt1.columns[i-1]))
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, 
        wspace=0.2, hspace=0.4)
        plt.suptitle(title)
        plt.show()
    
    if maxlevel == 3:        
        plt.figure()
        for i in  range(1,2**maxlevel+1):    
            plt.subplot(4,2,i)
            plt.plot(df_wpt1.index, df_wpt1.iloc[:,i-1])
            plt.plot(df_wpt2.index, df_wpt2.iloc[:,i-1])  
            plt.plot(df_wpt3.index, df_wpt3.iloc[:,i-1])            #plt.ylim((-30,30))
            plt.title('{}'.format(df_wpt1.columns[i-1]))
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, 
        wspace=0.4, hspace=0.6)
        plt.suptitle(title)
        plt.show()
    
    if maxlevel == 4:
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
    
    
HDPE_WOOD = pd.concat([df1,df2,df3],axis=1)[70:90.44]
HDPE_WOOD.columns = [0,1,2]
HDPE_WOOD.plot()
plot_wpt(df1,df2,df3,4,'合格与不合格')

# =============================================================================
# #导入示例数据测试(DF结构，索引为Time，列为幅值)
# df = pd.read_pickle('ut_example.pkl')
# data1 = df[70:90.44]['example'].values
# data2 = df[140:160.44]['example'].values
# 
# #截取两批512个点，合并成32X32二维数组
# ut_wpt = pd.concat([wpt1D(data1),wpt1D(data2)],axis=1).values
# =============================================================================
