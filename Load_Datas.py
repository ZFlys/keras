import os
import numpy as np
import pandas as pd


dataset_path = "./Datasets/"


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
            x.append(xi.iloc[:, i].values)
    print(np.array(x).shape)
    print(np.array(y).shape)
    return np.array(x), np.array(y)


if __name__ == '__main__':
    Load_Ut_Datas()
