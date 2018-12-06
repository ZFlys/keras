import os
import numpy as np
import pandas as pd


dataset_path = "./kingfadata/"
kingfa_data_path = "./kingfadatasets/"

def Load_Ut_Datas():
    x = []
    y = []
    for file in os.listdir(dataset_path):
        yi = os.path.splitext(file)[0]
        if 351 <= int(yi) <= 360:
            yi = 0
        elif 361 <= int(yi) <= 370:
            yi = 1
        else:
            yi = 2
        xi = pd.read_csv(dataset_path+file, header=None)
        xi.drop(0, axis=0, inplace=True)
        xi.drop(0, axis=1, inplace=True)
        for i in range(0, 100):
            y.append(yi)
            x.append(xi.iloc[:, i].values)
    x = np.array(x)
    y = np.array(y)
    print(x.shape)
    print(y.shape)
    x_pd = pd.DataFrame(x)
    y_pd = pd.DataFrame(y)
    x_pd.to_csv(kingfa_data_path + "data.csv")
    y_pd.to_csv(kingfa_data_path + "target.csv")
    return x, y


if __name__ == '__main__':
    Load_Ut_Datas()
