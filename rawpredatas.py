import os
import numpy as np
import pandas as pd


dataset_path = "./DatasetPre/"
datasetPre_path = "./DatasetPreCSV/"


def Load_Pre_Datas():
    x = []
    for file in os.listdir(dataset_path):
        xi = pd.read_excel(dataset_path+file, sheet_name=[0, 1, 2])
        for i in range(0, 3):
            if file == 'all.xlsx':
                for j in range(1, 26):
                    x.append(xi[i].iloc[:, j].values[:1500])
            else:
                for j in range(1, 11):
                    x.append(xi[i].iloc[:, j].values[:1500])
    x = np.array(x)
    print(x.shape)
    x_pd = pd.DataFrame(x)
    x_pd.to_csv(datasetPre_path+"pre1500.csv")

    return x


if __name__ == '__main__':
    Load_Pre_Datas()
