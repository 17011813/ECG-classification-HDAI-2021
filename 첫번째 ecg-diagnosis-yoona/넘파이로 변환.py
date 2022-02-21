import os
import pandas as pd
import numpy as np

"""path = "C:/Users/yoona/Desktop/ecg-diagnosis-yoona/data/CPSC/val"
file_list = os.listdir(path)
totaldata = np.empty(shape = (0,5000,8))
for file in file_list:
    data = pd.read_csv(path+"/"+file)
    data = data.loc[:, ['rhythm_I','rhythm_II','rhythm_V1','rhythm_V2','rhythm_V3','rhythm_V4','rhythm_V5','rhythm_V6']]
    data = data.to_numpy()
    data = data.reshape(1,5000,8)
    totaldata = np.concatenate((totaldata,data),axis=0)

np.save("./train_normal.npy",totaldata)"""

"""for file in file_list:
    label = pd.read_csv("C:/Users/yoona/Desktop/ecg-diagnosis-yoona/data/val_labels_org.csv").loc[[file[:-4]]][1]
    #print(label)"""
    #print(file)


#label = pd.read_csv("C:/Users/yoona/Desktop/ecg-diagnosis-yoona/data/val_labels_org.csv").loc[['8_2_009016_ecg']][1]
#print(label)

#train_org = np.load('./data/CPSC/train/train_arrhythmia_data_lbl_array_nonan.npy')
#train_org = np.concatenate((np.load('./data/CPSC/train/train_arrhythmia_data_lbl_array_nonan.npy'),
                      #np.load('./data/CPSC/train/train_normal_data_lbl_array_nonan.npy')), axis=0)
test_org = np.concatenate((np.load('./data/CPSC/val/validation_arrhythmia_data_lbl_array_nonan.npy'),np.load('./data/CPSC/val/validation_normal_data_lbl_array_nonan.npy')), axis=0)
test_X = test_org[:, :-1]

test_X = test_X.reshape(-1,5000,8)

print(test_X.shape)