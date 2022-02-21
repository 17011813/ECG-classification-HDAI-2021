import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import wfdb
import glob
def scaling(X, sigma=0.1):
    scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(1, X.shape[1]))
    myNoise = np.matmul(np.ones((X.shape[0], 1)), scalingFactor)
    return X * myNoise

def shift(sig, interval=20):
    for col in range(sig.shape[1]):
        offset = np.random.choice(range(-interval, interval))
        sig[:, col] += offset / 1000 
    return sig

def transform(sig, train=False):
    if train:
        if np.random.randn() > 0.5: sig = scaling(sig)
        if np.random.randn() > 0.5: sig = shift(sig)
    return sig

class ECGDataset(Dataset):
    def __init__(self, phase, data_dir, label_csv, leads):
        super(ECGDataset, self).__init__()
        self.phase = phase
        df = pd.read_csv(label_csv)
        #df = df[df['fold'].isin(folds)]
        self.data_dir = data_dir
        self.labels = df
        self.leads = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'] # 12 leads 에서 8 leads로 수정
        if leads == 'all':
            self.use_leads = np.where(np.in1d(self.leads, self.leads))[0]
        else:
            self.use_leads = np.where(np.in1d(self.leads, leads))[0]
        self.nleads = len(self.use_leads)
        # self.classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']
        self.classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        self.n_classes = len(self.classes)
        self.data_dict = {}
        self.label_dict = {}

    def __getitem__(self, index: int):
        row = self.labels.iloc[index]     # index를 전체 len 길이만큼 하나씩 다 가져오나보다
        patient_id = row['patient_id']
        # ecg_data, _ = wfdb.rdsamp(os.path.join(self.data_dir, patient_id))   # 여기서 numpy.ndarray로 가져옴

        ecg_data = np.loadtxt(os.path.join(self.data_dir, patient_id+".csv"),
                    skiprows=1, delimiter=",",usecols = (0,1,6,7,8,9,10,11), dtype=np.float32)


        ecg_data = transform(ecg_data, self.phase == 'train')
        nsteps, _ = ecg_data.shape
        ecg_data = ecg_data[-5000:, self.use_leads]
        result = np.zeros((5000, self.nleads)) # 10 s, 500 Hz
        result[-nsteps:, :] = ecg_data
        if self.label_dict.get(patient_id):
            labels = self.label_dict.get(patient_id)
        else:
            labels = row[self.classes].to_numpy(dtype=np.float32)
            self.label_dict[patient_id] = labels
        return torch.from_numpy(result.transpose()).float(), torch.from_numpy(labels).float()

    def __len__(self):
        return len(self.labels)


