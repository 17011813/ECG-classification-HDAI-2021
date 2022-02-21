import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from dataset import ECGDataset
from resnet import resnet34
from utils import cal_f1s, cal_aucs, split_data
import glob
import pandas as pd

"""def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data', help='Directory for data dir')
    parser.add_argument('--leads', type=str, default='all', help='ECG leads to use')
    parser.add_argument('--seed', type=int, default=42, help='Seed to split data')
    parser.add_argument('--num-classes', type=int, default=int, help='Num of diagnostic classes')
    parser.add_argument('--lr', '--learning-rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4, help='Num of workers to load data')
    parser.add_argument('--phase', type=str, default='train', help='Phase: train or test')
    parser.add_argument('--epochs', type=int, default=40, help='Training epochs')
    parser.add_argument('--resume', default=False, action='store_true', help='Resume')
    parser.add_argument('--use-gpu', default=True, action='store_true', help='Use GPU')
    parser.add_argument('--model-path', type=str, default='', help='Path to saved model')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.best_metric = 0
    data_dir = os.path.normpath(args.data_dir)
    database = os.path.basename(data_dir)

    if not args.model_path:
        args.model_path = f'models/resnet34_{database}_{args.leads}_{args.seed}.pth'

    if args.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = 'cpu'

    if args.leads == 'all':
        leads = 'all'
        nleads = 12
    else:
        leads = args.leads.split(',')
        nleads = len(leads)

    label_csv = os.path.join(data_dir, './CPSC/labels.csv')

    train_folds, val_folds, test_folds = split_data(seed=args.seed)
    train_dataset = ECGDataset('train', data_dir, label_csv, train_folds, leads)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True)
    val_dataset = ECGDataset('val', data_dir, label_csv, val_folds, leads)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)
    test_dataset = ECGDataset('test', data_dir, label_csv, test_folds, leads)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                             pin_memory=True)
    net = resnet34(input_channels=nleads).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.1)

    criterion = nn.BCEWithLogitsLoss()

    print(train_dataset)"""

"""import os
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
        self.leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        if leads == 'all':
            self.use_leads = np.where(np.in1d(self.leads, self.leads))[0]
        else:
            self.use_leads = np.where(np.in1d(self.leads, leads))[0]
        self.nleads = len(self.use_leads)
        self.classes = ['N', 'ST1', 'AF1', 'AF2', 'PAC', 'EAR', 'ST2', 'PVC', 'IR', 'VT', '1', '2_1', '2_2', '3', 'SB', 'JR']
        self.n_classes = len(self.classes)
        self.data_dict = {}
        self.label_dict = {}

    def __getitem__(self, index: int):
        row = self.labels.iloc[index]     # index를 전체 len 길이만큼 하나씩 다 가져오나보다
        patient_id = row['patient_id']
        # ecg_data, _ = wfdb.rdsamp(os.path.join(self.data_dir, patient_id))   # 여기서 numpy.ndarray로 가져옴

        name = os.path.join(self.data_dir, patient_id)
        #names = glob.glob(name + "*.csv")
        names = glob.glob("5_0_000001_ecg.csv")
        ecg_data = np.loadtxt(os.path.join(self.data_dir, patient_id+".csv"),  delimiter=",", dtype=np.float32)
        print("----------------------------",ecg_data)

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




if __name__ == "__main__":
    cal1 = ECGDataset('train', './data/CPSC/train', os.path.join('./data/', 'csv_labels.csv'), all)
    print(cal1.__getitem__(0))"""

"""import pandas as pd
data = pd.read_csv('./data/CPSC/train/5_0_000001_ecg.csv')
#print(data)
# 첫번째 행 제거해서 feature name 제거 : skiprows
# 8 leads로 맞추기 위해 중간 4개는 다 제거 : usecols
ecg_data = np.loadtxt(os.path.join('./data/CPSC/train/', '5_0_000001_ecg.csv'),
                    skiprows=1, usecols = (0,1,6,7,8,9,10,11), delimiter=",", dtype=np.float32)

print(ecg_data)"""

train_dir = './data/CPSC/train/'

for files in glob.glob(train_dir+ "*.csv"):
    x = pd.read_csv(files, low_memory=False)
    #X_train = torch.stack(torch.Tensor(x.values))
    X_train = torch.Tensor(x.values)
    #
    print(X_train)
    break
#X_train = torch.stack([torch.Tensor(i) for i in train_dir])

#print(X_train.shape)
