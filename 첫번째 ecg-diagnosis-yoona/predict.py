import argparse
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from resnet import resnet34
from dataset import ECGDataset
from utils import cal_scores, find_optimal_threshold, split_data
from sklearn.preprocessing import scale
from torch.utils.data import TensorDataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data', help='Directory to data dir')
    parser.add_argument('--leads', type=str, default='all', help='ECG leads to use')
    parser.add_argument('--seed', type=int, default=42, help='Seed to split data')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers to load data')
    parser.add_argument('--use-gpu', default=True, action='store_true', help='Use gpu')
    parser.add_argument('--model-path', type=str, default='models/resnet34_data_all_42.pth', help='Path to saved model')
    return parser.parse_args()


def get_thresholds(val_loader, net, device, threshold_path):
    print('Finding optimal thresholds...')
    if os.path.exists(threshold_path):
        return pickle.load(open(threshold_path, 'rb'))
    output_list, label_list = [], []
    for _, (data, label) in enumerate(tqdm(val_loader)):
        data, labels = data.to(device), label.to(device)
        output = net(data)
        output = torch.sigmoid(output)
        output_list.append(output.data.cpu().numpy())
        label_list.append(labels.data.cpu().numpy())
    y_trues = np.vstack(label_list)
    y_scores = np.vstack(output_list)
    thresholds = []
    for i in range(y_trues.shape[1]):
        y_true = y_trues[:, i]
        y_score = y_scores[:, i]
        threshold = find_optimal_threshold(y_true, y_score)
        thresholds.append(threshold)
    # pickle.dump(thresholds, open(threshold_path, 'wb'))
    return thresholds


def apply_thresholds(test_loader, net, device, thresholds):
    output_list, label_list = [], []
    for _, (data, label) in enumerate(tqdm(test_loader)):
        data, labels = data.type(torch.FloatTensor).to(device), labels.type(torch.FloatTensor).to(device)
        output = net(data).type(torch.FloatTensor).to(device)
        output = torch.sigmoid(output)
        output_list.append(output.data.cpu().numpy())
        label_list.append(labels.data.cpu().numpy())
    y_trues = np.vstack(label_list)
    y_scores = np.vstack(output_list)
    y_preds = []
    scores = [] 
    for i in range(len(thresholds)):
        y_true = y_trues[:, i]
        y_score = y_scores[:, i]
        y_pred = (y_score >= thresholds[i]).astype(int)
        scores.append(cal_scores(y_true, y_pred, y_score))
        y_preds.append(y_pred)
    y_preds = np.array(y_preds).transpose()
    scores = np.array(scores)
    print('Precisions:', scores[:, 0])
    print('Recalls:', scores[:, 1])
    print('F1s:', scores[:, 2])
    print('AUCs:', scores[:, 3])
    print('Accs:', scores[:, 4])
    print(np.mean(scores, axis=0))
    plot_cm(y_trues, y_preds)


def plot_cm(y_trues, y_preds, normalize=True, cmap=plt.cm.Blues):
    classes = ['SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'PVC', 'STD', 'STE']
    for i, label in enumerate(classes):
        y_true = y_trues[:, i]
        y_pred = y_preds[:, i]
        cm = confusion_matrix(y_true, y_pred)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=[0, 1], yticklabels=[0, 1],
           title=label,
           ylabel='True label',
           xlabel='Predicted label')
        plt.setp(ax.get_xticklabels(), ha="center")

        fmt = '.3f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        np.set_printoptions(precision=3)
        fig.tight_layout()
        plt.savefig(f'results/{label}.png')
        plt.close(fig)


if __name__ == "__main__":
    args = parse_args()
    data_dir = os.path.normpath(args.data_dir)
    database = os.path.basename(data_dir)
    if not args.model_path:
        args.model_path = f'models/resnet34_{database}_{args.leads}_{args.seed}.pth'
    args.threshold_path = f'models/{database}-threshold.pkl'
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = 'cpu'
    
    if args.leads == 'all':
        leads = 'all'
        nleads = 8
    else:
        leads = args.leads.split(',')
        nleads = len(leads)
    train_org = np.concatenate((np.load('./data/CPSC/train/train_arrhythmia_data_lbl_array_nonan.npy'),
                      np.load('./data/CPSC/train/train_normal_data_lbl_array_nonan.npy')), axis=0)
    test_org = np.concatenate((np.load('./data/CPSC/val/validation_arrhythmia_data_lbl_array_nonan.npy'),np.load('./data/CPSC/val/validation_normal_data_lbl_array_nonan.npy')), axis=0)

    train_X = train_org[:, :-1]  # 라벨 떼고
    test_X = test_org[:, :-1]  # 라벨 떼고

    #train_X = torch.Tensor(train_X)      # view로 학습하기 위해 np.array를 텐서로 바꿔줘야한다.
    #test_X = torch.Tensor(test_X)


    train_label = train_org[:, -1]
    train_label[np.where(train_label==0)] = 0
    train_label[np.where(train_label!=0)] = 1
    test_label = test_org[:, -1]
    test_label[np.where(test_label == 0)] = 0
    test_label[np.where(test_label != 0)] = 1

    #train_dataset = ECGDataset('train', train_dir, train_label_csv, leads)
    X_train = torch.from_numpy(train_X.reshape(-1,5000,8).transpose(0,2,1))      #(5000,8,xxxx)
    Y_train = torch.from_numpy(train_label)      #(xxxx,) or (xxxx,1)
    X_train = X_train.reshape(-1, 5000)
    X_train = scale(X_train, axis = 1)
    X_train = X_train.reshape(-1, 8, 5000)

    X_train = torch.from_numpy(X_train)
    #print(X_train.shape, Y_train.shape)
    #print(Y_train.shape)
    train_dataset = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    #test_dataset = ECGDataset('test', val_dir, val_label_csv, leads)
    X_test = torch.from_numpy(test_X.reshape(-1,5000,8).transpose(0,2,1))       #(5000,8,xxxx)
    X_test = X_test.reshape(-1,5000)
    Y_test = torch.from_numpy(test_label)       #(xxxx,) or (xxxx,1)

    X_test = scale(X_test, axis = 1)
    X_test = X_test.reshape(-1, 8, 5000)
    X_test = torch.from_numpy(X_test)
    test_dataset = TensorDataset(X_test, Y_test)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    train_org = np.concatenate((np.load('./data/CPSC/train/train_arrhythmia_data_lbl_array_nonan.npy'),
                      np.load('./data/CPSC/train/train_normal_data_lbl_array_nonan.npy')), axis=0)
    test_org = np.concatenate((np.load('./data/CPSC/val/validation_arrhythmia_data_lbl_array_nonan.npy'),np.load('./data/CPSC/val/validation_normal_data_lbl_array_nonan.npy')), axis=0)

    train_X = train_org[:, :-1]  # 라벨 떼고
    test_X = test_org[:, :-1]  # 라벨 떼고

    #train_X = torch.Tensor(train_X)      # view로 학습하기 위해 np.array를 텐서로 바꿔줘야한다.
    #test_X = torch.Tensor(test_X)


    train_label = train_org[:, -1]
    train_label[np.where(train_label==0)] = 0
    train_label[np.where(train_label!=0)] = 1
    test_label = test_org[:, -1]
    test_label[np.where(test_label == 0)] = 0
    test_label[np.where(test_label != 0)] = 1

    #train_dataset = ECGDataset('train', train_dir, train_label_csv, leads)
    X_train = torch.from_numpy(train_X.reshape(-1,5000,8).transpose(0,2,1))      #(5000,8,xxxx)
    Y_train = torch.from_numpy(train_label)      #(xxxx,) or (xxxx,1)
    X_train = X_train.reshape(-1, 5000)
    X_train = scale(X_train, axis = 1)
    X_train = X_train.reshape(-1, 8, 5000)

    X_train = torch.from_numpy(X_train)
    #print(X_train.shape, Y_train.shape)
    #print(Y_train.shape)
    train_dataset = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    #test_dataset = ECGDataset('test', val_dir, val_label_csv, leads)
    X_test = torch.from_numpy(test_X.reshape(-1,5000,8).transpose(0,2,1))       #(5000,8,xxxx)
    X_test = X_test.reshape(-1,5000)
    Y_test = torch.from_numpy(test_label)       #(xxxx,) or (xxxx,1)

    X_test = scale(X_test, axis = 1)
    X_test = X_test.reshape(-1, 8, 5000)
    X_test = torch.from_numpy(X_test)
    test_dataset = TensorDataset(X_test, Y_test)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    
    net = resnet34(input_channels=nleads).to(device)
    net.load_state_dict(torch.load(args.model_path, map_location=device))
    net.eval()
    
    thresholds = get_thresholds(test_loader, net, device, args.threshold_path)
    print('Thresholds:', thresholds)

    print('Results on test data:')
    apply_thresholds(test_loader, net, device, thresholds)
