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
from torch.utils.data import TensorDataset
from sklearn.preprocessing import scale


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data', help='Directory for data dir')
    parser.add_argument('--leads', type=str, default='all', help='ECG leads to use')
    parser.add_argument('--seed', type=int, default=42, help='Seed to split data')
    parser.add_argument('--num-classes', type=int, default=int, help='Num of diagnostic classes')
    parser.add_argument('--lr', '--learning-rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=113, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4, help='Num of workers to load data')
    parser.add_argument('--phase', type=str, default='train', help='Phase: train or test')
    parser.add_argument('--epochs', type=int, default=9, help='Training epochs')
    parser.add_argument('--resume', default=False, action='store_true', help='Resume')
    parser.add_argument('--use-gpu', default=True, action='store_true', help='Use GPU')
    parser.add_argument('--model-path', type=str, default='', help='Path to saved model')
    return parser.parse_args()


def train(dataloader, net, args, criterion, epoch, scheduler, optimizer, device):
    print('Training epoch %d:' % epoch)
    net.train()
    running_loss = 0
    output_list, labels_list = [], []
    for _, (data, labels) in enumerate(tqdm(dataloader)):
        data, labels = data.type(torch.FloatTensor).to(device), labels.type(torch.FloatTensor).to(device)
        #output = torch.Tensor(np.argmax(net(data).cpu().detach().numpy(), axis =-1)).to(device)
        output = net(data).type(torch.FloatTensor).to(device)
        labels = torch.reshape(labels, (output.shape[0], output.shape[1]))
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        output_list.append(output.data.cpu().numpy())
        labels_list.append(labels.data.cpu().numpy())
    # scheduler.step()
    print('Loss: %.4f' % running_loss)
    

def evaluate(dataloader, net, args, criterion, device):
    print('Validating...')
    net.eval()
    running_loss = 0
    output_list, labels_list = [], []
    for _, (data, labels) in enumerate(tqdm(dataloader)):
        data, labels = data.type(torch.FloatTensor).to(device), labels.type(torch.FloatTensor).to(device)
        output = net(data).type(torch.FloatTensor).to(device)
        labels = torch.reshape(labels, (output.shape[0], output.shape[1]))
        loss = criterion(output, labels)
        running_loss += loss.item()
        output = torch.sigmoid(output)
        output_list.append(output.data.cpu().numpy())
        labels_list.append(labels.data.cpu().numpy())
    print('Loss: %.4f' % running_loss)
    y_trues = np.vstack(labels_list)
    # y_trues = np.where(y_trues==0)
    y_scores = np.vstack(output_list)
    f1s = cal_f1s(y_trues, y_scores)
    avg_f1 = np.mean(f1s)
    print('F1s:', f1s)
    print('Avg F1: %.4f' % avg_f1)
    if args.phase == 'train' and avg_f1 > args.best_metric:
        args.best_metric = avg_f1
        torch.save(net.state_dict(), args.model_path)
    else:
        aucs = cal_aucs(y_trues, y_scores)
        avg_auc = np.mean(aucs)
        print('AUCs:', aucs)
        print('Avg AUC: %.4f' % avg_auc)


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
    net = resnet34(input_channels=nleads, num_classes=1)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.1)
    
    criterion = nn.BCEWithLogitsLoss()
    #criterion = nn.MSELoss()
    
    if args.phase == 'train':
        if args.resume:
            net.load_state_dict(torch.load(args.model_path, map_location=device))
        for epoch in range(args.epochs):
            train(train_loader, net, args, criterion, epoch, scheduler, optimizer, device)
            evaluate(test_loader, net, args, criterion, device)
    else:
        net.load_state_dict(torch.load(args.model_path, map_location=device))
        evaluate(test_loader, net, args, criterion, device)
