import argparse
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from resnet import resnet34
from utils import cal_scores, find_optimal_threshold
from sklearn.preprocessing import scale

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data/test', help='Directory to data dir')
    parser.add_argument('--leads', type=int, default=8, help='ECG leads to use')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')   # batch size 1로 해야 오류 안남
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers to load data')
    parser.add_argument('--use-gpu', default=True, action='store_true', help='Use gpu')
    parser.add_argument('--model-path', type=str, default='models/epoch12_auc0.9908.pth', help='Path to saved model')
    return parser.parse_args()

def get_thresholds(val_loader, net, device, threshold_path):
    print('Finding optimal thresholds...')
    if os.path.exists(threshold_path):
        return pickle.load(open(threshold_path, 'rb'))
    output_list, label_list = [], []
    for _, (data, labels) in enumerate(tqdm(val_loader)):
        data, labels = data.type(torch.FloatTensor).to(device), labels.type(torch.FloatTensor).to(device)
        output = net(data).type(torch.FloatTensor).to(device)
        labels = torch.reshape(labels, (output.shape[0], output.shape[1]))
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
    return thresholds


def apply_thresholds(test_loader, net, device, thresholds):
    output_list, label_list = [], []
    for idx, (data, labels) in enumerate(tqdm(test_loader)):
        data, labels = data.type(torch.FloatTensor).to(device), labels.type(torch.FloatTensor).to(device)
        output = net(data).type(torch.FloatTensor).to(device)
        labels = torch.reshape(labels, (output.shape[0], output.shape[1]))
        output = torch.sigmoid(output)
        output_list.append(output.data.cpu().numpy())
        label_list.append(labels.data.cpu().numpy())
    y_trues = np.vstack(label_list)
    y_scores = np.vstack(output_list)
    y_preds = []
    scores = []

    y_true = y_trues[:, 0]
    y_score = y_scores[:, 0]
    y_pred = (y_score >= thresholds[0]).astype(int)    # 여기서 threshold 기준으로 실제 값으로 변환

    wrong = (y_true != y_pred)
    print("오 분류한 validation 데이터 수: ",wrong.sum())
    wrong_list = np.asarray(np.where(wrong)).reshape(198)
    #print(wrong_list.shape)
    print("부정맥인데 정상으로 잘못 예측한 데이터: ",wrong_list[:83])        # 틀리게 분류한 애들 인덱스 출력
    print("정상인데 부정맥으로 잘못 예측한 데이터: ", wrong_list[83:])       # tuple 형식

    scores.append(cal_scores(y_true, y_pred, y_score))
    y_preds.append(y_pred)
    y_preds = np.array(y_preds).transpose()
    scores = np.array(scores)
    print('Test Precision:', scores[:, 0])
    print('Test Recall:', scores[:, 1])
    print('Test F1:', scores[:, 2])
    print('Test AUC:', scores[:, 3])
    print('Test Acc:', scores[:, 4])
    plot_cm(y_trues, y_preds)

def plot_cm(y_trues, y_preds, normalize=True, cmap=plt.cm.Blues):
    classes = ['Normal', 'Abnormal']
    for i, label in enumerate(classes):
        y_true = y_trues[:, 0]
        y_pred = y_preds[:, 0]
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
        args.model_path = f'models/resnet34_{database}_{args.leads}.pth'
    args.threshold_path = f'models/{database}-threshold.pkl'
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = 'cpu'

    test_org = np.concatenate((np.load('./data/val/arrhythmia.npy'), np.load('./data/val/normal.npy')), axis=0)

    test_X, test_label = test_org[:, :-1], test_org[:, -1]  # 라벨 떼고
    test_label[np.where(test_label == 0)] = 0
    test_label[np.where(test_label != 0)] = 1

    Y_test = torch.from_numpy(test_label)
    X_test = torch.from_numpy(scale(test_X.reshape(-1, 5000), axis=1).reshape(-1, 8, 5000))
    test_dataset = TensorDataset(X_test, Y_test)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    net = resnet34(input_channels=args.leads).to(device)
    net.load_state_dict(torch.load(args.model_path, map_location=device))
    net.eval()
    
    thresholds = get_thresholds(test_loader, net, device, args.threshold_path)
    print('Threshold:', thresholds)

    print('Results on test data:')
    apply_thresholds(test_loader, net, device, thresholds)

