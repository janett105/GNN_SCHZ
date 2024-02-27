# import os
# print(os.getcwd())
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import balanced_accuracy_score, recall_score , confusion_matrix
import torch.nn.functional as func
from torch_geometric.loader import DataLoader
from neurocombat_sklearn import CombatModel

from Model_GCN import GCN
from Dataset import FCGraphDataset
from ROC_AUC import ROC_threshold
from vizGraph import viz_graph

# dataset + parcels + combat 
name = 'data0_164parcel_'

n_splits = 10 # n fold CV
n_metrics = 3 # balanced accuracy, 
k_order = 10 # KNN 
n_epoch = 50

# 출력결과 파일 저장 
sys.stdout = open(f'results/stdouts/{name}.txt', 'w')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = FCGraphDataset('data')
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
eval_metrics = np.zeros((n_splits, n_metrics))
labels = pd.read_csv(Path(dataset.raw_dir)/'Labels_164parcels.csv').loc[:,'diagnosis']
labels = labels.map({'CONTROL' : 0, 'SCHZ' : 1}).values
batch = pd.read_csv('data/raw/Labels_164parcels.csv').loc[:,'dataset']
batch = batch.map({'UCLA_CNP' : 0, 'COBRE' : 1}).values
thresholds = {}

def GCN_train(loader):
    model.train()

    train_loss_all = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        output, h = model(data)
        train_loss = func.cross_entropy(output, data.y)
        train_loss.backward()
        train_loss_all += data.num_graphs * train_loss.item()
        optimizer.step()

    return train_loss_all / len(train_dataset)

def GCN_test(loader):
    model.eval()

    score=[]
    pred = []
    label = []
    val_loss_all = 0
    for data in loader:
        

        data = data.to(device)
        output, h = model(data)
        val_loss = func.cross_entropy(output, data.y)
        val_loss_all += data.num_graphs * val_loss.item()
        
        
        if dataset.num_classes == 2:
            print(f'{n_fold+1} fold | {epoch} epoch | predict_prob : {func.softmax(output, dim=1)}')
            pred.append((func.softmax(output, dim=1)[:, 1]>0.38).type(torch.int))
            label.append(data.y)
            score.append(func.softmax(output, dim=1)[:, 1])

    y_pred = torch.cat(pred, dim=0).cpu().detach().numpy()
    y_true = torch.cat(label, dim=0).cpu().detach().numpy()
    y_score = torch.cat(score, dim=0).cpu().detach().numpy()

    print(f'{n_fold+1} fold | {epoch} epoch | y_true : {y_true}')
    print(f'{n_fold+1} fold | {epoch} epoch | y_pred : {y_pred}')

    if dataset.num_classes == 2:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        epoch_sen = recall_score(y_true, y_pred)
        epoch_spe = tn / (tn + fp)
        epoch_bac = balanced_accuracy_score(y_true, y_pred)
        return epoch_sen, epoch_spe, epoch_bac, val_loss_all / len(val_dataset), y_true, y_score

for n_fold, (train_val, test) in enumerate(skf.split(labels, labels)):  
    print(f'=============== {n_fold+1} fold ===============')
    model = GCN(dataset.num_features, dataset.num_classes, k_order).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    cbt = CombatModel()

    train_val_dataset, test_dataset = dataset[train_val.tolist()], dataset[test.tolist()]
    train_val_labels = labels[train_val]    
    train_val_index = np.arange(len(train_val_dataset))
    train_val_batch = batch[train_val]
    test_batch = batch[test]
    train_idx, val_idx, train_labels, val_labels, train_batch, val_batch  = train_test_split(train_val_index, train_val_labels, train_val_batch, test_size=0.11, shuffle=True, stratify=train_val_labels)
    train_dataset, val_dataset = train_val_dataset[train_idx.tolist()], train_val_dataset[val_idx.tolist()]
    train_x = train_dataset.x
    val_x = val_dataset.x
    test_x = test_dataset.x

    # Combat harmonization
    # adjusted_train_x = cbt.fit_transform(x=train_x, sites=train_batch, discrete_covariates = train_labels)
    # train_dataset.x = adjusted_train_x
    # adjusted_val_x = cbt.fit_transform(x=val_x, sites=val_batch, discrete_covariates = val_labels)
    # val_dataset.x = adjusted_val_x
    # adjusted_test_x = cbt.transform(x=test_x, sites=test_batch)
    # test_dataset.x = adjusted_test_x

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True) # 64 graphs
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True) # 40 graphs
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True) # 12 graphs

    min_v_loss = np.inf
    for epoch in range(n_epoch):
        t_loss = GCN_train(train_loader)
        val_sen, val_spe, val_bac, v_loss, v_true, v_score = GCN_test(val_loader)
        test_sen, test_spe, test_bac, _, test_true, test_score = GCN_test(test_loader)

        if min_v_loss > v_loss:
            min_v_loss = v_loss
            best_val_bac = val_bac
            best_test_sen, best_test_spe, best_test_bac = test_sen, test_spe, test_bac
            torch.save(f'results/best_models/{name}', f'{n_fold + 1}fold.pth')
            print('CV: {:03d}, Epoch: {:03d}, Val Loss: {:.5f}, Val BAC: {:.5f}, Test BAC: {:.5f}, TEST SEN: {:.5f}, '
                  'TEST SPE: {:.5f}'.format(n_fold + 1, epoch + 1, min_v_loss, best_val_bac, best_test_bac,
                                            best_test_sen, best_test_spe))

        # 각 epoch에서의 val, test에서의 best threshold 
        thresholds_epochs = [[]]*n_epoch
        thresholds_epochs[epoch] = ROC_threshold(v_true, v_score, test_true, test_score)

    eval_metrics[n_fold, 0] = best_test_sen
    eval_metrics[n_fold, 1] = best_test_spe
    eval_metrics[n_fold, 2] = best_test_bac
    
    thresholds[f'{n_fold+1} fold'] = thresholds_epochs

eval_df = pd.DataFrame(eval_metrics)
eval_df.columns = ['SEN', 'SPE', 'BAC']
eval_df.index = ['Fold_%02i' % (i + 1) for i in range(skf.n_splits)]
print(eval_df)
print('Average Sensitivity: %.4f+-%.4f' % (eval_metrics[:, 0].mean(), eval_metrics[:, 0].std()))
print('Average Specificity: %.4f+-%.4f' % (eval_metrics[:, 1].mean(), eval_metrics[:, 1].std()))
print('Average Balanced Accuracy: %.4f+-%.4f' % (eval_metrics[:, 2].mean(), eval_metrics[:, 2].std()))
# {n_fold+1} fold의 각 epoch에서의 [best val threshold, best val threshold]
#print(thresholds)

sys.stdout.close()