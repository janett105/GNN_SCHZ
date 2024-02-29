import os
print(os.getcwd())
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import sys
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import balanced_accuracy_score, recall_score , confusion_matrix
import torch.nn.functional as func
from torch_geometric.loader import DataLoader, ImbalancedSampler
from neurocombat_sklearn import CombatModel

from Model_GCN import GCN
from Dataset import FCGraphDataset
from utils.ROC_AUC import ROC_threshold
from utils.vizGraph import viz_graph
from utils.SiteEffect import HC_SCZ_SiteEffectExists
from utils.FocalLoss import sigmoid_focal_loss
from utils.makeDir import createDirectory

n_splits = 10 # n fold CV
n_metrics = 3 # balanced accuracy, 
k_order = 10 # KNN 

n_epoch = 200
th=0.5
class_weights=torch.tensor([0.72,1.66])
UpsamplingExists = False
criterion = 'focal' #CE
"""
focal : gamma=2, alpha=0.75, reduction='sum'
CD : weights=[0.72,1.66]
"""
# dataset + parcels + combat + upsampling + loss func + n_epoch
filename = f'data0_164pc_cbtX_upX_{criterion}_{n_epoch}epc'

##########################################################################################
sys.stdout = open(f'results/stdouts/{filename}.txt', 'w')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = FCGraphDataset('data').to(device)
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
eval_metrics = np.zeros((n_splits, n_metrics))
labels = pd.read_csv(Path(dataset.raw_dir)/'Labels_164parcels.csv').loc[:,'diagnosis']
labels = labels.map({'CONTROL' : 0, 'SCHZ' : 1}).values
batch = pd.read_csv('data/raw/Labels_164parcels.csv').loc[:,'dataset']
batch = batch.map({'UCLA_CNP' : 0, 'COBRE' : 1}).values
#thresholds = {}
historys_loss=[]
historys_sen=[]
historys_spe=[]
historys_bac=[]

##########################################################################################
print(dataset)
print("=========================================")
print(dataset[0])
print("=========================================")
# HC와 SCZ환자의 site effect 비율
# HC, SCZ = HC_SCZ_SiteEffectExists()
# print(f"Combat 전 - HC - Site Effect Rate : {HC}")
# print(f"Combat 전 - SCZ - Site Effect Rate : {SCZ}")

def GCN_train(loader):
    model.train()

    label = []
    pred = []
    train_loss_all = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        output, h = model(data)

        if criterion == 'focal':
            train_loss = sigmoid_focal_loss(inputs=output[:,1].float(), targets=data.y.float(), gamma=2, alpha=0.75, reduction='sum')
        elif criterion == 'CE':
            train_loss = func.cross_entropy(output, data.y, weight=class_weights) 
        
        train_loss_all += data.num_graphs * train_loss.item()
        pred.append((func.softmax(output, dim=1)[:, 1]>th).type(torch.int))
        label.append(data.y)

        train_loss.backward()
        optimizer.step()

    y_pred = torch.cat(pred, dim=0).cpu().detach().numpy()
    y_true = torch.cat(label, dim=0).cpu().detach().numpy()

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    epoch_sen = recall_score(y_true, y_pred)
    epoch_spe = tn / (tn + fp)
    epoch_bac = balanced_accuracy_score(y_true, y_pred)
    return epoch_sen, epoch_spe, epoch_bac, train_loss_all / len(train_dataset)

def GCN_test(loader):
    model.eval()

    #score=[]
    pred = []
    label = []
    val_loss_all = 0
    for data in loader:
        data = data.to(device)
        output, h = model(data) 

        if criterion == 'focal':
            val_loss = sigmoid_focal_loss(inputs=output[:,1].float(), targets=data.y.float(), gamma=2, alpha=0.75, reduction='sum')
        elif criterion == 'CE':
            val_loss = func.cross_entropy(output, data.y, weight=class_weights) 
        val_loss_all += data.num_graphs * val_loss.item()

        pred.append((func.softmax(output, dim=1)[:, 1]>th).type(torch.int))
        label.append(data.y)
        #score.append(func.softmax(output, dim=1)[:, 1]) 
        print(f'{n_fold+1} fold | {epoch} epoch | predict_prob : {func.softmax(output, dim=1)}')

    y_pred = torch.cat(pred, dim=0).cpu().detach().numpy()
    y_true = torch.cat(label, dim=0).cpu().detach().numpy()
    #y_score = torch.cat(score, dim=0).cpu().detach().numpy()

    print(f'{n_fold+1} fold | {epoch} epoch | y_true : {y_true}')
    print(f'{n_fold+1} fold | {epoch} epoch | y_pred : {y_pred}')

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    epoch_sen = recall_score(y_true, y_pred)
    epoch_spe = tn / (tn + fp)
    epoch_bac = balanced_accuracy_score(y_true, y_pred)
    return epoch_sen, epoch_spe, epoch_bac, val_loss_all / len(val_dataset)

for n_fold, (train_val, test) in enumerate(skf.split(labels, labels)):  
    print(f'=============== {n_fold+1} fold ===============')
    model = GCN(dataset.num_features, dataset.num_classes, k_order).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    cbt = CombatModel().to(device)

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

    if UpsamplingExists==True:
        train_sampler = ImbalancedSampler(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=64, sampler=train_sampler)
    else: 
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True) 
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    min_v_loss = np.inf
    history_loss={'epoch':[], 't_loss':[], 'tt_loss':[]}
    history_sen={'epoch':[], 't_sen':[], 'tt_sen':[]}
    history_spe={'epoch':[], 't_spe':[], 'tt_spe':[]}
    history_bac={'epoch':[], 't_bac':[], 'tt_bac':[]}
    for epoch in range(n_epoch):
        train_sen, train_spe, train_bac, t_loss = GCN_train(train_loader)
        val_sen, val_spe, val_bac, v_loss = GCN_test(val_loader)
        test_sen, test_spe, test_bac, _ = GCN_test(test_loader)
        print(f'train loss : {t_loss} , val loss : {v_loss}')
        history_loss['epoch'].append(epoch)
        history_loss['t_loss'].append(t_loss)
        history_loss['tt_loss'].append(v_loss)
        history_sen['epoch'].append(epoch)
        history_sen['t_sen'].append(train_sen)
        history_sen['tt_sen'].append(test_sen)
        history_spe['epoch'].append(epoch)
        history_spe['t_spe'].append(train_spe)
        history_spe['tt_spe'].append(test_spe)
        history_bac['epoch'].append(epoch)
        history_bac['t_bac'].append(train_bac)
        history_bac['tt_bac'].append(test_bac)

        if min_v_loss > v_loss:
            min_v_loss = v_loss
            best_val_bac = val_bac
            best_test_sen, best_test_spe, best_test_bac = test_sen, test_spe, test_bac
            createDirectory(f'results/best_models/{filename}')
            torch.save(model.state_dict(), f'results/best_models/{filename}/{n_fold + 1}fold.pth')
            print('CV: {:03d}, Epoch: {:03d}, Val Loss: {:.5f}, Val BAC: {:.5f}, Test BAC: {:.5f}, TEST SEN: {:.5f}, '
                  'TEST SPE: {:.5f}'.format(n_fold + 1, epoch + 1, min_v_loss, best_val_bac, best_test_bac,
                                            best_test_sen, best_test_spe))

        #thresholds_epochs = [[]]*n_epoch
        #thresholds_epochs[epoch] = ROC_threshold(v_true, v_score, test_true, test_score)

    eval_metrics[n_fold, 0] = best_test_sen
    eval_metrics[n_fold, 1] = best_test_spe
    eval_metrics[n_fold, 2] = best_test_bac
    
    #thresholds[f'{n_fold+1} fold'] = thresholds_epochs
    historys_loss.append(history_loss)
    historys_sen.append(history_sen)
    historys_spe.append(history_spe)
    historys_bac.append(history_bac)

eval_df = pd.DataFrame(eval_metrics)
eval_df.columns = ['SEN', 'SPE', 'BAC']
eval_df.index = ['Fold_%02i' % (i + 1) for i in range(skf.n_splits)]
print(eval_df)
print('Average Sensitivity: %.4f+-%.4f' % (eval_metrics[:, 0].mean(), eval_metrics[:, 0].std()))
print('Average Specificity: %.4f+-%.4f' % (eval_metrics[:, 1].mean(), eval_metrics[:, 1].std()))
print('Average Balanced Accuracy: %.4f+-%.4f' % (eval_metrics[:, 2].mean(), eval_metrics[:, 2].std()))
#print(thresholds)

for met in range(n_metrics+1):
    if met==0:
        metname='loss' 
        hs=historys_loss
    elif met==1:
        metname='sen'
        hs=historys_sen
    elif met==2:
        metname='spe'
        hs=historys_spe
    else: 
        metname='bac'
        hs=historys_bac

    print(f'total {metname} : {hs}')
    sum_t=np.array(hs[0][f't_{metname}'])
    sum_tt=np.array(hs[0][f'tt_{metname}'])
    h={'epoch':hs[0]['epoch'], f'training_{metname}':[], f'test(val)_{metname}':[]}
    for i in range(1,n_splits):
        sum_t += np.array(hs[i][f't_{metname}'])
        sum_tt += np.array(hs[i][f'tt_{metname}'])
    h[f'training_{metname}'] = sum_t/n_splits
    h[f'test(val)_{metname}'] = sum_tt/n_splits
    print(f'avg {metname} : {h}')

    plt.plot(h['epoch'], h[f'training_{metname}'], marker='.', c='blue', label = f'training_{metname}')
    plt.plot(h['epoch'], h[f'test(val)_{metname}'], marker='.', c='red', label = f'test(val)_{metname}')
    plt.legend(loc='upper right')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel(f'avg {metname}')
    if metname=='loss':
        plt.ylim([0,10])
    else:
        plt.ylim([0,1])

    createDirectory(f'results/figs/{filename}')
    plt.savefig(f'results/figs/{filename}/{metname}.png')
    plt.show()

sys.stdout.close()