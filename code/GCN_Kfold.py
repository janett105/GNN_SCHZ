import torch
import numpy as np
import pandas as pd

import sys
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch_geometric.loader import DataLoader, ImbalancedSampler
from neurocombat_sklearn import CombatModel
from sklearn.utils.class_weight import compute_class_weight

from Model_GCN import GCN, GCN_train, GCN_test
from Dataset import FCGraphDataset
from utils.SiteEffect import HC_SCZ_SiteEffectExists
from utils.makeDir import createDirectory
from utils.vizMetrics import vizMetrics
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device='cpu'

# 고정값
n_splits = 10
n_metrics = 3  
k_order = 6
n_epoch = 50
# 설정값
#th=0.5
#UpsamplingExists = True
CombatExists = False
parcel = 116

dataset = FCGraphDataset('data')
whole = pd.read_csv(f'data/raw/Labels_{parcel}parcels.csv')
labels = whole.loc[:,'diagnosis']
labels = labels.map({'CONTROL' : 0, 'SCHZ' : 1}).values
batch = whole.loc[:,'dataset']
batch = batch.map({'UCLA_CNP' : 0, 'COBRE' : 1}).values
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)

balanced_class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
param_grid = {'class_weights':[torch.tensor(balanced_class_weights.astype(np.float32)), torch.tensor([1.0, 1.0])]}
print()
print(dataset)
print(dataset[0])
print("====================================================================")

# HC와 SCZ환자의 site effect 비율
# HC, SCZ = HC_SCZ_SiteEffectExists()
# print(f"Combat 전 - HC - Site Effect Rate : {HC}")
# print(f"Combat 전 - SCZ - Site Effect Rate : {SCZ}")
#########################################################################################################################
def GCN_Kfold(dataset, labels, batch, param_grid, skf, 
                CombatExists, UpsamplingExists, n_epoch, n_splits, n_metrics, k_order, parcel,
                device, savfig=True):
    for class_weights in param_grid['class_weights']:
        if savfig:
            # dataset + parcels + combat + upsampling + loss func + n_epoch
            filename = f'data0_{parcel}pc_cbt{"O" if CombatExists else"X"}_up{"O" if UpsamplingExists else "X"}_{class_weights}_{n_epoch}epc'
            sys.stdout = open(f'results/stdouts/new/{filename}.txt', 'w')
        
        eval_metrics = np.zeros((n_splits, n_metrics))
        train_metrics = np.zeros((n_splits, n_metrics))
        historys_loss=[]
        historys_sen=[]
        historys_spe=[]
        historys_bac=[]
        isnan=False
        #thresholds = {}

        for n_fold, (train_val, test) in enumerate(skf.split(labels, labels)):  
            if isnan==True:break
            print(f'=============== {n_fold+1} fold ===============')
            model = GCN(dataset.num_features, dataset.num_classes, k_order).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
            #cbt = CombatModel()

            train_val_dataset, test_dataset = dataset[train_val.tolist()], dataset[test.tolist()]
            train_val_labels = labels[train_val]    
            train_val_index = np.arange(len(train_val_dataset))
            train_val_batch = batch[train_val]
            test_batch = batch[test]
            train_idx, val_idx, train_labels, val_labels, train_batch, val_batch  = train_test_split(train_val_index, train_val_labels, train_val_batch, 
                                                                                                    test_size=0.11, shuffle=True, stratify=train_val_labels)
            train_dataset, val_dataset = train_val_dataset[train_idx.tolist()], train_val_dataset[val_idx.tolist()]

            # Combat harmonization
            # train_x = train_dataset.x
            # val_x = val_dataset.x
            # test_x = test_dataset.x
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
                train_sen, train_spe, train_bac, t_loss = GCN_train(model, optimizer, train_loader, class_weights, len(train_dataset), device)
                val_sen, val_spe, val_bac, v_loss = GCN_test(model, val_loader, class_weights, len(val_dataset), n_fold, epoch, device)

                if np.isnan(val_sen): 
                    isnan=True
                    break

                test_sen, test_spe, test_bac, _ = GCN_test(model, test_loader, class_weights, len(val_dataset), n_fold, epoch, device)

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
                    best_train_sen, best_train_spe, best_train_bac = train_sen, train_spe, train_bac
                    if savfig:
                        createDirectory(f'results/best_models/new/{filename}')
                        torch.save(model.state_dict(), f'results/best_models/new/{filename}/{n_fold + 1}fold.pth')
                        print('CV: {:03d}, Epoch: {:03d}, Val Loss: {:.5f}, Val BAC: {:.5f}, Test BAC: {:.5f}, TEST SEN: {:.5f}, '
                            'TEST SPE: {:.5f}'.format(n_fold + 1, epoch + 1, min_v_loss, best_val_bac, best_test_bac,
                                                        best_test_sen, best_test_spe))

                #thresholds_epochs = [[]]*n_epoch
                #thresholds_epochs[epoch] = ROC_threshold(v_true, v_score, test_true, test_score)

            if isnan==True:break

            eval_metrics[n_fold, 0] = best_test_sen
            eval_metrics[n_fold, 1] = best_test_spe
            eval_metrics[n_fold, 2] = best_test_bac
            train_metrics[n_fold, 0] = best_train_sen
            train_metrics[n_fold, 1] = best_train_spe
            train_metrics[n_fold, 2] = best_train_bac

            #thresholds[f'{n_fold+1} fold'] = thresholds_epochs
            historys_loss.append(history_loss)
            historys_sen.append(history_sen)
            historys_spe.append(history_spe)
            historys_bac.append(history_bac)

        if (isnan==False) and (savfig):
            # 각 fold에서 val_loss가 최소였던 한 epoch 성능에서의 10fold 평균, 최고라고 기대되는 test 성능끼리의 평균
            eval_df = pd.DataFrame(eval_metrics)
            eval_df.columns = ['SEN', 'SPE', 'BAC']
            eval_df.index = ['Fold_%02i' % (i + 1) for i in range(skf.n_splits)]
            print(eval_df)
            print('Average Sensitivity: %.4f+-%.4f' % (eval_metrics[:, 0].mean(), eval_metrics[:, 0].std()))
            print('Average Specificity: %.4f+-%.4f' % (eval_metrics[:, 1].mean(), eval_metrics[:, 1].std()))
            print('Average Balanced Accuracy: %.4f+-%.4f' % (eval_metrics[:, 2].mean(), eval_metrics[:, 2].std()))
            #print(thresholds)
            vizMetrics(historys_loss, historys_sen, historys_spe, historys_bac, filename)
        else:
            print('nan ending...')

        if savfig:sys.stdout.close()
        return eval_metrics, train_metrics


for i in range(2):
    UpsamplingExists=bool(i)
    GCN_Kfold(dataset, labels, batch, param_grid, skf, 
                    CombatExists, UpsamplingExists, n_epoch, n_splits, n_metrics, k_order, parcel,
                    device, savfig=True)