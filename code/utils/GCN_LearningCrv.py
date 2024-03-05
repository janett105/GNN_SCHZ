import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
import matplotlib.pyplot as plt

from makeDir import createDirectory
import sys,os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from GCN_Kfold import GCN_Kfold
from Dataset import FCGraphDataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device='cpu'

# 고정값
n_splits = 10
n_metrics = 3  
k_order = 6
n_epoch = 50
# 설정값
#th=0.5
param_grid = {'class_weights':[torch.tensor([0.72, 1.66])]}
UpsamplingExists = False
CombatExists = False
parcel ='116'
filename = f'data0_{parcel}pc_cbt{"O" if CombatExists else"X"}_up{"O" if UpsamplingExists else "X"}_{param_grid["class_weights"]}_{n_epoch}epc'

dataset = FCGraphDataset('data')
labels = pd.read_csv(f'data/raw/Labels_{parcel}parcels.csv').loc[:,'diagnosis']
labels = labels.map({'CONTROL' : 0, 'SCHZ' : 1}).values
batch = pd.read_csv(f'data/raw/Labels_{parcel}parcels.csv').loc[:,'dataset']
batch = batch.map({'UCLA_CNP' : 0, 'COBRE' : 1}).values
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)

print()
print(dataset)
print(dataset[0])
print("====================================================================")

# HC와 SCZ환자의 site effect 비율
# HC, SCZ = HC_SCZ_SiteEffectExists()
# print(f"Combat 전 - HC - Site Effect Rate : {HC}")
# print(f"Combat 전 - SCZ - Site Effect Rate : {SCZ}")
#########################################################################################################################
#GCN_Kfold(dataset, labels, batch, param_grid, skf, CombatExists, UpsamplingExists, n_epoch, n_splits, n_metrics, k_order, dataset.num_features, dataset.num_classes, device, savefiglog=False)

one = int(len(dataset)*0.2)
testlrndata= {'n_data':[one, one*2, one*3, one*4, len(dataset)],
               'sen_avg':[],
               'sen_std':[],
               'spe_avg':[],
               'spe_std':[],
               'bac_avg':[],
               'bac_std':[]}
trainlrndata= {'n_data':[one, one*2, one*3, one*4, len(dataset)],
               'sen_avg':[],
               'sen_std':[],
               'spe_avg':[],
               'spe_std':[],
               'bac_avg':[],
               'bac_std':[]}

for datarate in [0.2, 0.4, 0.6, 0.8, 1]:
    print("data rate : ", datarate)

    if datarate==1:
        testeval, traineval = GCN_Kfold(dataset, labels, batch, param_grid, skf, 
                       CombatExists, UpsamplingExists, n_epoch, n_splits, n_metrics, k_order, 
                       device, savfig=True)
    else:
        idx=np.arange(len(dataset))
        _, lenidx, _, lenlabels, _, lenbatch  = train_test_split(idx, labels, batch, 
                                                    test_size=datarate, shuffle=True, stratify=labels)
        lendataset = dataset[lenidx.tolist()]

        testeval, traineval = GCN_Kfold(lendataset, lenlabels, lenbatch, param_grid, skf, 
                            CombatExists, UpsamplingExists, n_epoch, n_splits, n_metrics, k_order, parcel, 
                            device, savfig=False)
        
    testlrndata['sen_avg'].append(testeval[:, 0].mean()) 
    testlrndata['sen_std'].append(testeval[:, 0].std())
    testlrndata['spe_avg'].append(testeval[:, 1].mean()) 
    testlrndata['spe_std'].append(testeval[:, 1].std())
    testlrndata['bac_avg'].append(testeval[:, 2].mean()) 
    testlrndata['bac_std'].append(testeval[:, 2].std())

    trainlrndata['sen_avg'].append(traineval[:, 0].mean()) 
    trainlrndata['sen_std'].append(traineval[:, 0].std())
    trainlrndata['spe_avg'].append(traineval[:, 1].mean()) 
    trainlrndata['spe_std'].append(traineval[:, 1].std())
    trainlrndata['bac_avg'].append(traineval[:, 2].mean()) 
    trainlrndata['bac_std'].append(traineval[:, 2].std())


for metname in ['sen', 'spe','bac']:
    plt.plot(trainlrndata['n_data'], trainlrndata[f'{metname}_avg'],
            color='blue', marker='o', label=f'Training {metname}')
    plt.fill_between(trainlrndata['n_data'],
                    np.array(trainlrndata[f'{metname}_avg']) + np.array(trainlrndata[f'{metname}_std']),
                    np.array(trainlrndata[f'{metname}_avg']) - np.array(trainlrndata[f'{metname}_std']),
                    alpha=.15, color='blue')

    plt.plot(testlrndata['n_data'], testlrndata[f'{metname}_avg'],
            color='red', marker='o', label=f'Test {metname}')
    plt.fill_between(testlrndata['n_data'],
                    np.array(testlrndata[f'{metname}_avg']) + np.array(testlrndata[f'{metname}_std']),
                    np.array(testlrndata[f'{metname}_avg']) - np.array(testlrndata[f'{metname}_std']),
                    alpha=.15, color='red')

    plt.grid()
    plt.xlabel('Number of training examples')
    plt.ylabel(f'{metname}')
    plt.legend(loc='lower right')
    plt.ylim([0, 1.03])
    plt.tight_layout()

    createDirectory(f'results/figs/new/{filename}/LearningCrv/')
    plt.savefig(f'results/figs/new/{filename}/LearningCrv/{metname}.png')
    plt.show()