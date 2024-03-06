import torch
from torch_geometric.utils import dense_to_sparse
import numpy as np
from torch_geometric.data import Data

from AdjacencyMat import compute_KNN_graph

def diagtoary(dataset):
    xarrays_list = []
    for data in dataset:
        x = data.x
        rows, cols = torch.triu_indices(x.size(0), x.size(1), offset=1)
        xarray = x[rows, cols]
        xarray = xarray.unsqueeze(0) 
        xarrays_list.append(xarray)
    xarrays = torch.cat(xarrays_list, dim=0)
    return xarrays

def newdataset(dataset, labels, cbt_xarrays, parcel):
    cbt_data_list=[]
    for idx, train_data in enumerate(dataset):
        cbt_x = torch.zeros((parcel, parcel))
        rows, cols = torch.triu_indices(parcel, parcel, offset=1)
        cbt_x[rows, cols] = torch.tensor(cbt_xarrays[idx]).float()

        cbt_adj = compute_KNN_graph(cbt_x)
        cbt_edge_index, cbt_edge_attr = dense_to_sparse(torch.tensor(cbt_adj))
        cbt_data_list.append(Data(x=cbt_x, y=torch.tensor(labels[idx]), edge_index=cbt_edge_index, edge_attr=cbt_edge_attr))
    return cbt_data_list

def CombatHrm(cbt,parcel,
              train_dataset, train_batch, train_labels,
              val_dataset, val_batch,val_labels,
              test_dataset, test_batch, test_labels):
    
    train_xarrays=diagtoary(train_dataset)
    #cbt_train_xarrays=cbt.fit_transform(data=train_xarrays, sites=train_batch.reshape(-1,1),  **{'discrete_covariates': train_labels.reshape(-1,1)})
    cbt_train_xarrays=cbt.fit_transform(data=train_xarrays, sites=train_batch.reshape(-1,1))
    cbt_traindata_list=newdataset(train_dataset, train_labels, cbt_train_xarrays, parcel)

    val_xarrays=diagtoary(val_dataset)
    cbt_val_xarrays=cbt.transform(data=val_xarrays, sites=val_batch.reshape(-1,1))
    cbt_valdata_list=newdataset(val_dataset, val_labels, cbt_val_xarrays, parcel)

    test_xarrays=diagtoary(test_dataset)
    cbt_test_xarrays=cbt.transform(data=test_xarrays, sites=test_batch.reshape(-1,1))
    cbt_testdata_list=newdataset(test_dataset, test_labels, cbt_test_xarrays, parcel)

    return cbt_traindata_list, cbt_valdata_list, cbt_testdata_list