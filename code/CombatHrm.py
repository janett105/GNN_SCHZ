import torch
from torch_geometric.utils import dense_to_sparse
import numpy as np

from AdjacencyMat import compute_KNN_graph

def CombatHrm(cbt,parcel,
              train_dataset, train_batch, train_labels,
              val_dataset, val_batch,
              test_dataset, test_batch):
    
    print('combat 전 : ', train_dataset[0].x)
    xarrays_list = []
    for train_data in train_dataset:
        x = train_data.x
        rows, cols = torch.triu_indices(x.size(0), x.size(1), offset=1)
        xarray = x[rows, cols]
        xarray = xarray.unsqueeze(0) 
        xarrays_list.append(xarray)
    xarrays = torch.cat(xarrays_list, dim=0)
    cbt_xarrays=cbt.fit_transform(data=xarrays, sites=train_batch.reshape(-1,1),  **{'discrete_covariates': train_labels.reshape(-1,1)})
    
    for idx, train_data in enumerate(train_dataset):
        cbt_x = torch.zeros((parcel, parcel))
        rows, cols = torch.triu_indices(parcel, parcel, offset=1)
        cbt_x[rows, cols] = torch.tensor(cbt_xarrays[idx]).float()

        cbt_adj = compute_KNN_graph(cbt_x)
        cbt_edge_index, cbt_edge_attr = dense_to_sparse(torch.tensor(cbt_adj))
        if idx==0:print(cbt_x)
        train_data.x, train_data.edge_index, train_data.edge_attr = cbt_x, cbt_edge_index, cbt_edge_attr
    print('combat 후 : ', train_dataset[0].x)

    for val_data in val_dataset:
        cbt_x=cbt.transform(data=val_data.x, sites=val_batch)
        cbt_adj = compute_KNN_graph(torch.from_numpy(cbt_x))
        cbt_adj = torch.from_numpy(cbt_adj).float()
        cbt_edge_index, cbt_edge_attr = dense_to_sparse(cbt_adj)

        val_data.x, val_data.edge_index, val_data.edge_attr = cbt_x, cbt_edge_index, cbt_edge_attr

    for test_data in test_dataset:
        cbt_x=cbt.transform(data=test_data.x, sites=test_batch)
        cbt_adj = compute_KNN_graph(torch.from_numpy(cbt_x))
        cbt_adj = torch.from_numpy(cbt_adj).float()
        cbt_edge_index, cbt_edge_attr = dense_to_sparse(cbt_adj)

        test_data.x, test_data.edge_index, test_data.edge_attr = cbt_x, cbt_edge_index, cbt_edge_attr