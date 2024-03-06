import torch
from torch_geometric.utils import dense_to_sparse
import numpy as np

from AdjacencyMat import compute_KNN_graph

def CombatHrm(cbt,
              train_dataset, train_batch, train_labels,
              val_dataset, val_batch,
              test_dataset, test_batch):

    xarrays = torch.tensor([0]*6670)
    for train_data in train_dataset:
        xarray = train_data.x[np.triu_indices_from(train_data.x, k=1)]
        xarrays = torch.cat([xarrays, xarray], dim=0)
    cbt_x=cbt.fit_transform(data=xarrays, sites=train_batch.reshape(-1,1),  **{'discrete_covariates': train_labels.reshape(-1,1)})
    cbt_adj = compute_KNN_graph(torch.from_numpy(cbt_x))
    cbt_adj = torch.from_numpy(cbt_adj).float()
    cbt_edge_index, cbt_edge_attr = dense_to_sparse(cbt_adj)

    train_data.x, train_data.edge_index, train_data.edge_attr = cbt_x, cbt_edge_index, cbt_edge_attr
    print('combat 후 : ', train_dataset[0])
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