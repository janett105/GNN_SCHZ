from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import dense_to_sparse

from AdjacencyMat import compute_KNN_graph

class FCGraphDataset(InMemoryDataset):
    """
    preprocessed fMRI data (by fmriprep)
    -> FC -> dataset(feature matrix(x), adjancency matrix(edge_index), edge_attribute, y(label))
    """
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

        self.load(self.processed_paths[0]) # data, slices load

    @property
    def raw_file_names(self):
        file_paths = sorted(list(Path(self.raw_dir).glob("*.npy")))
        return [str(file_path.name) for file_path in file_paths]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        self.labels = pd.read_csv(Path(self.raw_dir)/'Labels_116parcels.csv').loc[:,'diagnosis']
        self.labels = self.labels.map({'HC' : 0, 'SCHZ' : 1}).values
<<<<<<< Updated upstream
<<<<<<< Updated upstream
=======
    
>>>>>>> Stashed changes
=======
    
>>>>>>> Stashed changes

        data_list=[]
        for filepath, y in zip(self.raw_paths, self.labels):
            y = torch.tensor(y)
<<<<<<< Updated upstream
<<<<<<< Updated upstream

=======
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
            connectivity = np.load(filepath)
            x = torch.from_numpy(connectivity).float()

            adj = compute_KNN_graph(connectivity)
            adj = torch.from_numpy(adj).float()
            edge_index, edge_attr = dense_to_sparse(adj)

            data_list.append(Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])

dataset = FCGraphDataset('data')