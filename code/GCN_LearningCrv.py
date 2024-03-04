import pandas as pd
from pathlib import Path
import torch

from Model_GCN import GCN
from Dataset import FCGraphDataset

from tutorial.LearningCurve_tutorial import LearningCrv

k_order = 10 # KNN 

dataset = FCGraphDataset('data')
labels = pd.read_csv(Path(dataset.raw_dir)/'Labels_164parcels.csv').loc[:,'diagnosis']
labels = labels.map({'CONTROL' : 0, 'SCHZ' : 1}).values

model = GCN(dataset.num_features, dataset.num_classes, k_order)
print("DSD")
LearningCrv([model], dataset, labels)