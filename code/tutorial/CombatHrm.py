import numpy as np
import pandas as pd
from pathlib import Path

from neurocombat_sklearn import CombatModel

class CombatHrm():
    # Dataset 0 : UCLA_CNP (HC 52, SCZ 25)
    # Dataset 1 : COBRE ()
    n_samples0 = 77 
    n_samples1 = 117

    batch = pd.read_csv('data/raw/Labels.csv').loc[:,'dataset']
    batch = batch.map({'UCLA_CNP' : 0, 'COBRE' : 1}).values

    labels = pd.read_csv('data/raw/Labels.csv').loc[:,'diagnosis']
    labels = labels.map({'CONTROL' : 0, 'SCHZ' : 1}).values

    def fit_combat(combat, features, batch, labels):
        return combat.fit_transform(features, batch, covariates=labels)
    def transform_combat(): 
        return combat.transform(features, batch, covariates=labels)