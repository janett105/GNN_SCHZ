import torch
from torch.nn import Linear
import torch.nn.functional as func 
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.nn import ChebConv
import numpy as np
from sklearn.metrics import balanced_accuracy_score, recall_score , confusion_matrix

class GCN(torch.nn.Module):
    # Conv func : GCNConv, ChebConv
    # activation func : tanh, relu
    # (node 하나의 features #) -> 64 (actv func, drp) -> 64 (actv func, drp) -> 128 (actv func, drp) -> MLP(4)
    # input : node feature matrix(x), adjacency matrix(edge_index), 
    # output : h, out
    #    h : node embedding 최종 결과, 그래프의 node를 2차원(저차원)으로 mapping
    #    out: MLP를 통해 전체 node를 고려, node class가 4가지이므로 2-> 4 mapping
    def __init__(self, 
                 num_features,
                 num_classes,
                 k_order,
                 dropout = .5):
        super().__init__()

        self.p = dropout
        self.k = k_order
        
        self.conv1 = ChebConv(int(num_features), 64, K=self.k)
        self.conv2 = ChebConv(64, 64, K=self.k)
        self.conv3 = ChebConv(64, 128, K=self.k)
        self.lin1 = Linear(128, int(num_classes))

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch

        h = func.relu(self.conv1(x, edge_index, edge_attr))

        h = func.dropout(h, p=self.p, training=self.training)
        h = func.relu(self.conv2(h, edge_index, edge_attr))

        h = func.dropout(h, p=self.p, training=self.training)
        h = func.relu(self.conv3(h, edge_index, edge_attr))

        h = global_mean_pool(h, batch)
        out = self.lin1(h)
        return func.log_softmax(out, dim=1)
    
def GCN_train(model, optimizer, loader, weight, len_train_dataset, device='cpu'):
    model.train()

    label = []
    pred = []
    train_loss_all = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data).to(device)

        train_loss = func.nll_loss(output, data.y, weight=weight.to(device)).to(device)

        train_loss_all += data.num_graphs * train_loss.item()
        pred.append(output.argmax(dim=1))
        label.append(data.y)

        train_loss.backward()
        optimizer.step()

    y_pred = torch.cat(pred, dim=0).cpu().detach().numpy()
    y_true = torch.cat(label, dim=0).cpu().detach().numpy()

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    epoch_sen = recall_score(y_true, y_pred)
    epoch_spe = tn / (tn + fp)
    epoch_bac = balanced_accuracy_score(y_true, y_pred)
    return epoch_sen, epoch_spe, epoch_bac, train_loss_all / len_train_dataset

def GCN_test(model, loader, weight, len_val_dataset, n_fold, epoch, device='cpu'):
    model.eval()

    #score=[]
    pred = []
    label = []
    val_loss_all = 0
    for data in loader:
        data = data.to(device)
        output = model(data).to(device)

        if torch.isnan(output)[0,0]:
            print('WARNING!!!!!!!!!!!!!!!!!!output is nan')
            return np.nan, np.nan, np.nan, np.nan

        val_loss = func.nll_loss(output, data.y)
        val_loss_all += data.num_graphs * val_loss.item()

        pred.append(output.argmax(dim=1))
        label.append(data.y)
        #score.append(func.softmax(output, dim=1)[:, 1]) 
        print(f'{n_fold+1} fold | {epoch} epoch | predict_prob : {output}')

    y_pred = torch.cat(pred, dim=0).cpu().detach().numpy()
    y_true = torch.cat(label, dim=0).cpu().detach().numpy()
    #y_score = torch.cat(score, dim=0).cpu().detach().numpy()

    print(f'{n_fold+1} fold | {epoch} epoch | y_true : {y_true}')
    print(f'{n_fold+1} fold | {epoch} epoch | y_pred : {y_pred}')

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    epoch_sen = recall_score(y_true, y_pred)
    
    epoch_spe = tn / (tn + fp)
    epoch_bac = balanced_accuracy_score(y_true, y_pred)
    return epoch_sen, epoch_spe, epoch_bac, val_loss_all / len_val_dataset