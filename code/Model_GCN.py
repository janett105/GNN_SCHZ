import torch
from torch.nn import Linear
import torch.nn.functional as func 
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.nn import ChebConv
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
        torch.manual_seed(0)

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
        #h = func.dropout(h, p=self.p, training=self.training)
        h = func.relu(self.conv2(h, edge_index, edge_attr))
        #h = func.dropout(h, p=self.p, training=self.training)
        h = func.relu(self.conv3(h, edge_index, edge_attr))

        h = global_mean_pool(h, batch)
        out = self.lin1(h)
        return out, h
    
    def fit(self, train_loader, epochs=100):
        self.train()
        class_weights = torch.tensor([0.72,1.66])

        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        for epoch in range(epochs):
            for data in train_loader:  # Assuming data is a batch from DataLoader
                optimizer.zero_grad()
                output, h = self(data)
                train_loss = func.cross_entropy(output, data.y, weight=class_weights)
                train_loss.backward()
                optimizer.step()

    def score(self, test_loader):
        self.eval()  # 모델을 평가 모드로 설정

        pred = []
        label = []
        with torch.no_grad():  # 그라디언트 계산을 중지
            for data in test_loader:
                output, h = self(data)
                pred.append((func.softmax(output, dim=1)[:, 1]>0.5).type(torch.int))
                label.append(data.y)

            y_pred = torch.cat(pred, dim=0).cpu().detach().numpy()
            y_true = torch.cat(label, dim=0).cpu().detach().numpy()

            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            epoch_sen = recall_score(y_true, y_pred)
            epoch_spe = tn / (tn + fp)
            epoch_bac = balanced_accuracy_score(y_true, y_pred)

        #return epoch_sen, epoch_spe, epoch_bac
        return epoch_bac

    def get_params(self, deep=True):
        # 생성자에 전달된 모든 매개변수와 그 값을 반환합니다.
        return {'num_features': self.conv1.in_channels,
                'num_classes': self.lin1.out_features,
                'k_order': self.k,
                'dropout': self.p}