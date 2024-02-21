import torch
from torch.nn import Linear
import torch.nn.functional as func 
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.nn import ChebConv

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
        
        self.conv1 = ChebConv(int(num_features), 64, K=k_order)
        self.conv2 = ChebConv(64, 64, K=k_order)
        self.conv3 = ChebConv(64, 128, K=k_order)
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