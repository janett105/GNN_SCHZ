import time
import torch
from torch.nn import Linear, CrossEntropyLoss
from torch_geometric.datasets import KarateClub
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_networkx
import networkx as nx

import matplotlib.pyplot as plt
import numpy as np

dataset= KarateClub()

def viz_graph(h, color):
    # graph 시각화
    # 1. node embedding 후, 저차원의 tensor 
    # 2. node embedding 전, network
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    
    if torch.is_tensor(h):
        h = h.detach().cpu().numpy()
        plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
        #if epoch is not None and loss is not None:
        #    plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
    else:
        nx.draw_networkx(h, with_labels=False,
                     node_color=color, pos=nx.spring_layout(G, seed=0))
    
    plt.show()

class GCN(torch.nn.Module):
    # 34(node 하나의 features #) -> 4 (tanh) -> 4 (tanh) -> 2 (tanh) -> MLP
    # input : node feature matrix(x), adjacency matrix(edge_index)
    # output : h, out
    #    h : node embedding 최종 결과, 그래프의 node를 2차원(저차원)으로 mapping
    #    out: MLP를 통해 전체 node를 고려, node class가 4가지이므로 2-> 4 mapping
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        
        self.conv1 = GCNConv(data.num_node_features, 4)
        self.conv2 = GCNConv(4, 4)
        self.conv3 = GCNConv(4, 2)
        self.classifier = Linear(2, dataset.num_classes)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        h = self.conv3(h, edge_index)
        h = h.tanh()

        out = self.classifier(h)

        return out, h

def train_model(data):
    optimizer.zero_grad()
    out, h = model(data.x, edge_index)
    #viz_graph(h, color=data.y)
    print(out.shape)

    loss = lossfunc(out[data.train_mask], data.y[data.train_mask])  #loss 계산
    loss.backward() # gradients 계산
    optimizer.step() # 계산한 gradient로 parameter update

    return loss, h

# graph visualization
G = to_networkx(data, to_undirected=True)
viz_graph(G, color=data.y)

# model
model = GCN()
#print(model)
lossfunc = CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# train
for epoch in range(401):
    loss, h = train_model(data)
    if epoch%10==0:
        print(f'{epoch} epoch loss : {loss:.4f}')
        time.sleep(0.3)
print(f'final epoch loss : {loss:.4f}')
