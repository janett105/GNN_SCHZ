import torch
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx

from torch_geometric.datasets import KarateClub

def viz_graph(data, color):
    # graph 시각화
    # 1. node embedding 후, 저차원의 tensor 
    # 2. node embedding 전, network
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()
        plt.scatter(data[:, 0], data[:, 1], s=140, c=color, cmap="Set2")
        #if epoch is not None and loss is not None:
        #    plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
    else:
        G = to_networkx(data, to_undirected=True)
        nx.draw_networkx(G, with_labels=False,
                     node_color=color, pos=nx.spring_layout(G, seed=0))
    
    plt.show()

