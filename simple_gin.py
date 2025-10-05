import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GINEConv, global_add_pool

class SimpleGIN(nn.Module):
    def __init__(self, num_node_feat, num_edge_feat, num_hidden=128, num_layers=3, dropout=0.2):
        super(SimpleGIN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.convs = nn.ModuleList()  # 使用 torch.nn.ModuleList
        self.use_edge_attr = num_edge_feat is not None and int(num_edge_feat) > 0
        for i in range(num_layers):
            in_dim = num_node_feat if i == 0 else num_hidden
            mlp = nn.Sequential(
                nn.Linear(in_dim, num_hidden),
                nn.ReLU(),
                nn.Linear(num_hidden, num_hidden)
            )
            if self.use_edge_attr:
                # 显式使用边特征
                self.convs.append(GINEConv(nn=mlp, edge_dim=num_edge_feat))
            else:
                self.convs.append(GINConv(nn=mlp))
        self.mlp = nn.Sequential(
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, 1)  # 回归输出
        )

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        for i in range(self.num_layers):
            conv = self.convs[i]
            if isinstance(conv, GINEConv):
                x = conv(x, edge_index, edge_attr)
            else:
                x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_add_pool(x, batch)
        x = self.mlp(x)
        return x.squeeze()