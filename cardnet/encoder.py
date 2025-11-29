import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GINConv, GINEConv, NNConv, GATConv, GraphConv, SAGEConv
from .GINlayers import NNGINConv, NNGINConcatConv


class DecomGNN(nn.Module):
    def __init__(self, args, num_node_feat, num_edge_feat):
        super(DecomGNN, self).__init__()
        self.num_node_feat = num_node_feat
        self.num_edge_feat = num_edge_feat
        self.num_layers = args.num_layers
        self.num_hid = args.num_g_hid
        self.num_e_hid = args.num_e_hid
        self.num_out = args.out_g_ch
        self.model_type = args.model_type
        self.dropout = args.dropout
        self.convs = nn.ModuleList()
        
        # 添加层归一化以稳定训练
        self.layer_norms = nn.ModuleList()
        
        # 添加残差投影层（当维度不匹配时）
        self.residual_projections = nn.ModuleList()

        cov_layer = self.build_cov_layer(self.model_type)

        for l in range(self.num_layers):
            hidden_input_dim = self.num_node_feat if l == 0 else self.num_hid
            hidden_output_dim = self.num_out if l == self.num_layers - 1 else self.num_hid

            if self.model_type in ("GIN", "GINE", "GAT", "GCN", "SAGE"):
                self.convs.append(cov_layer(hidden_input_dim, hidden_output_dim))
            elif self.model_type in ("NN", "NNGIN", "NNGINConcat"):
                self.convs.append(cov_layer(hidden_input_dim, hidden_output_dim, self.num_e_hid))
            elif self.model_type == "NNGINETransformer":
                self.convs.append(cov_layer(hidden_input_dim, hidden_output_dim, self.num_edge_feat))
            else:
                raise ValueError("Unsupported model type: %s" % self.model_type)
            
            # 为每层添加LayerNorm
            self.layer_norms.append(nn.LayerNorm(hidden_output_dim))
            
            # 如果输入输出维度不同，添加投影层用于残差连接
            if hidden_input_dim != hidden_output_dim:
                self.residual_projections.append(nn.Linear(hidden_input_dim, hidden_output_dim))
            else:
                self.residual_projections.append(None)


    def build_cov_layer(self, model_type):
        if model_type == "GIN":
            return lambda in_ch, hid_ch: GINConv(nn=nn.Sequential(nn.Linear(in_ch, hid_ch), nn.ReLU(), nn.Linear(hid_ch, hid_ch)))
        elif model_type == "GINE":
            return lambda in_ch, hid_ch: GINEConv(nn=nn.Sequential(nn.Linear(in_ch, hid_ch), nn.ReLU(), nn.Linear(hid_ch, hid_ch)))
        elif model_type == "NN":
            return lambda in_ch, hid_ch, e_hid_ch: NNConv(in_ch, hid_ch,
                nn=nn.Sequential(nn.Linear(self.num_edge_feat, e_hid_ch), nn.ReLU(), nn.Linear(e_hid_ch, in_ch * hid_ch)))
        elif model_type == "NNGIN":
            return lambda in_ch, hid_ch, e_hid_ch: NNGINConv(
                edge_nn=nn.Sequential(nn.Linear(self.num_edge_feat, e_hid_ch), nn.ReLU(), nn.Linear(e_hid_ch, in_ch)),
                node_nn=nn.Sequential(nn.Linear(in_ch, hid_ch), nn.ReLU(), nn.Linear(hid_ch, hid_ch)))
        elif model_type == "NNGINConcat":
            return lambda in_ch, hid_ch, e_hid_ch: NNGINConcatConv(
                edge_nn=nn.Sequential(nn.Linear(self.num_edge_feat, e_hid_ch), nn.ReLU(), nn.Linear(e_hid_ch, in_ch)),
                node_nn=nn.Sequential(nn.Linear(in_ch * 2, hid_ch), nn.ReLU(), nn.Linear(hid_ch, hid_ch)))
        elif model_type == "NNGINETransformer":
            from .GINlayers import NNGINETransformerConv
            return lambda in_ch, hid_ch, edge_dim: NNGINETransformerConv(in_channels=in_ch, out_channels=hid_ch, edge_dim=edge_dim)
        elif model_type == "GAT":
            return GATConv
        elif model_type == "SAGE":
            return SAGEConv
        elif model_type == "GCN":
            return GraphConv
        else:
            raise ValueError("Unsupported model type: %s" % model_type)


    def forward(self, x, edge_index, edge_attr=None):
        if torch.isnan(x).any() or torch.isnan(edge_index).any():
            return torch.zeros(1, self.num_out, device=x.device)

        if edge_attr is not None and torch.isnan(edge_attr).any():
            return torch.zeros(1, self.num_out, device=x.device)

        for i in range(self.num_layers):
            # 保存输入用于残差连接
            identity = x
            
            try:
                # GNN卷积
                if self.model_type in ("GIN", "GINE", "GAT", "GCN", "SAGE"):
                    x = self.convs[i](x, edge_index)
                elif self.model_type in ("NN", "NNGIN", "NNGINConcat"):
                    x = self.convs[i](x, edge_index, edge_attr)
                elif self.model_type == "NNGINETransformer":
                    x = self.convs[i](x, edge_index, edge_attr)
                else:
                    raise ValueError("Unsupported model type: %s" % self.model_type)
            except Exception:
                return torch.zeros(1, self.num_out, device=x.device)

            # 残差连接（跳过最后一层，因为可能需要特定的输出维度）
            if i < self.num_layers - 1:
                # 如果维度不匹配，使用投影
                if self.residual_projections[i] is not None:
                    identity = self.residual_projections[i](identity)
                
                # 残差连接: x = x + identity
                x = x + identity
                
                # 层归一化
                x = self.layer_norms[i](x)
                
                # Dropout
                x = F.dropout(x, p=self.dropout, training=self.training)
                
                if torch.isnan(x).any() or torch.isinf(x).any():
                    return torch.zeros(1, self.num_out, device=x.device)
            else:
                # 最后一层也应用LayerNorm
                x = self.layer_norms[i](x)

        x = torch.unsqueeze(torch.sum(x, dim=0), dim=0)
        if torch.isnan(x).any() or torch.isinf(x).any():
            return torch.zeros(1, self.num_out, device=x.device)
        return x


class Attention(nn.Module):
    def __init__(self, n_expert, n_hidden, v_hidden):
        super(Attention, self).__init__()
        self.n_expert = n_expert
        self.n_hidden = n_hidden
        self.v_hidden = v_hidden
        self.w1 = nn.Parameter(torch.FloatTensor(self.n_hidden, self.v_hidden))
        self.w2 = nn.Parameter(torch.FloatTensor(self.n_expert, self.n_hidden))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.w1.size(1))
        self.w1.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.w2.size(1))
        self.w2.data.uniform_(-stdv, stdv)

    def forward(self, x):
        x = torch.transpose(x, 0, 1)
        if self.v_hidden != x.size(0):
            patch_num = x.size(1)
            return torch.ones(self.n_expert, patch_num, device=x.device) / patch_num

        support = F.tanh(self.w1.matmul(x))
        if torch.isnan(support).any() or torch.isinf(support).any():
            patch_num = x.size(1)
            return torch.ones(self.n_expert, patch_num, device=x.device) / patch_num

        output = F.softmax(self.w2.matmul(support), dim=1)
        if torch.isnan(output).any() or torch.isinf(output).any():
            patch_num = x.size(1)
            return torch.ones(self.n_expert, patch_num, device=x.device) / patch_num
        return output


class CardEncoder(nn.Module):
    """Encodes a decomposed graph (list of node/edge tensors) into a single feature vector."""
    def __init__(self, args, num_node_feat, num_edge_feat):
        super(CardEncoder, self).__init__()
        self.num_node_feat = num_node_feat
        self.num_edge_feat = num_edge_feat
        self.num_expert = args.num_expert
        self.out_g_ch = args.out_g_ch
        self.num_att_hid = args.num_att_hid
        self.pool_type = args.pool_type

        self.graph2vec = DecomGNN(args, self.num_node_feat, self.num_edge_feat)
        self.att_layer = Attention(self.num_expert, args.num_att_hid, self.out_g_ch)
        self.mlp_in_ch = self.num_expert * self.out_g_ch if self.pool_type == "att" else self.out_g_ch

    def forward(self, decomp_x, decomp_edge_index, decomp_edge_attr):
        g = None
        for x, edge_index, edge_attr in zip(decomp_x, decomp_edge_index, decomp_edge_attr):
            x, edge_index, edge_attr = x.squeeze(), edge_index.squeeze(), edge_attr.squeeze()
            if torch.isnan(x).any() or torch.isnan(edge_attr).any():
                if g is None:
                    g = torch.zeros(1, self.out_g_ch, device=x.device)
                else:
                    g = torch.cat([g, torch.zeros(1, self.out_g_ch, device=x.device)], dim=0)
                continue

            if g is None:
                g = self.graph2vec(x, edge_index, edge_attr)
            else:
                g = torch.cat([g, self.graph2vec(x, edge_index, edge_attr)], dim=0)

        if g is None or torch.isnan(g).any() or torch.isinf(g).any():
            x = torch.zeros(1, self.mlp_in_ch, device=next(self.parameters()).device)
        else:
            if self.pool_type == "sum":
                x = torch.sum(g, dim=0).unsqueeze(dim=0)
            elif self.pool_type == "mean":
                x = torch.mean(g, dim=0).unsqueeze(dim=0)
            elif self.pool_type == "max":
                x, _ = torch.max(g, dim=0, keepdim=True)
            else:
                att_wights = self.att_layer(g)
                g = att_wights.matmul(g)
                x = g.view((1, self.num_expert * self.out_g_ch))

        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.zeros_like(x)

        return x
