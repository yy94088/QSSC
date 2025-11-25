import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv, GCNConv
import networkx as nx
from typing import List, Tuple
import queue


class MultiLayerGNN(nn.Module):
    """
    基于SubMB思想的多层图神经网络模块
    该模块将查询图分解为多个层次的子图结构，并使用图注意力网络处理这些多层结构
    """
    def __init__(self, args, num_node_feat, num_edge_feat):
        super(MultiLayerGNN, self).__init__()
        self.num_node_feat = num_node_feat
        self.num_edge_feat = num_edge_feat
        self.num_layers = args.num_layers
        self.num_hid = args.num_g_hid
        self.num_e_hid = args.num_e_hid
        self.num_out = args.out_g_ch
        self.model_type = args.model_type
        self.dropout = args.dropout
        self.k_hops = getattr(args, 'k_hops', [1, 2, 3])  # 默认k-hop值
        
        # 多层图注意力网络层
        self.intra_gnn_layers = nn.ModuleList()  # 处理同层内连接
        self.inter_gnn_layers = nn.ModuleList()  # 处理层间连接
        
        # 构建多层GNN
        for l in range(self.num_layers):
            hidden_input_dim = self.num_node_feat if l == 0 else self.num_hid
            hidden_output_dim = self.num_out if l == self.num_layers - 1 else self.num_hid
            
            # 同层内连接处理层
            self.intra_gnn_layers.append(self.build_gnn_layer(hidden_input_dim, hidden_output_dim))
            # 层间连接处理层
            self.inter_gnn_layers.append(self.build_gnn_layer(hidden_input_dim, hidden_output_dim))

        # 融合层
        self.fusion_layer = nn.Linear(hidden_output_dim * 2, hidden_output_dim)
        
    def build_gnn_layer(self, in_channels, out_channels):
        """
        构建GNN层
        """
        if self.model_type == "GAT":
            return GATConv(in_channels, out_channels, heads=4, dropout=self.dropout)
        elif self.model_type == "SAGE":
            return SAGEConv(in_channels, out_channels)
        elif self.model_type == "GCN":
            return GCNConv(in_channels, out_channels)
        else:
            # 默认使用GAT
            return GATConv(in_channels, out_channels, heads=4, dropout=self.dropout)
            
    def forward(self, x, intra_edge_index, inter_edge_index):
        """
        前向传播
        
        Args:
            x: 节点特征
            intra_edge_index: 同层内连接边索引
            inter_edge_index: 层间连接边索引
        """
        for i in range(self.num_layers):
            # 处理同层内连接
            h_intra = self.intra_gnn_layers[i](x, intra_edge_index)
            
            # 处理层间连接
            h_inter = self.inter_gnn_layers[i](x, inter_edge_index)
            
            # 融合同层和层间信息
            h_combined = torch.cat([h_intra, h_inter], dim=1)
            x = self.fusion_layer(h_combined)
            
            # 添加激活函数和dropout
            x = F.relu(x)
            if i < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
                
        # 聚合节点嵌入到图级表示
        x = torch.unsqueeze(torch.sum(x, dim=0), dim=0)
        return x


class MultiLayerGraphConstructor:
    """
    多层图结构构建器
    将原始查询图分解为多个k-hop子图，并构建多层图结构
    """
    
    def __init__(self, k_hops: List[int] = [1, 2, 3]):
        self.k_hops = k_hops
    
    def construct_multilayer_graph(self, query_graph: nx.Graph) -> Tuple[List[nx.Graph], List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        构建多层图结构
        
        Args:
            query_graph: 原始查询图
            
        Returns:
            subgraphs: k-hop子图列表
            intra_edges: 同层内连接边列表 [(layer_node_id, layer_node_id), ...]
            inter_edges: 层间连接边列表 [(layer_node_id, layer_node_id), ...]
        """
        subgraphs = []
        node_mappings = []  # 记录每个子图节点到原始图节点的映射
        
        # 生成k-hop子图
        for k in self.k_hops:
            k_subgraphs = []
            k_node_mappings = []
            
            for node in query_graph.nodes():
                k_hop_nodes = self._k_hop_nodes(query_graph, node, k)
                subgraph = query_graph.subgraph(k_hop_nodes).copy()
                
                # 重新编号节点以避免冲突
                mapping = {n: i for i, n in enumerate(subgraph.nodes())}
                subgraph = nx.relabel_nodes(subgraph, mapping)
                
                k_subgraphs.append(subgraph)
                k_node_mappings.append({v: k for k, v in mapping.items()})  # 反向映射
                
            subgraphs.extend(k_subgraphs)
            node_mappings.extend(k_node_mappings)
            
        # 构建同层内连接（同一k-hop子图内的边）
        intra_edges = self._build_intra_edges(subgraphs)
        
        # 构建层间连接（不同k-hop但包含相同原始节点的子图之间的连接）
        inter_edges = self._build_inter_edges(subgraphs, node_mappings)
        
        return subgraphs, intra_edges, inter_edges
    
    def _k_hop_nodes(self, graph: nx.Graph, source: int, k: int) -> List[int]:
        """
        获取k-hop邻居节点
        """
        visited = {source}
        frontier = {source}
        
        for _ in range(k):
            new_frontier = set()
            for node in frontier:
                for neighbor in graph.neighbors(node):
                    if neighbor not in visited:
                        new_frontier.add(neighbor)
                        visited.add(neighbor)
            frontier = new_frontier
            if not frontier:
                break
                
        return list(visited)
    
    def _build_intra_edges(self, subgraphs: List[nx.Graph]) -> List[Tuple[int, int]]:
        """
        构建同层内连接边
        """
        intra_edges = []
        node_offset = 0
        
        for subgraph in subgraphs:
            for edge in subgraph.edges():
                intra_edges.append((edge[0] + node_offset, edge[1] + node_offset))
            node_offset += subgraph.number_of_nodes()
            
        return intra_edges
    
    def _build_inter_edges(self, subgraphs: List[nx.Graph], node_mappings: List[dict]) -> List[Tuple[int, int]]:
        """
        构建层间连接边
        """
        inter_edges = []
        node_offset = 0
        subgraph_offsets = []
        
        # 计算每个子图的节点偏移量
        offset = 0
        for subgraph in subgraphs:
            subgraph_offsets.append(offset)
            offset += subgraph.number_of_nodes()
        
        # 查找包含相同原始节点的子图并建立连接
        for i in range(len(subgraphs)):
            for j in range(i + 1, len(subgraphs)):
                # 检查两个子图是否包含相同的原始节点
                mapping_i = node_mappings[i]
                mapping_j = node_mappings[j]
                
                common_nodes = set(mapping_i.values()) & set(mapping_j.values())
                if common_nodes:
                    # 为每个共同节点在两个子图之间建立连接
                    for node in common_nodes:
                        node_i = [k for k, v in mapping_i.items() if v == node][0]
                        node_j = [k for k, v in mapping_j.items() if v == node][0]
                        inter_edges.append(
                            (node_i + subgraph_offsets[i], 
                             node_j + subgraph_offsets[j])
                        )
        
        return inter_edges


# 使用示例
"""
# 在模型中使用多层GNN
class CardNetWithMultiLayer(nn.Module):
    def __init__(self, args, num_node_feat, num_edge_feat):
        super(CardNetWithMultiLayer, self).__init__()
        self.multilayer_gnn = MultiLayerGNN(args, num_node_feat, num_edge_feat)
        # ... 其他层 ...
        
    def forward(self, query_graph):
        # 构建多层图结构
        constructor = MultiLayerGraphConstructor(k_hops=[1, 2, 3])
        subgraphs, intra_edges, inter_edges = constructor.construct_multilayer_graph(query_graph)
        
        # 将图结构转换为PyG格式
        x, intra_edge_index, inter_edge_index = self._convert_to_pyg_format(
            subgraphs, intra_edges, inter_edges)
        
        # 使用多层GNN处理
        graph_embedding = self.multilayer_gnn(x, intra_edge_index, inter_edge_index)
        
        # ... 后续处理 ...
        return graph_embedding
"""