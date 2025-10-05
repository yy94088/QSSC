import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GINEConv, global_add_pool, global_mean_pool
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from collections import defaultdict
from annoy import AnnoyIndex  # 需要安装annoy

class QueryEncoder(nn.Module):
    """将query图编码为固定维度的向量表示"""
    def __init__(self, num_node_feat, num_edge_feat, hidden_dim=128, num_layers=3):
        super(QueryEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.convs = nn.ModuleList()
        self.use_edge_attr = num_edge_feat is not None and int(num_edge_feat) > 0
        
        for i in range(num_layers):
            in_dim = num_node_feat if i == 0 else hidden_dim
            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            if self.use_edge_attr:
                self.convs.append(GINEConv(nn=mlp, edge_dim=num_edge_feat))
            else:
                self.convs.append(GINConv(nn=mlp))
        
        # 多级池化，获得更丰富的图表示
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # 2倍因为用了mean和add pooling
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        for conv in self.convs:
            if isinstance(conv, GINEConv):
                x = conv(x, edge_index, edge_attr)
            else:
                x = conv(x, edge_index)
            x = F.relu(x)
        
        # 使用多种池化方式获得更丰富的表示
        x_mean = global_mean_pool(x, batch)
        x_add = global_add_pool(x, batch)
        x_combined = torch.cat([x_mean, x_add], dim=-1)
        
        return self.projection(x_combined)

class SimilarityComputer(nn.Module):
    """计算query之间的多层次相似性，使用Annoy近似"""
    def __init__(self, embedding_dim=128):
        super(SimilarityComputer, self).__init__()
        self.embedding_dim = embedding_dim
        
        # 学习不同相似性维度的权重
        self.similarity_weights = nn.Parameter(torch.ones(4) / 4)  # 4种相似性
        
        # 相似性变换矩阵
        self.structural_transform = nn.Linear(embedding_dim, embedding_dim)
        self.semantic_transform = nn.Linear(embedding_dim, embedding_dim)
        self.statistical_transform = nn.Linear(embedding_dim, embedding_dim)
        self.content_transform = nn.Linear(embedding_dim, embedding_dim)
    
    def forward(self, query_embeddings):
        """
        计算query之间的相似性矩阵，使用Annoy (ANN) 近似
        Args:
            query_embeddings: [num_queries, embedding_dim]
        Returns:
            similarity_matrix: [num_queries, num_queries] (近似，稀疏填充)
        """
        # 确保输入是2D的
        if query_embeddings.dim() != 2:
            raise ValueError(f"Expected 2D tensor, got {query_embeddings.dim()}D tensor with shape {query_embeddings.shape}")
        
        batch_size = query_embeddings.size(0)
        
        # 计算加权嵌入 (融合多层次)
        weights = F.softmax(self.similarity_weights, dim=0)
        structural_emb = self.structural_transform(query_embeddings)
        semantic_emb = self.semantic_transform(query_embeddings)
        statistical_emb = self.statistical_transform(query_embeddings)
        content_emb = self.content_transform(query_embeddings)
        
        weighted_emb = (weights[0] * structural_emb + 
                        weights[1] * semantic_emb + 
                        weights[2] * statistical_emb + 
                        weights[3] * content_emb)
        
        # 归一化以用于cosine相似
        weighted_emb_norm = F.normalize(weighted_emb, p=2, dim=1)
        weighted_emb_np = weighted_emb_norm.cpu().detach().numpy().astype('float32')
        
        # 构建Annoy索引
        dim = self.embedding_dim
        index = AnnoyIndex(dim, 'angular')  # 'angular' for cosine similarity
        is_built = False  # 标志变量，跟踪索引是否已构建

        for i in range(batch_size):
            index.add_item(i, weighted_emb_np[i])
            if not is_built:  # 仅在索引未构建时调用 build
                index.build(50)  # 50 trees, 提高近邻查找精度
                is_built = True

        # 查询k最近邻；k=min(batch_size, 50) 以平衡精度和速度
        k = min(batch_size, 50)  # 调整k以控制近似程度；更大k更精确但更慢
        similarity_matrix = torch.zeros(batch_size, batch_size, device=query_embeddings.device)

        for i in range(batch_size):
            # 查询top-k邻居（包括自己）
            indices, distances = index.get_nns_by_item(i, k, include_distances=True)
            for j, dist in zip(indices[1:], distances[1:]):  # 跳过自己 (indices[0] == i)
                if j < batch_size:  # 确保索引有效
                    # Annoy的angular距离转换为cosine相似度：cosine_sim = 1 - dist^2/2
                    sim = 1 - (dist * dist / 2)
                    similarity_matrix[i, j] = sim
                    similarity_matrix[j, i] = sim  # 使对称
        
        # 对角线设为1 (self-similarity)
        similarity_matrix.fill_diagonal_(1.0)
        
        # 可选: 应用sigmoid如果需要，但cosine已合适
        # similarity_matrix = F.sigmoid(similarity_matrix)
        
        return similarity_matrix

class AdaptiveQuerySelector(nn.Module):
    """自适应选择最相关的相似query"""
    def __init__(self, embedding_dim=128, max_neighbors=10):
        super(AdaptiveQuerySelector, self).__init__()
        self.max_neighbors = max_neighbors
        
        # 学习注意力权重的网络
        self.attention_net = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),  # query pair embedding
            nn.ReLU(),
            nn.Linear(embedding_dim, 1)
        )
        
        # 动态调整邻居数量的网络
        self.neighbor_count_net = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, target_embedding, all_embeddings, similarity_matrix, target_idx):
        """
        为target query选择最相关的邻居
        Args:
            target_embedding: [embedding_dim] - 目标query的embedding
            all_embeddings: [num_queries, embedding_dim] - 所有query的embeddings
            similarity_matrix: [num_queries, num_queries] - 相似性矩阵
            target_idx: int - 目标query的索引
        """
        # 动态决定邻居数量
        dynamic_k = int(self.neighbor_count_net(target_embedding).item() * self.max_neighbors)
        dynamic_k = max(1, min(dynamic_k, self.max_neighbors))
        
        # 获取最相似的k个query (使用ANN近似的similarity_matrix)
        target_similarities = similarity_matrix[target_idx]
        _, top_indices = torch.topk(target_similarities, k=min(dynamic_k + 1, similarity_matrix.size(0)))
        
        # 移除自己
        neighbor_indices = top_indices[top_indices != target_idx][:dynamic_k]
        
        if len(neighbor_indices) == 0:
            neighbor_indices = torch.tensor([], device=all_embeddings.device, dtype=torch.long)
            print(f"[AdaptiveQuerySelector] No neighbors found for query idx {target_idx}")
            return torch.zeros_like(target_embedding), torch.zeros(1, device=target_embedding.device), neighbor_indices, torch.zeros(1, device=target_embedding.device)
        
        # 计算注意力权重
        neighbor_embeddings = all_embeddings[neighbor_indices]
        target_expanded = target_embedding.unsqueeze(0).expand(len(neighbor_indices), -1)
        pair_embeddings = torch.cat([target_expanded, neighbor_embeddings], dim=-1)
        
        attention_scores = self.attention_net(pair_embeddings).squeeze(-1)
        attention_weights = F.softmax(attention_scores, dim=0)
        
        # 加权聚合邻居信息
        aggregated_neighbor = torch.sum(attention_weights.unsqueeze(-1) * neighbor_embeddings, dim=0)
        
        return aggregated_neighbor, attention_weights, neighbor_indices, attention_weights

class QueryMemoryBank(nn.Module):
    """可学习的query记忆库"""
    def __init__(self, embedding_dim=128, memory_size=100, temperature=0.1):
        super(QueryMemoryBank, self).__init__()
        self.embedding_dim = embedding_dim
        self.memory_size = memory_size
        self.temperature = temperature
        
        # 可学习的记忆embeddings
        self.memory_embeddings = nn.Parameter(torch.randn(memory_size, embedding_dim))
        self.memory_cardinalities = nn.Parameter(torch.randn(memory_size))
        
        # 记忆更新网络
        self.update_net = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
    
    def forward(self, query_embedding):
        """
        从记忆库中检索相关信息
        """
        # 计算与记忆的相似度
        similarities = torch.mm(query_embedding.unsqueeze(0), self.memory_embeddings.t()).squeeze(0)
        similarities = similarities / self.temperature
        
        # 软检索 - 加权聚合记忆
        attention_weights = F.softmax(similarities, dim=0)
        retrieved_embedding = torch.sum(attention_weights.unsqueeze(-1) * self.memory_embeddings, dim=0)
        retrieved_cardinality = torch.sum(attention_weights * self.memory_cardinalities)
        
        # 更新当前embedding
        combined_input = torch.cat([query_embedding, retrieved_embedding])
        updated_embedding = self.update_net(combined_input)
        
        return updated_embedding, retrieved_cardinality, attention_weights
    
    def update_memory(self, query_embeddings, cardinalities, learning_rate=0.01):
        """更新记忆库"""
        with torch.no_grad():
            for query_emb, card in zip(query_embeddings, cardinalities):
                # 分离梯度以避免inplace操作问题
                query_emb = query_emb.detach()
                card = card.detach()
                
                # 找到最相似的记忆位置
                similarities = torch.mm(query_emb.unsqueeze(0), self.memory_embeddings.t()).squeeze(0)
                best_idx = torch.argmax(similarities)
                
                # 使用copy_更新，避免直接赋值
                self.memory_embeddings[best_idx].copy_(
                    (1 - learning_rate) * self.memory_embeddings[best_idx] + learning_rate * query_emb
                )
                self.memory_cardinalities[best_idx].copy_(
                    (1 - learning_rate) * self.memory_cardinalities[best_idx] + learning_rate * card
                )

class QSimNet(nn.Module):
    """完整的Query Similarity Network"""
    def __init__(self, num_node_feat, num_edge_feat, hidden_dim=128, memory_size=500):
        super(QSimNet, self).__init__()
        
        # 核心组件
        self.query_encoder = QueryEncoder(num_node_feat, num_edge_feat, hidden_dim)
        self.similarity_computer = SimilarityComputer(hidden_dim)
        self.adaptive_selector = AdaptiveQuerySelector(hidden_dim)
        self.memory_bank = QueryMemoryBank(hidden_dim, memory_size)
        
        # 最终预测网络
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),  # target + neighbor + memory
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 对比学习的投影头
        self.contrastive_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
    
    def forward(self, batch_data, training=True):
        """
        Args:
            batch_data: list of (x, edge_index, edge_attr, cardinality) tuples
        """
        batch_size = len(batch_data)
        
        # 1. 编码所有query
        query_embeddings = []
        true_cardinalities = []
        
        for x, edge_index, edge_attr, card in batch_data:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            query_emb = self.query_encoder(x, edge_index, edge_attr, batch)
            # 确保embedding是1D的
            if query_emb.dim() > 1:
                query_emb = query_emb.squeeze()
            query_embeddings.append(query_emb)
            true_cardinalities.append(card)
        
        query_embeddings = torch.stack(query_embeddings)
        true_cardinalities = torch.stack(true_cardinalities)
        
        # 2. 计算相似性矩阵 (现在用Annoy近似)
        similarity_matrix = self.similarity_computer(query_embeddings)
        
        # 3. 为每个query预测cardinality
        predictions = []
        contrastive_embeddings = []
        all_debug_info = []
        
        for i in range(batch_size):
            target_emb = query_embeddings[i]
            
            # 自适应选择相似邻居
            neighbor_emb, attention_weights, neighbor_indices, attention_weights_for_debug = self.adaptive_selector(
                target_emb, query_embeddings, similarity_matrix, i
            )
            
            # 从记忆库检索
            memory_emb, memory_card, memory_weights = self.memory_bank(target_emb)
            
            # 融合信息进行预测
            combined_features = torch.cat([target_emb, neighbor_emb, memory_emb])
            prediction = self.predictor(combined_features)
            predictions.append(prediction)
            
            # 对比学习的embedding
            contrastive_emb = self.contrastive_head(target_emb)
            contrastive_embeddings.append(contrastive_emb)

            # 确保neighbor_indices有效
            if 'neighbor_indices' not in locals() or neighbor_indices is None:
                neighbor_indices = torch.tensor([], device=query_embeddings.device, dtype=torch.long)

            # 确保target_idx有效
            if 'target_idx' not in locals():
                target_idx = -1  # 默认值，表示无效索引

            # 确保target_embedding有效
            if 'target_embedding' not in locals():
                target_embedding = torch.zeros(1, device=query_embeddings.device)  # 默认值，表示无效embedding

            # 收集调试信息
            all_debug_info.append({
                'neighbor_indices': neighbor_indices.tolist() if neighbor_indices.numel() > 0 else [],
                'attention_weights': attention_weights_for_debug.tolist() if attention_weights_for_debug.numel() > 1 else [],
                'memory_weights': memory_weights.tolist(),
                'retrieved_cardinality': memory_card.item()
            })
        
        predictions = torch.stack(predictions).squeeze(-1)
        contrastive_embeddings = torch.stack(contrastive_embeddings)
        
        # 4. 训练期间不进行记忆库的inplace更新，避免破坏autograd计算图
        # if training:
        #     self.memory_bank.update_memory(query_embeddings.detach(), true_cardinalities.detach())
        
        return {
            'predictions': predictions,
            'similarity_matrix': similarity_matrix,
            'contrastive_embeddings': contrastive_embeddings,
            'query_embeddings': query_embeddings,
            'debug_info': {
                'neighbor_indices': [info['neighbor_indices'] for info in all_debug_info],
                'attention_weights': [info['attention_weights'] for info in all_debug_info],
                'memory_weights': [info['memory_weights'] for info in all_debug_info],
                'retrieved_cardinality': [info['retrieved_cardinality'] for info in all_debug_info]
            }
        }

class ContrastiveLoss(nn.Module):
    """基于相似性的对比学习损失"""
    def __init__(self, temperature=0.5, cardinality_threshold=0.2):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.cardinality_threshold = cardinality_threshold
    
    def forward(self, embeddings, cardinalities, similarity_matrix):
        """
        Args:
            embeddings: [batch_size, embedding_dim] - 对比学习的embeddings
            cardinalities: [batch_size] - 真实cardinality (log scale)
            similarity_matrix: [batch_size, batch_size] - query结构相似性
        """
        batch_size = embeddings.size(0)
        
        # 构造正负样本
        # 正样本：cardinality相近且结构相似的query pair
        card_diff = torch.abs(cardinalities.unsqueeze(1) - cardinalities.unsqueeze(0))
        structure_sim = similarity_matrix
        
        positive_mask = (card_diff < self.cardinality_threshold) & (structure_sim > 0.5)
        negative_mask = (card_diff > self.cardinality_threshold * 2) | (structure_sim < 0.3)
        
        # 计算embedding相似度
        embeddings_norm = F.normalize(embeddings, dim=1)
        embedding_sim = torch.mm(embeddings_norm, embeddings_norm.t()) / self.temperature
        
        # 对比损失
        positive_loss = -torch.log(F.sigmoid(embedding_sim) + 1e-8) * positive_mask.float()
        negative_loss = -torch.log(F.sigmoid(-embedding_sim) + 1e-8) * negative_mask.float()
        
        total_loss = (positive_loss.sum() + negative_loss.sum()) / (positive_mask.sum() + negative_mask.sum() + 1e-8)
        
        return total_loss

def compute_graph_statistics(x, edge_index, edge_attr=None):
    """计算图的统计特征，用于增强相似性计算"""
    num_nodes = x.size(0)
    num_edges = edge_index.size(1)
    
    # 基本统计
    node_degrees = torch.zeros(num_nodes)
    for i in range(num_edges):
        src, dst = edge_index[0, i], edge_index[1, i]
        node_degrees[src] += 1
        node_degrees[dst] += 1
    
    stats = {
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'avg_degree': node_degrees.mean(),
        'max_degree': node_degrees.max(),
        'density': 2 * num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
    }
    
    return stats