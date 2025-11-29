"""
数据增强策略用于查询图
通过对查询图进行小的扰动来增加训练样本的多样性
"""

import torch
import random
import copy


class QueryGraphAugmentation:
    """
    查询图数据增强
    策略：
    1. 特征噪声：在节点/边特征上添加小噪声
    2. Dropout边：随机丢弃一些边
    3. 特征Mask：随机mask部分节点特征
    """
    
    def __init__(self, 
                 feature_noise_std=0.05,
                 edge_dropout_rate=0.1,
                 feature_mask_rate=0.1):
        self.feature_noise_std = feature_noise_std
        self.edge_dropout_rate = edge_dropout_rate
        self.feature_mask_rate = feature_mask_rate
    
    def augment_features(self, node_features, edge_features):
        """
        对节点和边特征添加增强
        
        Args:
            node_features: list of tensors
            edge_features: list of tensors
        
        Returns:
            augmented_node_features, augmented_edge_features
        """
        aug_node_features = []
        aug_edge_features = []
        
        for node_feat, edge_feat in zip(node_features, edge_features):
            # 1. 添加高斯噪声
            if self.feature_noise_std > 0 and random.random() > 0.3:
                node_noise = torch.randn_like(node_feat) * self.feature_noise_std
                node_feat = node_feat + node_noise
                
                edge_noise = torch.randn_like(edge_feat) * self.feature_noise_std
                edge_feat = edge_feat + edge_noise
            
            # 2. 特征Mask（随机将某些维度置0）
            if self.feature_mask_rate > 0 and random.random() > 0.5:
                node_mask = torch.rand(node_feat.size()) > self.feature_mask_rate
                node_feat = node_feat * node_mask.float().to(node_feat.device)
            
            aug_node_features.append(node_feat)
            aug_edge_features.append(edge_feat)
        
        return aug_node_features, aug_edge_features
    
    def augment_edge_dropout(self, edge_index_list, edge_attr_list):
        """
        随机丢弃一些边（保持图连通性）
        
        Args:
            edge_index_list: list of edge indices
            edge_attr_list: list of edge attributes
        
        Returns:
            augmented_edge_index_list, augmented_edge_attr_list
        """
        aug_edge_index_list = []
        aug_edge_attr_list = []
        
        for edge_index, edge_attr in zip(edge_index_list, edge_attr_list):
            if self.edge_dropout_rate > 0 and random.random() > 0.5:
                # 计算要保留的边数
                num_edges = edge_index.size(1)
                num_keep = max(1, int(num_edges * (1 - self.edge_dropout_rate)))
                
                # 随机选择要保留的边
                keep_indices = torch.randperm(num_edges)[:num_keep]
                keep_indices = keep_indices.sort()[0]  # 保持顺序
                
                edge_index = edge_index[:, keep_indices]
                edge_attr = edge_attr[keep_indices]
            
            aug_edge_index_list.append(edge_index)
            aug_edge_attr_list.append(edge_attr)
        
        return aug_edge_index_list, aug_edge_attr_list
    
    def __call__(self, decomp_x, decomp_edge_index, decomp_edge_attr, training=True):
        """
        应用数据增强
        
        Args:
            decomp_x: list of node features
            decomp_edge_index: list of edge indices
            decomp_edge_attr: list of edge attributes
            training: 是否在训练模式（测试时不增强）
        
        Returns:
            augmented data
        """
        if not training:
            return decomp_x, decomp_edge_index, decomp_edge_attr
        
        # 特征增强
        aug_x, aug_edge_attr = self.augment_features(decomp_x, decomp_edge_attr)
        
        # 边dropout（可选，根据概率决定是否应用）
        if random.random() > 0.7:  # 30%概率应用边dropout
            aug_edge_index, aug_edge_attr = self.augment_edge_dropout(
                decomp_edge_index, aug_edge_attr
            )
        else:
            aug_edge_index = decomp_edge_index
        
        return aug_x, aug_edge_index, aug_edge_attr


class MixupAugmentation:
    """
    Mixup数据增强：混合两个查询的特征和标签
    适用于回归任务
    """
    
    def __init__(self, alpha=0.2):
        """
        Args:
            alpha: Beta分布的参数，控制混合比例
        """
        self.alpha = alpha
    
    def __call__(self, x1, y1, x2, y2):
        """
        混合两个样本
        
        Args:
            x1, x2: 特征
            y1, y2: 标签（基数）
        
        Returns:
            mixed_x, mixed_y, lambda_
        """
        if self.alpha > 0:
            lambda_ = random.betavariate(self.alpha, self.alpha)
        else:
            lambda_ = 1.0
        
        # 混合特征
        mixed_x = lambda_ * x1 + (1 - lambda_) * x2
        
        # 混合标签（在log空间）
        mixed_y = lambda_ * y1 + (1 - lambda_) * y2
        
        return mixed_x, mixed_y, lambda_


def apply_augmentation_to_batch(decomp_x, decomp_edge_index, decomp_edge_attr, 
                                augmentor, training=True):
    """
    对批次数据应用增强
    
    Args:
        decomp_x: batch中的节点特征列表
        decomp_edge_index: batch中的边索引列表
        decomp_edge_attr: batch中的边特征列表
        augmentor: QueryGraphAugmentation实例
        training: 是否训练模式
    
    Returns:
        增强后的数据
    """
    if not training or augmentor is None:
        return decomp_x, decomp_edge_index, decomp_edge_attr
    
    return augmentor(decomp_x, decomp_edge_index, decomp_edge_attr, training=training)
