import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class MLLLoss(nn.Module):
	def __init__(self):
		super(MLLLoss, self).__init__()

	def forward(self, mu, sigma, target):
		return 0.5 * ( torch.log(torch.pow(sigma, 2)) + torch.pow((mu - target), 2) / torch.pow(sigma, 2))

class ContrastiveLoss(nn.Module):
    """
    基于基数的对比学习损失。
    目标：让基数相近的查询图在嵌入空间中也相互靠近。
    """
    def __init__(self, temperature=0.1, cardinality_threshold=1.0):
        """
        Args:
            temperature (float): 温度参数，控制相似度分布的锐利程度。
            cardinality_threshold (float): 定义“基数相近”的阈值（在log尺度上）。
        """
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.cardinality_threshold = cardinality_threshold

    def forward(self, embeddings, cardinalities):
        """
        Args:
            embeddings (Tensor): [batch_size, embedding_dim] - 图嵌入向量。
            cardinalities (Tensor): [batch_size] - 真实的基数 (log scale)。
        """
        batch_size = embeddings.size(0)
        if batch_size <= 1:
            return torch.tensor(0.0, device=embeddings.device)
        
        # 检查输入是否包含NaN或inf
        if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
            return torch.tensor(0.0, device=embeddings.device)
        
        if torch.isnan(cardinalities).any() or torch.isinf(cardinalities).any():
            return torch.tensor(0.0, device=embeddings.device)

        # 归一化嵌入向量
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        
        # 检查归一化后是否包含NaN
        if torch.isnan(embeddings_norm).any():
            return torch.tensor(0.0, device=embeddings.device)
        
        # 计算嵌入向量之间的余弦相似度矩阵
        embedding_sim = torch.mm(embeddings_norm, embeddings_norm.t()) / self.temperature
        
        # 数值稳定性：限制相似度的范围
        embedding_sim = torch.clamp(embedding_sim, -5.0, 5.0)

        # 计算基数差异矩阵
        card_diff = torch.abs(cardinalities.unsqueeze(1) - cardinalities.unsqueeze(0))

        # 创建正样本和负样本的掩码 (mask)
        # 正样本：基数差异小于阈值的查询对 (不包括自身)
        positive_mask = (card_diff < self.cardinality_threshold) & ~torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)
        
        # 防止没有正样本导致NaN
        if positive_mask.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device)

        # 计算对比损失 (InfoNCE Loss 的简化形式)
        # 对于每个样本，我们希望它与正样本的相似度更高
        
        # 将相似度矩阵按行应用softmax
        exp_sim = torch.exp(embedding_sim)
        
        # 添加数值稳定性检查
        exp_sim = torch.clamp(exp_sim, min=1e-12, max=1e12)
        
        # 计算每个样本与其所有其他样本的相似度总和
        sum_exp_sim = exp_sim.sum(1, keepdim=True)
        # 防止除零错误
        sum_exp_sim = torch.clamp(sum_exp_sim, min=1e-12)
        
        log_prob = embedding_sim - torch.log(sum_exp_sim)

        # 计算正样本对的平均log-likelihood
        # 我们希望最大化这个值，所以损失是它的负数
        mean_log_prob_pos = (positive_mask * log_prob).sum(1) / (positive_mask.sum(1) + 1e-8)

        loss = -mean_log_prob_pos.mean()
        
        # 检查最终损失是否为NaN
        if torch.isnan(loss):
            return torch.tensor(0.0, device=embeddings.device)
        
        return loss

