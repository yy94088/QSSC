import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class MLLLoss(nn.Module):
	def __init__(self):
		super(MLLLoss, self).__init__()

	def forward(self, mu, sigma, target):
		return 0.5 * ( torch.log(torch.pow(sigma, 2)) + torch.pow((mu - target), 2) / torch.pow(sigma, 2))

class QErrorLoss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(QErrorLoss, self).__init__()
        self.epsilon = epsilon
    
    def forward(self, pred, target):
        
        pred_card = torch.exp(pred) + self.epsilon
        true_card = torch.exp(target) + self.epsilon

        q_error = torch.maximum(pred_card / true_card, true_card / pred_card)

        loss = torch.log(q_error).mean()
        
        return loss


class AdaptiveWeightedMSELoss(nn.Module):

    def __init__(self, weight_exp=1.0):
        super(AdaptiveWeightedMSELoss, self).__init__()
        self.weight_exp = weight_exp
    
    def forward(self, pred, target):

        mse = (pred - target) ** 2

        weights = 1.0 / (1.0 + torch.abs(target) ** self.weight_exp)

        weights = weights / weights.mean()

        weighted_mse = (mse * weights).mean()
        
        return weighted_mse


class DistillationLoss(nn.Module):

    def __init__(
        self,
        alpha=0.5,
        temperature=2.0,
        confidence_threshold=1.0,
        min_quality=0.0,
        hard_only_quality_threshold=0.2,
    ):

        super(DistillationLoss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.confidence_threshold = confidence_threshold
        self.min_quality = min_quality
        self.hard_only_quality_threshold = hard_only_quality_threshold
        self.epsilon = 1e-8
    
    def forward(self, student_pred, soft_label, hard_label, alpha_scale=1.0):

        student_pred = student_pred.view(-1)
        hard_label = hard_label.view(-1)
        hard_loss = F.mse_loss(student_pred, hard_label)

        if soft_label is None or torch.isnan(soft_label).any():
            return hard_loss, torch.ones_like(hard_label)

        soft_label = soft_label.view(-1)
        valid_mask = torch.isfinite(student_pred) & torch.isfinite(hard_label) & torch.isfinite(soft_label)
        if valid_mask.sum().item() == 0:
            return hard_loss, torch.zeros_like(hard_label)

        student_pred = student_pred[valid_mask]
        hard_label = hard_label[valid_mask]
        soft_label = soft_label[valid_mask]

        soft_label_error = torch.abs(soft_label - hard_label)

        quality_weights = torch.sigmoid((self.confidence_threshold - soft_label_error) / self.temperature)
        quality_weights = torch.clamp(quality_weights, min=self.min_quality, max=1.0)

        scaled_student = student_pred / self.temperature
        scaled_soft = soft_label / self.temperature
        soft_mse = (scaled_student - scaled_soft) ** 2
        weight_sum = torch.clamp(quality_weights.sum(), min=self.epsilon)
        soft_loss = (soft_mse * quality_weights).sum() / weight_sum

        avg_quality = quality_weights.mean()
        adaptive_alpha = self.alpha * float(alpha_scale) * avg_quality
        if avg_quality.item() < self.hard_only_quality_threshold:
            adaptive_alpha = adaptive_alpha * 0.0

        total_loss = (1 - adaptive_alpha) * hard_loss + adaptive_alpha * soft_loss * (self.temperature ** 2)
        
        return total_loss, quality_weights


class ContrastiveLoss(nn.Module):

    def __init__(self, temperature=0.1, cardinality_threshold=1.0):

        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.cardinality_threshold = cardinality_threshold

    def forward(self, embeddings, cardinalities):

        batch_size = embeddings.size(0)
        if batch_size <= 1:
            return torch.tensor(0.0, device=embeddings.device)

        if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
            return torch.tensor(0.0, device=embeddings.device)
        
        if torch.isnan(cardinalities).any() or torch.isinf(cardinalities).any():
            return torch.tensor(0.0, device=embeddings.device)

        embeddings_norm = F.normalize(embeddings, p=2, dim=1)

        if torch.isnan(embeddings_norm).any():
            return torch.tensor(0.0, device=embeddings.device)

        embedding_sim = torch.mm(embeddings_norm, embeddings_norm.t()) / self.temperature

        embedding_sim = torch.clamp(embedding_sim, -5.0, 5.0)

        card_diff = torch.abs(cardinalities.unsqueeze(1) - cardinalities.unsqueeze(0))

        positive_mask = (card_diff < self.cardinality_threshold) & ~torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)

        if positive_mask.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device)

        exp_sim = torch.exp(embedding_sim)

        exp_sim = torch.clamp(exp_sim, min=1e-12, max=1e12)

        sum_exp_sim = exp_sim.sum(1, keepdim=True)

        sum_exp_sim = torch.clamp(sum_exp_sim, min=1e-12)
        
        log_prob = embedding_sim - torch.log(sum_exp_sim)

        mean_log_prob_pos = (positive_mask * log_prob).sum(1) / (positive_mask.sum(1) + 1e-8)

        loss = -mean_log_prob_pos.mean()

        if torch.isnan(loss):
            return torch.tensor(0.0, device=embeddings.device)
        
        return loss

