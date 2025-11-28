"""
Improved Query Memory Bank for Cardinality Estimation

Key Improvements:
1. Quality-aware memory management: Track prediction accuracy for each memory entry
2. Adaptive similarity threshold: Dynamically adjust based on memory quality
3. Multi-level memory structure: Separate high-quality and exploratory memories
4. Uncertainty-aware retrieval: Consider prediction confidence in memory retrieval
5. Progressive memory warming: Start with empty memory and gradually fill with validated entries
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ImprovedQueryMemoryBank(nn.Module):
    """
    Enhanced memory bank with quality tracking and adaptive retrieval
    """
    def __init__(self, embedding_dim=128, memory_size=100, high_quality_ratio=0.7, 
                 temperature=0.1, base_similarity_threshold=0.85):
        super(ImprovedQueryMemoryBank, self).__init__()
        self.embedding_dim = embedding_dim
        self.memory_size = memory_size
        self.temperature = temperature
        self.base_similarity_threshold = base_similarity_threshold
        
        # Split memory into high-quality and exploratory zones
        self.hq_size = int(memory_size * high_quality_ratio)
        self.exp_size = memory_size - self.hq_size
        
        # Memory storage - initialized to zeros instead of random
        self.register_buffer('memory_embeddings', torch.zeros(memory_size, embedding_dim))
        self.register_buffer('memory_cardinalities', torch.zeros(memory_size))
        self.register_buffer('memory_used', torch.zeros(memory_size, dtype=torch.bool))
        
        # Quality tracking
        self.register_buffer('memory_quality_scores', torch.zeros(memory_size))  # 0-1, higher is better
        self.register_buffer('memory_usage_count', torch.zeros(memory_size, dtype=torch.long))
        self.register_buffer('memory_prediction_errors', torch.zeros(memory_size))  # Track average errors
        
        # Adaptive threshold
        self.register_buffer('current_threshold', torch.tensor(base_similarity_threshold))
        
        # Statistics for monitoring
        self.register_buffer('total_retrievals', torch.tensor(0, dtype=torch.long))
        self.register_buffer('successful_retrievals', torch.tensor(0, dtype=torch.long))

    def _compute_cosine_similarity(self, query_embedding, memory_embeddings):
        """Compute normalized cosine similarity"""
        # Normalize embeddings
        query_norm = F.normalize(query_embedding, p=2, dim=-1)
        memory_norm = F.normalize(memory_embeddings, p=2, dim=-1)
        
        # Compute cosine similarity
        similarities = torch.mm(query_norm, memory_norm.t())
        return similarities.squeeze()

    def _compute_adaptive_threshold(self):
        """Dynamically adjust similarity threshold based on memory quality"""
        if not self.memory_used.any():
            return self.base_similarity_threshold
        
        used_indices = torch.where(self.memory_used)[0]
        avg_quality = self.memory_quality_scores[used_indices].mean()
        
        # If memory quality is high, we can be more selective (higher threshold)
        # If memory quality is low, we should be less selective (lower threshold)
        if avg_quality > 0.7:
            threshold = min(self.base_similarity_threshold + 0.05, 0.95)
        elif avg_quality > 0.5:
            threshold = self.base_similarity_threshold
        else:
            threshold = max(self.base_similarity_threshold - 0.1, 0.7)
        
        # Use .data to avoid gradient tracking
        self.current_threshold.data.fill_(threshold)
        return threshold

    def forward(self, query_embedding, query_cardinality=None, return_confidence=False):
        """
        Retrieve relevant memories and compute aggregated information
        
        Args:
            query_embedding: Query embedding tensor [1, embedding_dim]
            query_cardinality: Optional true cardinality (for training)
            return_confidence: If True, also return confidence score
        
        Returns:
            retrieved_embedding: Aggregated memory embedding
            confidence: Optional confidence score
        """
        # If no memories yet, return zero vector
        if not self.memory_used.any():
            zero_embedding = torch.zeros_like(query_embedding)
            return (zero_embedding, 0.0) if return_confidence else zero_embedding
        
        # Get used memories
        used_indices = torch.where(self.memory_used)[0]
        used_embeddings = self.memory_embeddings[used_indices]
        used_cardinalities = self.memory_cardinalities[used_indices]
        used_qualities = self.memory_quality_scores[used_indices]
        
        # Compute similarities using cosine similarity
        similarities = self._compute_cosine_similarity(query_embedding, used_embeddings)
        
        # Ensure similarities is at least 1D
        if similarities.dim() == 0:
            similarities = similarities.unsqueeze(0)
        
        # Adaptive thresholding
        threshold = self._compute_adaptive_threshold()
        
        # Filter by similarity and quality
        similarity_mask = similarities >= threshold
        quality_mask = used_qualities > 0.3  # Only use reasonably good memories
        valid_mask = similarity_mask & quality_mask
        
        # Update statistics (use .data to avoid gradient tracking)
        self.total_retrievals.data.add_(1)
        
        if not valid_mask.any():
            # No good matches found - try relaxing constraints
            if similarity_mask.any():
                valid_mask = similarity_mask
            else:
                # Fall back to top-k most similar
                k = min(3, len(similarities))
                _, top_k_indices = torch.topk(similarities, k)
                valid_mask = torch.zeros_like(similarity_mask)
                valid_mask[top_k_indices] = True
        
        if valid_mask.any():
            # Update statistics (use .data to avoid gradient tracking)
            self.successful_retrievals.data.add_(1)
            
            valid_indices = torch.where(valid_mask)[0]
            valid_similarities = similarities[valid_indices]
            valid_embeddings = used_embeddings[valid_indices]
            valid_qualities = used_qualities[valid_indices]
            
            # Compute attention weights combining similarity and quality
            # Higher weight for both similar and high-quality memories
            combined_scores = valid_similarities * (0.7 + 0.3 * valid_qualities)
            attention_weights = F.softmax(combined_scores / self.temperature, dim=-1)
            
            # Aggregate embeddings
            if attention_weights.dim() == 0:
                attention_weights = attention_weights.unsqueeze(0)
            retrieved_embedding = torch.sum(
                attention_weights.unsqueeze(-1) * valid_embeddings, 
                dim=0, 
                keepdim=True
            )
            
            # Compute confidence based on similarity and quality
            confidence = (valid_similarities * valid_qualities).max().item()
            
            # Update usage counts (only in training, use .data to avoid gradient tracking)
            if self.training:
                for idx in valid_indices:
                    original_idx = used_indices[idx]
                    self.memory_usage_count.data[original_idx] += 1
        else:
            retrieved_embedding = torch.zeros_like(query_embedding)
            confidence = 0.0
        
        if return_confidence:
            return retrieved_embedding, confidence
        return retrieved_embedding

    def update_memory(self, query_embedding, query_cardinality, prediction_error=None):
        """
        Add or update memory entry with quality tracking
        
        Args:
            query_embedding: Query embedding [1, embedding_dim]
            query_cardinality: True cardinality
            prediction_error: Optional prediction error (lower is better)
        """
        # Only update during training
        if not self.training:
            return False
        
        # Normalize embedding before storing
        query_embedding = F.normalize(query_embedding, p=2, dim=-1)
        
        # Check if similar entry exists
        if self.memory_used.any():
            used_indices = torch.where(self.memory_used)[0]
            used_embeddings = self.memory_embeddings[used_indices]
            similarities = self._compute_cosine_similarity(query_embedding, used_embeddings)
            
            # If very similar entry exists, update it instead of adding new
            if similarities.max() > 0.95:
                most_similar_idx = used_indices[similarities.argmax()]
                return self._update_existing_entry(most_similar_idx, query_embedding, 
                                                   query_cardinality, prediction_error)
        
        # Find slot to insert
        slot_idx = self._find_insertion_slot()
        if slot_idx is not None:
            self._add_to_slot(slot_idx, query_embedding, query_cardinality, prediction_error)
            return True
        
        return False

    def _update_existing_entry(self, slot_idx, query_embedding, query_cardinality, prediction_error):
        """Update an existing memory entry with exponential moving average"""
        alpha = 0.3  # EMA coefficient
        
        with torch.no_grad():
            # Update embedding with EMA
            old_embedding = self.memory_embeddings[slot_idx]
            new_embedding = alpha * query_embedding.squeeze(0) + (1 - alpha) * old_embedding
            self.memory_embeddings.data[slot_idx] = F.normalize(new_embedding, p=2, dim=-1)
            
            # Update cardinality with EMA
            old_card = self.memory_cardinalities[slot_idx]
            if isinstance(query_cardinality, torch.Tensor):
                new_card = alpha * query_cardinality.item() + (1 - alpha) * old_card.item()
            else:
                new_card = alpha * query_cardinality + (1 - alpha) * old_card.item()
            self.memory_cardinalities.data[slot_idx] = new_card
            
            # Update quality score
            if prediction_error is not None:
                # 将误差转换为质量分数 (0-1, higher is better)
                # 使用sigmoid型函数: quality = 1 / (1 + error)
                # 这样error=0时quality=1, error增大时quality平滑下降
                new_quality = 1.0 / (1.0 + abs(prediction_error))
                old_quality = self.memory_quality_scores[slot_idx]
                self.memory_quality_scores.data[slot_idx] = alpha * new_quality + (1 - alpha) * old_quality
        
        return True

    def _find_insertion_slot(self):
        """Find the best slot for inserting a new memory entry"""
        # First, try to find unused slot
        unused_indices = torch.where(~self.memory_used)[0]
        if len(unused_indices) > 0:
            # Prefer exploratory zone for new entries
            exp_zone_start = self.hq_size
            exp_unused = unused_indices[unused_indices >= exp_zone_start]
            if len(exp_unused) > 0:
                return exp_unused[0].item()
            return unused_indices[0].item()
        
        # All slots used - need to replace
        return self._find_replacement_slot()

    def _find_replacement_slot(self):
        """
        Find slot to replace using a multi-criteria strategy:
        1. Low quality entries in exploratory zone
        2. Low usage entries with low quality
        3. Oldest entries (implicitly via usage count)
        """
        # Separate high-quality and exploratory zones
        hq_indices = torch.arange(self.hq_size, device=self.memory_used.device)
        exp_indices = torch.arange(self.hq_size, self.memory_size, device=self.memory_used.device)
        
        # First, try to replace in exploratory zone
        exp_qualities = self.memory_quality_scores[exp_indices]
        exp_usage = self.memory_usage_count[exp_indices]
        
        # Compute replacement score (lower is better)
        # Prefer low quality and low usage
        exp_scores = exp_qualities + 0.1 * (exp_usage / (exp_usage.max() + 1))
        worst_exp_idx = exp_indices[exp_scores.argmin()].item()
        
        # Only replace in HQ zone if exploratory entry is still good
        if exp_scores.min() > 0.6:
            hq_qualities = self.memory_quality_scores[hq_indices]
            hq_usage = self.memory_usage_count[hq_indices]
            hq_scores = hq_qualities + 0.1 * (hq_usage / (hq_usage.max() + 1))
            
            if hq_scores.min() < 0.4:  # Only replace truly bad HQ entries
                return hq_indices[hq_scores.argmin()].item()
        
        return worst_exp_idx

    def _add_to_slot(self, slot_idx, query_embedding, query_cardinality, prediction_error):
        """Add new entry to specified slot"""
        with torch.no_grad():
            # Normalize embedding before storing
            query_embedding = F.normalize(query_embedding, p=2, dim=-1)
            
            self.memory_embeddings.data[slot_idx] = query_embedding.squeeze(0)
            
            if isinstance(query_cardinality, torch.Tensor):
                self.memory_cardinalities.data[slot_idx] = query_cardinality.squeeze() if query_cardinality.numel() > 1 else query_cardinality.item()
            else:
                self.memory_cardinalities.data[slot_idx] = query_cardinality
            
            self.memory_used.data[slot_idx] = True
            self.memory_usage_count.data[slot_idx] = 0
            
            # Initialize quality score
            if prediction_error is not None:
                # 使用与_update_existing_entry相同的计算方式
                initial_quality = 1.0 / (1.0 + abs(prediction_error))
                self.memory_quality_scores.data[slot_idx] = initial_quality
            else:
                # Default to medium quality if no error provided
                self.memory_quality_scores.data[slot_idx] = 0.5

    def promote_to_hq(self, slot_idx):
        """Promote a good entry from exploratory to high-quality zone"""
        if slot_idx < self.hq_size:
            return  # Already in HQ zone
        
        # Find worst entry in HQ zone
        hq_qualities = self.memory_quality_scores[:self.hq_size]
        worst_hq_idx = hq_qualities.argmin().item()
        
        # Only promote if significantly better
        if self.memory_quality_scores[slot_idx] > hq_qualities[worst_hq_idx] + 0.2:
            # Swap entries
            self._swap_entries(slot_idx, worst_hq_idx)

    def _swap_entries(self, idx1, idx2):
        """Swap two memory entries"""
        with torch.no_grad():
            # Swap all attributes using .data
            temp_emb = self.memory_embeddings[idx1].clone()
            self.memory_embeddings.data[idx1] = self.memory_embeddings[idx2]
            self.memory_embeddings.data[idx2] = temp_emb
            
            temp_card = self.memory_cardinalities[idx1].clone()
            self.memory_cardinalities.data[idx1] = self.memory_cardinalities[idx2]
            self.memory_cardinalities.data[idx2] = temp_card
            
            temp_used = self.memory_used[idx1].clone()
            self.memory_used.data[idx1] = self.memory_used[idx2]
            self.memory_used.data[idx2] = temp_used
            
            temp_quality = self.memory_quality_scores[idx1].clone()
            self.memory_quality_scores.data[idx1] = self.memory_quality_scores[idx2]
            self.memory_quality_scores.data[idx2] = temp_quality
            
            temp_usage = self.memory_usage_count[idx1].clone()
            self.memory_usage_count.data[idx1] = self.memory_usage_count[idx2]
            self.memory_usage_count.data[idx2] = temp_usage
            
            temp_error = self.memory_prediction_errors[idx1].clone()
            self.memory_prediction_errors.data[idx1] = self.memory_prediction_errors[idx2]
            self.memory_prediction_errors.data[idx2] = temp_error

    def get_statistics(self):
        """Get memory bank statistics for monitoring"""
        if not self.memory_used.any():
            return {
                'total_entries': 0,
                'avg_quality': 0.0,
                'retrieval_success_rate': 0.0,
                'hq_entries': 0,
                'exp_entries': 0
            }
        
        used_indices = torch.where(self.memory_used)[0]
        used_qualities = self.memory_quality_scores[used_indices]
        
        hq_used = (used_indices < self.hq_size).sum().item()
        exp_used = (used_indices >= self.hq_size).sum().item()
        
        success_rate = (self.successful_retrievals.float() / self.total_retrievals.float()).item() \
                      if self.total_retrievals > 0 else 0.0
        
        return {
            'total_entries': len(used_indices),
            'avg_quality': used_qualities.mean().item(),
            'retrieval_success_rate': success_rate,
            'hq_entries': hq_used,
            'exp_entries': exp_used,
            'current_threshold': self.current_threshold.item()
        }


class MemoryAugmentedPredictor(nn.Module):
    """
    A predictor that combines base prediction with memory-augmented refinement
    """
    def __init__(self, base_predictor, memory_bank, refinement_hidden_dim=64):
        super(MemoryAugmentedPredictor, self).__init__()
        self.base_predictor = base_predictor
        self.memory_bank = memory_bank
        
        # Refinement network that learns to combine base prediction with memory info
        embedding_dim = memory_bank.embedding_dim
        self.refinement_net = nn.Sequential(
            nn.Linear(embedding_dim * 2 + 2, refinement_hidden_dim),  # +2 for base pred and confidence
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(refinement_hidden_dim, refinement_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(refinement_hidden_dim // 2, 1),
            nn.Tanh()  # Output is a scaling factor
        )
    
    def forward(self, query_embedding, base_prediction):
        """
        Refine base prediction using memory
        
        Args:
            query_embedding: Query embedding [1, embedding_dim]
            base_prediction: Base cardinality prediction
        
        Returns:
            refined_prediction: Memory-augmented prediction
        """
        # Retrieve from memory with confidence
        retrieved_embedding, confidence = self.memory_bank(query_embedding, return_confidence=True)
        
        # If low confidence, rely more on base prediction
        if confidence < 0.3:
            return base_prediction
        
        # Prepare input for refinement network
        confidence_tensor = torch.tensor([[confidence]], device=query_embedding.device)
        base_pred_tensor = base_prediction.view(1, 1) if base_prediction.dim() == 0 else base_prediction
        
        refinement_input = torch.cat([
            query_embedding,
            retrieved_embedding,
            base_pred_tensor,
            confidence_tensor
        ], dim=-1)
        
        # Compute refinement factor
        refinement_factor = self.refinement_net(refinement_input)
        
        # Apply refinement: refined = base * (1 + alpha * refinement_factor)
        # where alpha is confidence-weighted
        alpha = confidence * 0.3  # Max 30% adjustment
        refined_prediction = base_pred_tensor * (1 + alpha * refinement_factor)
        
        return refined_prediction.squeeze()
