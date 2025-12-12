import torch
import numpy as np
import random
from .active_util import _to_cuda, _to_dataloaders, _to_datasets, print_eval_res
import datetime
from cardnet.loss import QErrorLoss, AdaptiveWeightedMSELoss, DistillationLoss
import hashlib
import pickle


class ActiveLearner:
	"""Simplified ActiveLearner - regression only, no classification or active learning."""
	
	def __init__(self, args, QD=None):
		self.args = args
		self.QD = QD
		
		# Query prediction cache: disabled by default to avoid performance issues
		self.query_cache = {}
		self.cache_enabled = getattr(args, 'use_cache', True)  # Changed to False by default
		self.cache_hits = 0
		self.cache_misses = 0
		
		# Loss function type
		self.loss_type = getattr(args, 'loss_type', 'mse')
		
		# Distillation parameters
		self.use_distillation = getattr(args, 'use_distillation', False)
		if self.use_distillation:
			self.distillation_loss = DistillationLoss(
				alpha=getattr(args, 'distillation_alpha', 0.5),
				temperature=getattr(args, 'distillation_temperature', 2.0),
				confidence_threshold=getattr(args, 'confidence_threshold', 1.0)
			)
			print(f"Knowledge Distillation Enabled: alpha={args.distillation_alpha}, "
				  f"T={args.distillation_temperature}, threshold={args.confidence_threshold}")
		
		if self.cache_enabled:
			print("Query prediction cache enabled (Warning: may slow down training)")
	
	def _get_loss_function(self):
		"""Get appropriate loss function based on configuration."""
		if self.loss_type == 'qerror' and QErrorLoss is not None:
			print("Using Q-Error Loss")
			return QErrorLoss()
		elif self.loss_type == 'weighted_mse' and AdaptiveWeightedMSELoss is not None:
			print(f"Using Adaptive Weighted MSE Loss")
			return AdaptiveWeightedMSELoss(weight_exp=self.args.weight_exp)
		elif self.loss_type == 'hybrid' and QErrorLoss is not None:
			print("Using Hybrid Loss (MSE + Q-Error)")
			qerror_loss = QErrorLoss()
			return lambda pred, target: (
				torch.nn.functional.mse_loss(pred, target) + 
				0.5 * qerror_loss(pred, target)
			)
		else:
			return torch.nn.MSELoss()
	
	def _get_query_hash(self, decomp_x, decomp_edge_index, decomp_edge_attr):
		"""
		Generate a lightweight hash for a query based on its graph structure.
		Uses simplified hashing for better performance.
		"""
		try:
			# Use a lightweight hash based on tensor shapes and sum of elements
			hash_parts = []
			
			# Hash based on structure (shapes and sums)
			for x in decomp_x:
				if isinstance(x, torch.Tensor):
					hash_parts.append(f"{x.shape}_{x.sum().item():.6f}")
			
			for ei in decomp_edge_index:
				if isinstance(ei, torch.Tensor):
					hash_parts.append(f"{ei.shape}_{ei.sum().item()}")
			
			for ea in decomp_edge_attr:
				if isinstance(ea, torch.Tensor):
					hash_parts.append(f"{ea.shape}_{ea.sum().item():.6f}")
			
			# Create hash from concatenated strings (much faster than pickle)
			hash_str = "|".join(hash_parts)
			return hashlib.md5(hash_str.encode()).hexdigest()
		except Exception as e:
			# If hashing fails, return None to skip caching
			return None
	
	def clear_cache(self):
		"""Clear the query prediction cache."""
		self.query_cache.clear()
		self.cache_hits = 0
		self.cache_misses = 0
		print("Query cache cleared")
	
	def get_cache_stats(self):
		"""Get cache statistics."""
		total_queries = self.cache_hits + self.cache_misses
		hit_rate = self.cache_hits / total_queries if total_queries > 0 else 0
		return {
			'cache_size': len(self.query_cache),
			'cache_hits': self.cache_hits,
			'cache_misses': self.cache_misses,
			'hit_rate': hit_rate
		}
	
	def print_cache_stats(self):
		"""Print cache statistics."""
		stats = self.get_cache_stats()
		print(f"\n{'='*50}")
		print(f"Query Cache Statistics:")
		print(f"  Cache Size: {stats['cache_size']}")
		print(f"  Cache Hits: {stats['cache_hits']}")
		print(f"  Cache Misses: {stats['cache_misses']}")
		print(f"  Hit Rate: {stats['hit_rate']:.2%}")
		print(f"{'='*50}\n")

	def train(self, model, criterion, train_datasets, val_datasets, optimizer, scheduler=None):
		"""
		Train the model for cardinality estimation (regression only).
		"""
		if self.args.cuda:
			model.to(self.args.device)
		epochs = self.args.epochs

		train_loaders = _to_dataloaders(datasets=train_datasets)
		start = datetime.datetime.now()
		
		# Use improved loss function
		base_criterion = self._get_loss_function()

		for loader_idx, dataloader in enumerate(train_loaders):
			model.train()
			print(f"Training the {loader_idx}/{len(train_loaders)} Training set")
			
			for epoch in range(epochs):
				epoch_loss = 0.0
				
				for i, batch_data in enumerate(dataloader):
					# Unpack batch
					if len(batch_data) == 6:
						decomp_x, decomp_edge_index, decomp_edge_attr, card, _, soft_card = batch_data
					else:
						decomp_x, decomp_edge_index, decomp_edge_attr, card, soft_card = batch_data[:5]
					
					if self.args.cuda:
						decomp_x = _to_cuda(decomp_x)
						decomp_edge_index = _to_cuda(decomp_edge_index)
						decomp_edge_attr = _to_cuda(decomp_edge_attr)
						card = card.cuda()
						if soft_card is not None:
							soft_card = soft_card.cuda()

					# Forward pass
					output, hid_g = model(decomp_x, decomp_edge_index, decomp_edge_attr)
					
					# Handle output dimensions
					if isinstance(output, torch.Tensor):
						output = output.squeeze()
						if output.numel() > 1:
							output = output.mean()

					# Compute loss with optional distillation
					if self.use_distillation and soft_card is not None:
						# Use distillation loss
						loss, quality_weights = self.distillation_loss(output, soft_card, card)
						
						# Log quality weights occasionally for monitoring
						if i % 100 == 0:
							avg_quality = quality_weights.mean().item()
							print(f"    Batch {i}: Soft label quality weight = {avg_quality:.4f}")
					else:
						# Use standard loss
						loss = base_criterion(output, card)
					
					epoch_loss += loss.item()
					loss = loss / self.args.batch_size

					# Backward and optimize
					loss.backward()
					optimizer.step()
					optimizer.zero_grad()

				# Learning rate scheduling
				if scheduler is not None and (epoch + 1) % self.args.decay_patience == 0:
					scheduler.step()
				
				print(f"  {loader_idx}-th QuerySet, {epoch}-th Epoch: Loss={epoch_loss:.4f}")

			# Evaluate after each loader
			all_eval_res = self.evaluate(model, criterion, val_datasets, print_res=True)
		
		end = datetime.datetime.now()
		end = datetime.datetime.now()
		elapse_time = (end - start).total_seconds()
		print(f"Training time: {elapse_time:.4f}s")
		
		return model, elapse_time
	def evaluate(self, model, criterion, eval_datasets, print_res=False):
		"""Evaluate model on validation/test datasets with caching support."""
		if self.args.cuda:
			model.to(self.args.device)
		model.eval()
		
		all_eval_res = []
		eval_loaders = _to_dataloaders(datasets=eval_datasets)
		
		for loader_idx, dataloader in enumerate(eval_loaders):
			res = []
			loss, l1 = 0.0, 0.0
			start = datetime.datetime.now()
			
			for i, batch_data in enumerate(dataloader):
				# Unpack batch
				if len(batch_data) == 6:
					decomp_x, decomp_edge_index, decomp_edge_attr, card, _, soft_card = batch_data
				else:
					decomp_x, decomp_edge_index, decomp_edge_attr, card, soft_card = batch_data[:5]

				if self.args.cuda:
					decomp_x = _to_cuda(decomp_x)
					decomp_edge_index = _to_cuda(decomp_edge_index)
					decomp_edge_attr = _to_cuda(decomp_edge_attr)
					card = card.cuda()
					if soft_card is not None:
						soft_card = soft_card.cuda()

				# Forward pass
				with torch.no_grad():
					output, hid_g = model(decomp_x, decomp_edge_index, decomp_edge_attr)

				# Flatten outputs and targets for unified processing
				output_flat = output.view(-1)
				card_flat = card.view(-1)
				soft_card_flat = soft_card.view(-1) if soft_card is not None else None

				# Compute metrics
				loss += criterion(output, card).item()

				batch_size = output_flat.size(0)
				for j in range(batch_size):
					true_val = card_flat[j].item()
					pred_val = output_flat[j].item()
					soft_val = soft_card_flat[j].item() if soft_card_flat is not None else float('nan')

					# L1 error computation
					l1_error = abs(true_val - pred_val)
					l1 += l1_error
					res.append((true_val, pred_val, soft_val))
			
			end = datetime.datetime.now()
			elapse_time = (end - start).total_seconds()
			all_eval_res.append((res, loss, l1, elapse_time))

		if print_res:
			print_eval_res(all_eval_res)
			# Print cache stats after evaluation
			if self.cache_enabled:
				self.print_cache_stats()
		
		return all_eval_res
		if print_res:
			print_eval_res(all_eval_res)
		
		return all_eval_res