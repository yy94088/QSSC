import torch
import numpy as np
import random
from .active_util import _to_cuda, _to_dataloaders, _to_datasets, print_eval_res
import datetime

# Import improved loss functions
try:
	from cardnet.loss import QErrorLoss, AdaptiveWeightedMSELoss
except ImportError:
	QErrorLoss = None
	AdaptiveWeightedMSELoss = None

# Import data augmentation (optional)
try:
	from cardnet.data_augmentation import QueryGraphAugmentation, apply_augmentation_to_batch
	DATA_AUG_AVAILABLE = True
except ImportError:
	DATA_AUG_AVAILABLE = False


class ActiveLearner:
	"""Simplified ActiveLearner - regression only, no classification or active learning."""
	
	def __init__(self, args, QD=None):
		self.args = args
		self.QD = QD
		self.high_error_threshold = getattr(args, 'high_error_threshold', 1.0)
		self.distill_alpha = getattr(args, 'distill_alpha', 0.0)
		
		# Loss function type
		self.loss_type = getattr(args, 'loss_type', 'mse')
		
		# Data augmentation
		self.use_data_augmentation = getattr(args, 'use_data_augmentation', False)
		if self.use_data_augmentation and DATA_AUG_AVAILABLE:
			self.augmentor = QueryGraphAugmentation(
				feature_noise_std=getattr(args, 'aug_noise_std', 0.05),
				edge_dropout_rate=getattr(args, 'aug_edge_dropout', 0.1),
				feature_mask_rate=getattr(args, 'aug_feature_mask', 0.1)
			)
			print(f"✓ Data augmentation enabled")
		else:
			self.augmentor = None
	
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
					# Unpack batch (handle both formats)
					if len(batch_data) == 6:
						decomp_x, decomp_edge_index, decomp_edge_attr, card, _, soft_card = batch_data
					else:
						decomp_x, decomp_edge_index, decomp_edge_attr, card, soft_card = batch_data[:5]
					
					# Data augmentation
					if self.augmentor is not None and model.training:
						decomp_x, decomp_edge_index, decomp_edge_attr = apply_augmentation_to_batch(
							decomp_x, decomp_edge_index, decomp_edge_attr, 
							self.augmentor, training=True
						)
					
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
					
					# Check for NaN/Inf
					if torch.isnan(output).any() or torch.isinf(output).any():
						print(f"Warning: NaN/Inf in output. Skipping batch.")
						optimizer.zero_grad()
						continue
					
					if torch.isnan(card).any() or torch.isinf(card).any():
						print(f"Warning: NaN/Inf in target. Skipping batch.")
						optimizer.zero_grad()
						continue

					# Compute loss
					loss = base_criterion(output, card)
					
					if torch.isnan(loss):
						print(f"Warning: NaN in loss. Skipping batch.")
						optimizer.zero_grad()
						continue
						
					epoch_loss += loss.item()

					# Optional knowledge distillation
					if soft_card is not None and self.distill_alpha > 0:
						if not (torch.isnan(soft_card).any() or torch.isinf(soft_card).any()):
							distill_loss = base_criterion(output, soft_card)
							if not torch.isnan(distill_loss):
								loss += self.distill_alpha * distill_loss

					loss = loss / self.args.batch_size

					if torch.isnan(loss):
						print(f"Warning: NaN in final loss. Skipping batch.")
						optimizer.zero_grad()
						continue

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
		elapse_time = (end - start).total_seconds()
		print(f"Training time: {elapse_time:.4f}s")
		return model, elapse_time

	def evaluate(self, model, criterion, eval_datasets, print_res=False):
		"""Evaluate model on validation/test datasets."""
		if self.args.cuda:
			model.to(self.args.device)
		model.eval()
		
		all_eval_res = []
		eval_loaders = _to_dataloaders(datasets=eval_datasets)
		
		high_error_log_file = "high_error_queries.log" if hasattr(self.args, 'log_high_error') and self.args.log_high_error else None
		
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
				
					# Handle output dimensions
					if isinstance(output, torch.Tensor):
						output = output.squeeze()
						if output.numel() > 1:
							output = output.mean()
				
				# Compute metrics
				try:
					loss += criterion(output, card).item()
				except:
					loss_batch = criterion(output, card)
					loss += loss_batch.item() if isinstance(loss_batch, float) else loss_batch.sum().item()
				
				# Handle single sample or batch
				if isinstance(output, torch.Tensor) and output.dim() == 0:
					true_val = card.item()
					pred_val = output.item()
					soft_val = soft_card.item() if soft_card is not None else float('nan')
					l1_error = abs(true_val - pred_val)
					l1 += l1_error
					res.append((true_val, pred_val, soft_val))
					
					# Log high error queries
					if high_error_log_file and l1_error > self.high_error_threshold:
						with open(high_error_log_file, "a") as f:
							f.write(f"Loader: {loader_idx}, Query: {i}, "
									f"True: {true_val:.4f}, Pred: {pred_val:.4f}, "
									f"Error: {l1_error:.4f}\n")
				else:
					# Batched outputs
					batch_size = output.size(0) if output.dim() > 0 else 1
					for j in range(batch_size):
						true_val = card[j].item() if card.dim() > 0 else card.item()
						pred_val = output[j].item() if output.dim() > 0 else output.item()
						soft_val = soft_card[j].item() if (soft_card is not None and soft_card.dim() > 0) else float('nan')
						l1_error = abs(true_val - pred_val)
						l1 += l1_error
						res.append((true_val, pred_val, soft_val))

			end = datetime.datetime.now()
			elapse_time = (end - start).total_seconds()
			all_eval_res.append((res, loss, l1, elapse_time))

		if print_res:
			print_eval_res(all_eval_res)
		
		return all_eval_res
