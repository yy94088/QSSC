import torch
import numpy as np
import random
from .active_util import _to_cuda, _to_dataloaders, _to_datasets, print_eval_res
import datetime
from cardnet.loss import QErrorLoss, AdaptiveWeightedMSELoss


class ActiveLearner:
	"""Simplified ActiveLearner - regression only, no classification or active learning."""
	
	def __init__(self, args, QD=None):
		self.args = args
		self.QD = QD
		
		# Loss function type
		self.loss_type = getattr(args, 'loss_type', 'mse')
	
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

					# Compute loss
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
