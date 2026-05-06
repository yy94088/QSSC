import torch
import numpy as np
import random
from .active_util import _to_cuda, _to_dataloaders, _to_datasets, print_eval_res
import datetime
import os
import math
import json
from cardnet.loss import QErrorLoss, AdaptiveWeightedMSELoss, DistillationLoss
import hashlib
import pickle
import copy


class ActiveLearner:
	"""Simplified ActiveLearner - regression only, no classification or active learning."""
	
	def __init__(self, args, QD=None):
		self.args = args
		self.QD = QD
		self.current_mode = getattr(args, 'mode', 'unknown')
		
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
				confidence_threshold=getattr(args, 'confidence_threshold', 1.0),
				min_quality=getattr(args, 'distill_min_quality', 0.0),
				hard_only_quality_threshold=getattr(args, 'distill_hard_only_quality_threshold', 0.2),
			)
			self.distill_warmup_epochs = int(getattr(args, 'distill_warmup_epochs', 0))
			self.distill_ramp_epochs = int(getattr(args, 'distill_ramp_epochs', 0))
			print(f"Knowledge Distillation Enabled: alpha={args.distillation_alpha}, "
				  f"T={args.distillation_temperature}, threshold={args.confidence_threshold}")
		
		if self.cache_enabled:
			print("Query prediction cache enabled (Warning: may slow down training)")

	def set_run_mode(self, mode):
		"""Set current running mode for tagging analysis files."""
		self.current_mode = mode

	def _extract_graph_stats(self, decomp_x, decomp_edge_index):
		"""Extract lightweight graph-structure statistics for error analysis."""
		num_subgraphs = len(decomp_x)
		nodes_per_subgraph = [int(x.size(0)) for x in decomp_x if isinstance(x, torch.Tensor)]
		edges_per_subgraph = [int(ei.size(1)) for ei in decomp_edge_index if isinstance(ei, torch.Tensor)]

		total_nodes = int(sum(nodes_per_subgraph))
		total_edges = int(sum(edges_per_subgraph))
		avg_nodes = float(total_nodes / max(num_subgraphs, 1))
		avg_edges = float(total_edges / max(num_subgraphs, 1))
		edge_node_ratio = float(total_edges / max(total_nodes, 1))

		return {
			"num_subgraphs": int(num_subgraphs),
			"total_nodes": total_nodes,
			"total_edges": total_edges,
			"avg_nodes_per_subgraph": avg_nodes,
			"avg_edges_per_subgraph": avg_edges,
			"edge_node_ratio": edge_node_ratio,
			"nodes_per_subgraph": nodes_per_subgraph,
			"edges_per_subgraph": edges_per_subgraph,
		}

	def _analyze_error_reason(self, true_log2, pred_log2, q_error, query_size, graph_stats, soft_val):
		"""Generate heuristic tags for why a prediction error may be large."""
		tags = []
		delta = pred_log2 - true_log2

		if delta > 0:
			tags.append("over_estimation")
		elif delta < 0:
			tags.append("under_estimation")

		if q_error >= 100:
			tags.append("extreme_q_error")
		elif q_error >= 30:
			tags.append("very_high_q_error")

		if query_size >= getattr(self.args, 'large_query_size_threshold', 12):
			tags.append("large_query_size")
		if graph_stats["num_subgraphs"] >= getattr(self.args, 'complex_subgraph_threshold', 8):
			tags.append("many_decomposed_subgraphs")

		if true_log2 <= 1.0:
			tags.append("small_cardinality_sensitive")
		elif true_log2 >= 15.0:
			tags.append("large_cardinality_long_tail")

		ratio = graph_stats["edge_node_ratio"]
		if ratio < 1.0:
			tags.append("sparse_structure")
		elif ratio > 2.0:
			tags.append("dense_structure")

		if np.isnan(soft_val) or np.isinf(soft_val):
			tags.append("no_reliable_teacher_signal")

		if not tags:
			tags.append("unknown_complex_pattern")

		return tags

	def _get_high_qerror_save_path(self):
		"""Resolve output path for high q-error analysis records."""
		if getattr(self.args, 'high_qerror_save_path', ''):
			return self.args.high_qerror_save_path

		base_dir = os.path.join(self.args.save_res_dir, self.args.dataset)
		os.makedirs(base_dir, exist_ok=True)
		file_name = "high_qerror_queries_{}_{}_{}.jsonl".format(
			self.args.dataset,
			self.current_mode,
			self.args.model_type,
		)
		return os.path.join(base_dir, file_name)

	def _write_high_qerror_records(self, records):
		"""Persist high q-error records to JSONL + compact CSV summary."""
		if not records:
			return

		def _to_jsonable(obj):
			"""Recursively convert tensors/ndarrays/scalars to JSON-serializable types."""
			if isinstance(obj, torch.Tensor):
				if obj.numel() == 1:
					return obj.item()
				return obj.detach().cpu().tolist()
			if isinstance(obj, np.ndarray):
				return obj.tolist()
			if isinstance(obj, np.generic):
				return obj.item()
			if isinstance(obj, dict):
				return {str(k): _to_jsonable(v) for k, v in obj.items()}
			if isinstance(obj, (list, tuple)):
				return [_to_jsonable(v) for v in obj]
			return obj

		save_path = self._get_high_qerror_save_path()
		os.makedirs(os.path.dirname(save_path), exist_ok=True)

		write_mode = 'a' if getattr(self.args, 'append_high_qerror_log', True) else 'w'
		with open(save_path, write_mode, encoding='utf-8') as f:
			for item in records:
				f.write(json.dumps(_to_jsonable(item), ensure_ascii=True) + "\n")

		# Write an easy-to-scan csv sidecar for quick inspection.
		csv_path = save_path.replace('.jsonl', '.summary.csv') if save_path.endswith('.jsonl') else save_path + '.summary.csv'
		csv_header = "mode,eval_set,sample,q_error,signed_log10_qe,query_size,total_nodes,total_edges,query_file_name,query_load_path,query_local_idx,reason_tags\n"
		csv_mode = 'a' if (getattr(self.args, 'append_high_qerror_log', True) and os.path.exists(csv_path)) else 'w'
		with open(csv_path, csv_mode, encoding='utf-8') as cf:
			if csv_mode == 'w':
				cf.write(csv_header)
			for item in records:
				query_meta = item.get("query_meta", {})
				cf.write(
					"{},{},{},{:.6f},{:.6f},{},{},{},{},{},{},{}\n".format(
						item["mode"],
						item["eval_set_index"],
						item["sample_index"],
						item["q_error"],
						item["signed_log10_q_error"],
						item["query_size"],
						item["graph_stats"]["total_nodes"],
						item["graph_stats"]["total_edges"],
						str(query_meta.get("file_name", "")),
						str(query_meta.get("query_load_path", "")),
						str(query_meta.get("local_idx", "")),
						"|".join(item["reason_tags"]),
					)
				)

		print("Saved {} high q-error records to {}".format(len(records), save_path))
	
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
		best_val_l1 = float("inf")
		best_epoch = -1
		best_loader_idx = -1
		best_model_state = None
		best_optimizer_state = None
		best_scheduler_state = None
		orig_record_high_qerror = bool(getattr(self.args, 'record_high_qerror', False))
		
		# Use improved loss function
		base_criterion = self._get_loss_function()

		for loader_idx, dataloader in enumerate(train_loaders):
			model.train()
			print(f"Training the {loader_idx}/{len(train_loaders)} Training set")
			
			for epoch in range(epochs):
				epoch_loss = 0.0
				if self.use_distillation:
					if epoch < self.distill_warmup_epochs:
						distill_alpha_scale = 0.0
					elif self.distill_ramp_epochs <= 0:
						distill_alpha_scale = 1.0
					else:
						progress = (epoch - self.distill_warmup_epochs + 1) / float(self.distill_ramp_epochs)
						distill_alpha_scale = max(0.0, min(1.0, progress))
				else:
					distill_alpha_scale = 1.0
				
				for i, batch_data in enumerate(dataloader):
					# Unpack batch: always include subgraph_distances for relation network
					if len(batch_data) >= 7:
						decomp_x, decomp_edge_index, decomp_edge_attr, card, soft_card, subgraph_distances, _ = batch_data[:7]
					elif len(batch_data) >= 6:
						decomp_x, decomp_edge_index, decomp_edge_attr, card, soft_card, subgraph_distances = batch_data[:6]
					else:
						decomp_x, decomp_edge_index, decomp_edge_attr, card, soft_card = batch_data[:5]
						n = len(decomp_x)
						subgraph_distances = torch.zeros((n, n), dtype=torch.long)
					
					if self.args.cuda:
						decomp_x = _to_cuda(decomp_x)
						decomp_edge_index = _to_cuda(decomp_edge_index)
						decomp_edge_attr = _to_cuda(decomp_edge_attr)
						card = card.cuda()
						subgraph_distances = subgraph_distances.cuda()
						if soft_card is not None:
							soft_card = soft_card.cuda()

					# Forward pass
					output, hid_g = model(
						decomp_x,
						decomp_edge_index,
						decomp_edge_attr,
						subgraph_distances=subgraph_distances
					)
					
					# Handle output dimensions
					if isinstance(output, torch.Tensor):
						output = output.squeeze()
						if output.numel() > 1:
							output = output.mean()

					# Compute loss with optional distillation
					if self.use_distillation and soft_card is not None:
						# Use distillation loss
						loss, quality_weights = self.distillation_loss(
							output,
							soft_card,
							card,
							alpha_scale=distill_alpha_scale,
						)
						
						# Log quality weights occasionally for monitoring
						if i % 100 == 0:
							avg_quality = quality_weights.mean().item()
							eff_alpha = self.distillation_loss.alpha * distill_alpha_scale * avg_quality
							print(
								f"    Batch {i}: Soft label quality weight = {avg_quality:.4f}, "
								f"alpha_scale={distill_alpha_scale:.4f}, effective_alpha={eff_alpha:.4f}"
							)
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
				
				# Select best checkpoint by validation L1 (lower is better).
				if self.cache_enabled:
					self.clear_cache()
				self.args.record_high_qerror = False
				val_eval_res = self.evaluate(model, criterion, val_datasets, print_res=False)
				self.args.record_high_qerror = orig_record_high_qerror

				total_l1 = 0.0
				total_cnt = 0
				for res, _, l1_sum, _ in val_eval_res:
					total_l1 += float(l1_sum)
					total_cnt += len(res)
				val_l1 = (total_l1 / max(total_cnt, 1))

				if val_l1 < best_val_l1:
					best_val_l1 = val_l1
					best_epoch = epoch
					best_loader_idx = loader_idx
					best_model_state = copy.deepcopy(model.state_dict())
					best_optimizer_state = copy.deepcopy(optimizer.state_dict())
					if scheduler is not None:
						best_scheduler_state = copy.deepcopy(scheduler.state_dict())

				print(
					f"  {loader_idx}-th QuerySet, {epoch}-th Epoch: "
					f"Loss={epoch_loss:.4f}, ValL1={val_l1:.6f}, BestValL1={best_val_l1:.6f}"
				)

		# Restore best validation checkpoint before returning/saving.
		if best_model_state is not None:
			model.load_state_dict(best_model_state)
			optimizer.load_state_dict(best_optimizer_state)
			if scheduler is not None and best_scheduler_state is not None:
				scheduler.load_state_dict(best_scheduler_state)
			print(
				f"Restored best validation checkpoint: "
				f"loader={best_loader_idx}, epoch={best_epoch}, best_val_l1={best_val_l1:.6f}"
			)

		# Final validation report/logging on best model state.
		if self.cache_enabled:
			self.clear_cache()
		self.args.record_high_qerror = orig_record_high_qerror
		_ = self.evaluate(model, criterion, val_datasets, print_res=True)
		
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
		high_qerror_records = []
		log10_2 = math.log10(2.0)
		record_high_qerror = getattr(self.args, 'record_high_qerror', False)
		qerror_threshold = float(getattr(self.args, 'qerror_record_threshold', 10.0))
		high_qerror_topk = int(getattr(self.args, 'high_qerror_topk', 200))
		eval_loaders = _to_dataloaders(datasets=eval_datasets)
		
		def _normalize_query_meta(raw_meta, sample_idx=0):
			"""Normalize dataloader-collated metadata into a plain dict."""
			if raw_meta is None:
				return {}

			if isinstance(raw_meta, list):
				if not raw_meta:
					return {}
				pick = raw_meta[min(sample_idx, len(raw_meta) - 1)]
				if isinstance(pick, dict):
					return pick
				return {}

			if isinstance(raw_meta, dict):
				normalized = {}
				for key, val in raw_meta.items():
					if isinstance(val, list):
						normalized[key] = val[min(sample_idx, len(val) - 1)] if len(val) > 0 else None
					elif isinstance(val, torch.Tensor):
						if val.numel() == 1:
							normalized[key] = val.item()
						else:
							idx = min(sample_idx, val.numel() - 1)
							normalized[key] = val.view(-1)[idx].item()
					else:
						normalized[key] = val
				return normalized

			return {}

		for loader_idx, dataloader in enumerate(eval_loaders):
			res = []
			loss, l1 = 0.0, 0.0
			start = datetime.datetime.now()
			
			for i, batch_data in enumerate(dataloader):
				# Unpack batch
				query_meta = {}
				if len(batch_data) >= 8:
					decomp_x, decomp_edge_index, decomp_edge_attr, card, soft_card, subgraph_distances, query_size, query_meta = batch_data[:8]
				elif len(batch_data) >= 7:
					decomp_x, decomp_edge_index, decomp_edge_attr, card, soft_card, subgraph_distances, query_size = batch_data[:7]
				elif len(batch_data) >= 6:
					decomp_x, decomp_edge_index, decomp_edge_attr, card, soft_card, subgraph_distances = batch_data[:6]
					query_size = torch.tensor([len(decomp_x)], dtype=torch.long)
				else:
					decomp_x, decomp_edge_index, decomp_edge_attr, card, soft_card = batch_data[:5]
					n = len(decomp_x)
					subgraph_distances = torch.zeros((n, n), dtype=torch.long)
					query_size = torch.tensor([len(decomp_x)], dtype=torch.long)

				# Check cache
				cache_hit = False
				if self.cache_enabled:
					query_hash = self._get_query_hash(decomp_x, decomp_edge_index, decomp_edge_attr)
					if query_hash is not None and query_hash in self.query_cache:
						output = self.query_cache[query_hash]
						cache_hit = True
						self.cache_hits += 1
					else:
						self.cache_misses += 1

				if self.args.cuda:
					card = card.cuda()
					subgraph_distances = subgraph_distances.cuda()
					if soft_card is not None:
						soft_card = soft_card.cuda()
					
					if not cache_hit:
						decomp_x = _to_cuda(decomp_x)
						decomp_edge_index = _to_cuda(decomp_edge_index)
						decomp_edge_attr = _to_cuda(decomp_edge_attr)
					else:
						# Ensure output is on the correct device
						output = output.to(self.args.device)

				# Forward pass
				if not cache_hit:
					with torch.no_grad():
						output, hid_g = model(
							decomp_x,
							decomp_edge_index,
							decomp_edge_attr,
							subgraph_distances=subgraph_distances
						)
					
					# Save to cache if enabled
					if self.cache_enabled and query_hash is not None:
						self.query_cache[query_hash] = output.detach().cpu()

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
					if isinstance(query_size, torch.Tensor):
						size_val = int(query_size.view(-1)[min(j, query_size.numel() - 1)].item())
					else:
						size_val = int(query_size)

					# L1 error computation
					l1_error = abs(true_val - pred_val)
					l1 += l1_error
					res.append((true_val, pred_val, soft_val, size_val))

					if record_high_qerror:
						signed_log10_q_error = (pred_val - true_val) * log10_2
						abs_log10_q_error = abs(signed_log10_q_error)
						q_error = float(pow(10.0, abs_log10_q_error))
						if q_error >= qerror_threshold:
							graph_stats = self._extract_graph_stats(decomp_x, decomp_edge_index)
							query_fingerprint = self._get_query_hash(decomp_x, decomp_edge_index, decomp_edge_attr)
							normalized_meta = _normalize_query_meta(query_meta, sample_idx=j)
							reason_tags = self._analyze_error_reason(
								true_log2=true_val,
								pred_log2=pred_val,
								q_error=q_error,
								query_size=size_val,
								graph_stats=graph_stats,
								soft_val=soft_val,
							)
							high_qerror_records.append({
								"mode": self.current_mode,
								"timestamp": datetime.datetime.now().isoformat(),
								"eval_set_index": int(loader_idx),
								"batch_index": int(i),
								"sample_index": int(j),
								"query_size": int(size_val),
								"query_fingerprint": query_fingerprint,
								"true_log2_card": float(true_val),
								"pred_log2_card": float(pred_val),
								"teacher_log2_card": None if (np.isnan(soft_val) or np.isinf(soft_val)) else float(soft_val),
								"signed_log10_q_error": float(signed_log10_q_error),
								"abs_log10_q_error": float(abs_log10_q_error),
								"q_error": float(q_error),
								"graph_stats": graph_stats,
								"query_meta": normalized_meta,
								"reason_tags": reason_tags,
							})
			
			end = datetime.datetime.now()
			elapse_time = (end - start).total_seconds()
			all_eval_res.append((res, loss, l1, elapse_time))

		if print_res:
			print_eval_res(all_eval_res)
			# Print cache stats after evaluation
			if self.cache_enabled:
				self.print_cache_stats()

		if record_high_qerror and high_qerror_records:
			high_qerror_records = sorted(
				high_qerror_records,
				key=lambda x: x["q_error"],
				reverse=True,
			)
			high_qerror_records = high_qerror_records[:high_qerror_topk]
			self._write_high_qerror_records(high_qerror_records)
			print(
				"High q-error analysis summary: kept top {} samples (threshold >= {:.2f})".format(
					len(high_qerror_records),
					qerror_threshold,
				)
			)
		
		return all_eval_res
		if print_res:
			print_eval_res(all_eval_res)
		
		return all_eval_res