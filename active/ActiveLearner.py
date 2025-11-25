import torch
import torch.distributions
import numpy as np
import random
from .active_util import _to_cuda, _to_dataloaders, _to_datasets, print_eval_res
from scipy.stats import  entropy, gmean
import datetime
import math
import logging
from cardnet.loss import ContrastiveLoss

class ActiveLearner:
	def __init__(self, args, QD=None):
		self.args = args
		self.QD = QD
		self.high_error_threshold = args.high_error_threshold if hasattr(args, 'high_error_threshold') else 1.0
		self.budget = args.budget
		self.uncertainty = args.uncertainty
		self.active_iters = args.active_iters
		self.active_epochs = args.active_epochs
		self.distill_alpha = args.distill_alpha
		self.distill_kl_alpha = getattr(args, 'distill_kl_alpha', 0.0)  # KL散度蒸馏权重
		self.distill_kl_bins = getattr(args, 'distill_kl_bins', 10)   # KL散度区间数
		self.biased_sample = args.biased_sample
		self.max_iterations = getattr(args, 'max_iterations', 5)
		self.tolerance = getattr(args, 'tolerance', 1e-4)

	def train(self, model, criterion, criterion_cal,
			  train_datasets, val_datasets, optimizer, scheduler=None, active = False):
		if self.args.cuda:
			model.to(self.args.device)
		epochs = self.active_epochs if active else self.args.epochs

		train_loaders = _to_dataloaders(datasets= train_datasets)
		start = datetime.datetime.now()
		contrastive_criterion = ContrastiveLoss(cardinality_threshold=self.args.cardinality_threshold)

		for loader_idx, dataloader in enumerate(train_loaders):
			model.train()
			print("Training the {}/{} Training set".format(loader_idx, len(train_loaders)))
			for epoch in range(epochs):
				# batch_embeddings = []
				# batch_cards = []
				epoch_loss, epoch_loss_cla, epoch_loss_distill, epoch_loss_distill_kl, epoch_loss_con = 0.0, 0.0, 0.0, 0.0, 0.0
				for i, (decomp_x, decomp_edge_index, decomp_edge_attr, card, label, soft_card) in \
						enumerate(dataloader):
					if self.args.cuda:
						decomp_x, decomp_edge_index, decomp_edge_attr = \
							_to_cuda(decomp_x), _to_cuda(decomp_edge_index), _to_cuda(decomp_edge_attr)
						card, label, soft_card = card.cuda(), label.cuda(), soft_card.cuda()

					# 使用迭代预测直到收敛（训练时也使用估计的card值）
					if self.args.memory_size > 0:
						# 迭代预测直到收敛
						with torch.no_grad():
							estimated_card, iterations = model.iterative_predict(
								decomp_x, decomp_edge_index, decomp_edge_attr,
								max_iterations=self.max_iterations,
								tolerance=self.tolerance
							)
						
						# 使用估计的card进行最终预测
						output, output_cla, hid_g = model(decomp_x, decomp_edge_index, decomp_edge_attr, card=estimated_card)
					else:
						# 常规预测方法
						output, output_cla, hid_g = model(decomp_x, decomp_edge_index, decomp_edge_attr, card)
					
					# 统一处理输出维度
					if isinstance(output, torch.Tensor):
						output = output.squeeze()
						# 如果仍然有多个元素，取均值
						if output.numel() > 1:
							output = output.mean()
					
					# 检查输出是否为NaN或inf
					if torch.isnan(output).any() or torch.isinf(output).any():
						print(f"Warning: NaN or Inf detected in model output. Skipping this batch.")
						optimizer.zero_grad()
						continue
					
					# 检查输入数据是否有效
					if torch.isnan(card).any() or torch.isinf(card).any():
						print(f"Warning: NaN or Inf detected in target card. Skipping this batch.")
						optimizer.zero_grad()
						continue

					loss = criterion(output, card) * (1 - self.distill_alpha - self.distill_kl_alpha)
					
					# 检查损失是否为NaN
					if torch.isnan(loss):
						print(f"Warning: NaN detected in loss. Skipping this batch.")
						optimizer.zero_grad()
						continue
						
					epoch_loss += loss.item()

					if soft_card is not None and not (torch.isnan(soft_card).any() or torch.isinf(soft_card).any()):
						# 原始MSE蒸馏损失
						if self.distill_alpha > 0:
							distill_loss = criterion(output, soft_card)  # MSE for regression
							# 检查蒸馏损失是否为NaN
							if not torch.isnan(distill_loss):
								loss += self.distill_alpha * distill_loss
								epoch_loss_distill += distill_loss.item()
							
						# KL散度蒸馏损失 - 基于区间策略的概率分布
						if self.distill_kl_alpha > 0:
							# 将数值转换为概率分布后再计算KL散度
							# 首先确定区间的边界（基于真实值和教师预测值的范围）
							min_val = min(torch.min(card).item(), torch.min(soft_card).item(), torch.min(output).item())
							max_val = max(torch.max(card).item(), torch.max(soft_card).item(), torch.max(output).item())
							
							# 添加一些边界扩展以确保覆盖所有值
							range_extension = (max_val - min_val) * 0.1
							min_val -= range_extension
							max_val += range_extension
							
							# 创建等间距的bins
							bins = torch.linspace(min_val, max_val, self.distill_kl_bins + 1).to(card.device)
							
							# 计算每个值落在各个区间内的概率（使用正态分布近似）
							def value_to_prob_distribution(value, bins, sigma=1.0):
								# 对于每个值，计算其在所有bins中的概率密度
								probs = torch.zeros(len(bins) - 1).to(value.device)
								for i in range(len(bins) - 1):
									# 使用累积分布函数计算值落在区间内的概率
									# 近似为以value为中心的正态分布
									cdf_high = 0.5 * (1 + torch.erf((bins[i+1] - value) / (sigma * math.sqrt(2))))
									cdf_low = 0.5 * (1 + torch.erf((bins[i] - value) / (sigma * math.sqrt(2))))
									probs[i] = cdf_high - cdf_low
								return probs / (probs.sum() + 1e-8)  # 归一化
							
							# 计算教师预测和学生预测的概率分布
							batch_size = card.shape[0]
							teacher_probs = torch.zeros(batch_size, self.distill_kl_bins).to(card.device)
							student_probs = torch.zeros(batch_size, self.distill_kl_bins).to(card.device)
							
							for b in range(batch_size):
								teacher_probs[b] = value_to_prob_distribution(soft_card[b], bins)
								student_probs[b] = value_to_prob_distribution(output[b], bins)
							
							# 添加小的epsilon值避免log(0)
							eps = 1e-8
							teacher_probs = torch.clamp(teacher_probs, eps, 1.0)
							student_probs = torch.clamp(student_probs, eps, 1.0)
							
							# 计算KL散度: D_KL(P||Q) = sum(P * log(P/Q))
							kl_loss = torch.sum(teacher_probs * torch.log(teacher_probs / student_probs), dim=1).mean()
							
							if not torch.isnan(kl_loss):
								loss += self.distill_kl_alpha * kl_loss
								epoch_loss_distill_kl += kl_loss.item()

					if self.args.multi_task and self.args.coeff > 0:
						# 检查分类输出是否有效
						if output_cla is not None and not (torch.isnan(output_cla).any() or torch.isinf(output_cla).any()):
							loss_cla = criterion_cal(output_cla, label)
							# 检查分类损失是否为NaN
							if not torch.isnan(loss_cla):
								loss += loss_cla * self.args.coeff
								epoch_loss_cla += loss_cla.item()

					loss = loss / self.args.batch_size

					# 检查最终损失是否为NaN
					if torch.isnan(loss):
						print(f"Warning: NaN detected in final loss. Skipping backward pass.")
						optimizer.zero_grad()
						continue

					loss.backward(retain_graph=(self.args.contrastive_weight > 0))

					if self.args.contrastive_weight > 0 and hid_g is not None and hid_g.size(0) > 1:
						# 确保hid_g不包含NaN
						if not (torch.isnan(hid_g).any() or torch.isinf(hid_g).any()):
							loss_con = contrastive_criterion(hid_g, card)
							# 检查对比损失是否为NaN
							if not torch.isnan(loss_con):
								loss += loss_con * self.args.contrastive_weight
								epoch_loss_con += loss_con.item()
								loss.backward()

					optimizer.step()
					optimizer.zero_grad()

						# batch_embeddings = []
						# batch_cards = []


					# 存储训练预测（仅最后一次epoch）
					if epoch == epochs - 1 and self.QD is not None:
						dataset = train_datasets[loader_idx]
						query_idx = i * self.args.batch_size
						if query_idx < len(dataset.queries):
							query = self.QD.get_query_by_index(self.args.pattern, self.args.size, query_idx)
							if query is not None:
								self.QD.update_train_prediction(query, output.item())

				if scheduler is not None and (epoch + 1) % self.args.decay_patience == 0:
					scheduler.step()
				print("{}-th QuerySet, {}-th Epoch: Reg. Loss={:.4f}, Cla. Loss={:.4f}, Distill Loss={:.4f}, Distill KL Loss={:.4f}"
					  .format(loader_idx, epoch, epoch_loss, epoch_loss_cla, epoch_loss_distill, epoch_loss_distill_kl))

			# Evaluation the model
			all_eval_res = self.evaluate(model, criterion, val_datasets, print_res = True)
		end = datetime.datetime.now()
		elapse_time = (end - start).total_seconds()
		print("Training time: {:.4f}s".format(elapse_time))
		return model, elapse_time

	def evaluate(self, model, criterion, eval_datasets, print_res=False):
		if self.args.cuda:
			model.to(self.args.device)
		model.eval()
		all_eval_res = []
		eval_loaders = _to_dataloaders(datasets=eval_datasets)
		
		# 创建或加载高误差查询记录文件
		high_error_log_file = "high_error_queries.log"
		
		for loader_idx, dataloader in enumerate(eval_loaders):
			res = []
			loss, l1 = 0.0, 0.0
			start = datetime.datetime.now()
			for i, (decomp_x, decomp_edge_index, decomp_edge_attr, card, label, soft_card) in \
					enumerate(dataloader):
				dataset = eval_datasets[loader_idx]
				query_idx = i * self.args.batch_size
				if query_idx >= len(dataset.queries):
					continue
				# 移除同构图检查，直接使用模型进行预测
				if self.args.cuda:
					decomp_x, decomp_edge_index, decomp_edge_attr = \
						_to_cuda(decomp_x), _to_cuda(decomp_edge_index), _to_cuda(decomp_edge_attr)
					card, label, soft_card = card.cuda(), label.cuda(), soft_card.cuda()
				
				# 使用迭代预测直到收敛
				with torch.no_grad():
					if self.args.memory_size > 0:
						# 迭代预测直到收敛
						output, iterations = model.iterative_predict(
							decomp_x, decomp_edge_index, decomp_edge_attr,
							max_iterations=self.max_iterations,
							tolerance=self.tolerance
						)
						output_cla = None  # 分类输出在迭代预测中未使用
						hid_g = None       # 隐藏状态在迭代预测中未使用
					else:
						# 常规预测方法
						output, output_cla, hid_g = model(decomp_x, decomp_edge_index, decomp_edge_attr)
					
					# 统一处理输出维度
					if isinstance(output, torch.Tensor):
						output = output.squeeze()
						# 如果仍然有多个元素，取均值
						if output.numel() > 1:
							output = output.mean()
				
				# accumulate loss (works for scalar or batch)
				try:
					loss += criterion(card, output).item()
				except Exception:
					# fallback: compute per-sample loss sum
					loss_batch = criterion(output, card)
					loss += loss_batch.item() if isinstance(loss_batch, float) else loss_batch.sum().item()
				
				# handle scalar (single sample) and batched outputs
				if isinstance(output, torch.Tensor) and output.dim() == 0:
					# single sample
					true_val = card.item()
					pred_val = output.item()
					soft_val = soft_card.item() if soft_card is not None else float('nan')
					l1_error = abs(true_val - pred_val)
					l1 += l1_error
					res.append((true_val, pred_val, soft_val))
					
					# log high error if enabled
					if hasattr(self.args, 'log_high_error') and self.args.log_high_error and l1_error > self.high_error_threshold:
						query_file_name = "unknown"
						if self.QD is not None:
							pattern = self.args.pattern
							size = self.args.size
							query_file_name = self.QD.query_file_names.get((pattern, size, query_idx), "unknown")
						with open(high_error_log_file, "a") as f:
							f.write(f"Loader: {loader_idx}, Query Index: {query_idx}, "
									f"Query File: {query_file_name}, "
									f"True Card: {true_val:.4f}, Pred Card: {pred_val:.4f}, "
									f"Soft Card: {soft_val:.4f}, "
									f"L1 Error: {l1_error:.4f}\n")
				else:
					# batched outputs
					batch_size = output.size(0)
					# card and soft_card should be tensors of shape [batch_size]
					for j in range(batch_size):
						sample_idx = query_idx + j
						if sample_idx >= len(dataset.queries):
							continue
						true_val = card[j].item()
						pred_val = output[j].item()
						soft_val = soft_card[j].item() if soft_card is not None else float('nan')
						l1_error = abs(true_val - pred_val)
						l1 += l1_error
						res.append((true_val, pred_val, soft_val))
						
						# log high error if enabled
						if hasattr(self.args, 'log_high_error') and self.args.log_high_error and l1_error > self.high_error_threshold:
							query_file_name = "unknown"
							if self.QD is not None:
								pattern = self.args.pattern
								size = self.args.size
								query_file_name = self.QD.query_file_names.get((pattern, size, sample_idx), "unknown")
							with open(high_error_log_file, "a") as f:
								f.write(f"Loader: {loader_idx}, Query Index: {sample_idx}, "
										f"Query File: {query_file_name}, "
										f"True Card: {true_val:.4f}, Pred Card: {pred_val:.4f}, "
										f"Soft Card: {soft_val:.4f}, "
										f"L1 Error: {l1_error:.4f}\n")

			end = datetime.datetime.now()
			elapse_time = (end - start).total_seconds()
			all_eval_res.append((res, loss, l1, elapse_time))

		if print_res:
			print_eval_res(all_eval_res)
		return all_eval_res

	def active_test(self, model, test_datasets, reject_set = None):
		if not test_datasets or all(len(dataset) == 0 for dataset in test_datasets):
			raise ValueError("Test datasets are empty!")
		print("Test datasets size:", [len(dataset) for dataset in test_datasets])
		assert self.args.multi_task, "Classification Task Disabled, Cannot Deploy Active Learning!"
		model.eval()
		test_uncertainties = []
		testset_dict = {}
		for dataset_idx, dataset in enumerate(test_datasets):
			for i in range(len(dataset)):
				# skip the test queries in the reject sets
				if reject_set is not None and (dataset_idx, i) in reject_set:
					continue
				decomp_x, decomp_edge_index, decomp_edge_attr, _, _, _ = dataset[i]
				if self.args.cuda:
					decomp_x, decomp_edge_index, decomp_edge_attr = \
						_to_cuda(decomp_x), _to_cuda(decomp_edge_index), _to_cuda(decomp_edge_attr)

				# print(decomp_x)
				output, output_cla, hid_g = model(decomp_x, decomp_edge_index, decomp_edge_attr)
				uncertainty = self.compute_uncertainty(output_cla, output)
				testset_dict[len(test_uncertainties)] = (dataset_idx, i)
				test_uncertainties.append(uncertainty)
		return self.select_active_sets(test_uncertainties, testset_dict, test_datasets)
	
	def select_active_sets(self, test_uncertainties, testset_dict, test_datasets):
		test_uncertainties = np.array(test_uncertainties)
		if len(test_uncertainties) == 0:
			return [], []
        # 避免除零，添加小值平滑
		test_uncertainties = test_uncertainties + 1e-10
		test_uncertainties = test_uncertainties / np.sum(test_uncertainties)
		num_selected = min(self.budget, len(test_uncertainties))  # 动态调整
		if self.biased_sample:
            # 过滤零概率
			non_zero_indices = np.where(test_uncertainties > 0)[0]
			if len(non_zero_indices) < num_selected:
				print(f"Warning: Only {len(non_zero_indices)} non-zero uncertainties, selecting all")
				indices = non_zero_indices
			else:
				indices = np.random.choice(non_zero_indices, size=num_selected, replace=False,
                                         p=test_uncertainties[non_zero_indices]/np.sum(test_uncertainties[non_zero_indices]))
		else:
			indices = np.argsort(test_uncertainties)[-num_selected:]
		selected_set = [testset_dict[idx] for idx in indices]
		active_sets = []
		for dataset_idx, i in selected_set:
			decomp_x, decomp_edge_index, decomp_edge_attr, card, soft_card = test_datasets[dataset_idx].queries[i]
			active_sets.append((decomp_x, decomp_edge_index, decomp_edge_attr, card, soft_card))
			return active_sets, selected_set


	def compute_uncertainty(self, output_cal, output):
		assert self.uncertainty == "entropy" or self.uncertainty == "confident" or self.uncertainty == "margin" \
			   or self.uncertainty == "random" or self.uncertainty == "consist", \
			"Unsupported uncertainty criterion"
		output_cal, output = output_cal.squeeze(), output.squeeze()
		output_cal = torch.exp(output_cal) # transform to probability
		if self.args.cuda:
			output_cal = output_cal.cpu()
			output = output.cpu()
		output = output.item()
		output_cal = output_cal.detach().numpy()
		if self.uncertainty == "entropy":
			return entropy(output_cal)
		elif self.uncertainty == "confident":
			return 1.0 - np.max(output_cal)
		elif self.uncertainty == "margin":
			res = output_cal[np.argsort(output_cal)[-1]] - output_cal[np.argsort(output_cal)[-2]]
			return res
		elif self.uncertainty == "random":
			return random.random()
		elif self.uncertainty == "consist":
			reg_mag = math.ceil( math.log10( math.pow(2, output)))
			cla_mag = np.argmax(output_cal)
			return math.pow((reg_mag - cla_mag), 2)

	def merge_datasets(self, train_datasets, active_sets):
		active_train_datasets = []
		for dataset in train_datasets:
			active_train_datasets += dataset.queries
		active_train_datasets += active_sets
		return _to_datasets([active_train_datasets])

	def print_selected_set_info(self, selected_set):
		cnt_dict = {}
		for dataset_idx, i in selected_set:
			if dataset_idx not in cnt_dict.keys():
				cnt_dict[dataset_idx] = 0
			cnt_dict[dataset_idx] += 1
		print("Selected set info: # Selected Queries: {}".format(len(selected_set)))
		for key in sorted(cnt_dict.keys()):
			print("# Select Query in {}-th Test set: {}.".format(key, cnt_dict[key]))


	def active_train(self, model, criterion, criterion_cla, train_datasets, val_datasets, test_datasets, optimizer, scheduler=None, pretrain=True):
		reject_set = []
		if pretrain:
			model, _ = self.train(model, criterion, criterion_cla, train_datasets, val_datasets, optimizer, scheduler, active=False)
			active_train_datasets = train_datasets
		for iter in range(self.active_iters):
			active_sets, selected_set = self.active_test(model, test_datasets, reject_set)
			if not active_sets:
				print(f"Skipping active learning iteration {iter} due to empty active_sets")
				continue
			reject_set += selected_set
			print("reject set size: {}".format(len(reject_set)))
			self.print_selected_set_info(reject_set)
			active_train_datasets = self.merge_datasets(train_datasets, active_sets)
			# 更新 train_graphs for active sets
			if self.QD is not None:
				for idx, (query, true_card) in enumerate(active_sets):
					graph_key = self.QD.serialize_graph(query)
					self.QD.train_graphs[graph_key] = {"graph": query, "pred": None}
					self.QD.query_indices[(self.args.pattern, self.args.size, len(self.QD.all_queries[(self.args.pattern, self.args.size)]) + idx)] = query
			print("The {}-th active Learning.".format(iter))
			model, _ = self.train(model, criterion, criterion_cla, active_train_datasets, val_datasets, optimizer, scheduler, active=True)


	def ensemble_evaluate(self, models, criterion, eval_datasets, print_res = False):
		"""
		get the final result of the ensemble models
		"""
		for model in models:
			if self.args.cuda:
				model.to(self.args.device)
			model.eval()
		all_eval_res = []
		eval_loaders = _to_dataloaders(datasets=eval_datasets)
		
		# 创建或加载高误差查询记录文件
		high_error_log_file = "high_error_queries_ensemble.log"
		
		for loader_idx, dataloader in enumerate(eval_loaders):
			res = []
			loss, l1 = 0.0, 0.0
			start = datetime.datetime.now()
			for i, (decomp_x, decomp_edge_index, decomp_edge_attr, card, label, soft_card) in \
					enumerate(dataloader):
				if self.args.cuda:
					decomp_x, decomp_edge_index, decomp_edge_attr = \
						_to_cuda(decomp_x), _to_cuda(decomp_edge_index), _to_cuda(decomp_edge_attr)
					card, label, soft_card = card.cuda(), label.cuda(), soft_card.cuda()

				# get the ensemble result
				outputs, losses, l1s = [], [], []
				for model in models:
					output, output_cla, hid_g = model(decomp_x, decomp_edge_index, decomp_edge_attr, card)
					output = output.squeeze()
					outputs.append(output.item())
					losses.append(criterion(card, output).item())
					l1s.append(torch.abs(card - output).item())
				#print(outputs, losses, l1s)
				geo_output = gmean(outputs)
				loss += np.mean(losses)
				l1_error = np.mean(l1s)
				l1 += l1_error

				res.append((card.item(), geo_output, soft_card.item()))
				
				# 记录高误差查询（如果启用）
				if hasattr(self.args, 'log_high_error') and self.args.log_high_error and l1_error > self.high_error_threshold:
					# 获取查询文件名
					query_file_name = "unknown"
					if self.QD is not None:
						# 根据loader_idx和query_idx获取查询
						pattern = self.args.pattern
						size = self.args.size
						query = self.QD.get_query_by_index(pattern, size, i)
						if query is not None:
							# 查找对应的查询文件名
							for idx, q in self.QD.query_indices.items():
								if idx[0] == pattern and idx[1] == size and self.QD.is_isomorphic(query, q):
									query_file_name = f"{pattern}_{size}_{idx[2]}"
									break
					
					with open(high_error_log_file, "a") as f:
						f.write(f"Loader: {loader_idx}, Query Index: {i}, "
								f"Query File: {query_file_name}, "
								f"True Card: {card.item():.4f}, Pred Card: {geo_output:.4f}, "
								f"L1 Error: {l1_error:.4f}\n")
								
			end = datetime.datetime.now()
			elapse_time = (end - start).total_seconds()
			all_eval_res.append((res, loss, l1, elapse_time))

		if print_res:
			print_eval_res(all_eval_res)
		return all_eval_res



	def ensemble_active_test(self, models, test_datasets, reject_set = None):

		for model in models:
			model.eval()
		test_uncertainties = []
		testset_dict = {}
		for dataset_idx, dataset in enumerate(test_datasets):
			for i in range(len(dataset)):
				# skip the test queries in the reject sets
				if reject_set is not None and (dataset_idx, i) in reject_set:
					continue
				decomp_x, decomp_edge_index, decomp_edge_attr, card, label, soft_card = dataset[i]
				if self.args.cuda:
					decomp_x, decomp_edge_index, decomp_edge_attr = \
						_to_cuda(decomp_x), _to_cuda(decomp_edge_index), _to_cuda(decomp_edge_attr)
					card, label, soft_card = card.cuda(), label.cuda(), soft_card.cuda()

				outputs = []
				for model in models:
					output, output_cla, hid_g = model(decomp_x, decomp_edge_index, decomp_edge_attr, card)
					output = output.squeeze()
					if self.args.cuda:
						output = output.cpu()
					outputs.append(output.item())
				# uncertainty is the ensemble variance
				uncertainty = np.var(outputs)
				testset_dict[len(test_uncertainties)] = (dataset_idx, i)
				test_uncertainties.append(uncertainty)
		return self.select_active_sets(test_uncertainties, testset_dict, test_datasets)


	def ensemble_active_train(self, models, criterion, criterion_cla, train_datasets, val_datasets, test_datasets, optimizers, schedulers = None, pretrain = True):

		cur_models = []
		reject_set = []

		if pretrain: # pretrain all ensembled models
			for model, optimizer, scheduler in zip(models, optimizers, schedulers):
				model, _ = self.train(model, criterion, criterion_cla, train_datasets, val_datasets, optimizer, scheduler, active = False)
				cur_models.append(model)

		print("Ensemble Eval Result of {} Models:".format(len(cur_models)))
		self.ensemble_evaluate(cur_models, criterion, val_datasets, print_res=True)
		for iter in range(self.active_iters):
			active_sets, selected_set = self.ensemble_active_test(cur_models, test_datasets, reject_set)

			# merge the reject set
			reject_set += selected_set
			print("reject set size: {}".format(len(reject_set)))
			self.print_selected_set_info(reject_set)

			active_train_datasets = self.merge_datasets(train_datasets, active_sets)
			print("The {}-th active Learning.".format(iter))

			tmp_models = []
			for model, optimizer, scheduler in zip(cur_models, optimizers, schedulers):
				model, _ = self.train(model, criterion, criterion_cla, active_train_datasets, val_datasets, optimizer,
								  	scheduler, active=True)
				tmp_models.append(model)
			cur_models = tmp_models
			print("Ensemble Eval Result of {} Models:".format(len(cur_models)))
			self.ensemble_evaluate(cur_models, criterion, val_datasets, print_res=True)