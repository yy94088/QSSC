import torch
import numpy as np
import random
from .active_util import _to_cuda, _to_dataloaders, _to_datasets, print_eval_res
from scipy.stats import  entropy, gmean
import datetime
import math
from cardnet.loss import ContrastiveLoss

class ActiveLearner(object):
	def __init__(self, args, QD):
		self.args = args
		self.budget = args.budget
		self.uncertainty = args.uncertainty
		self.active_iters = args.active_iters
		self.active_epochs = args.active_epochs
		self.distill_alpha = args.distill_alpha
		self.biased_sample = args.biased_sample
		self.QD = QD

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
				epoch_loss, epoch_loss_cla, epoch_loss_distill, epoch_loss_con = 0.0, 0.0, 0.0, 0.0
				for i, (decomp_x, decomp_edge_index, decomp_edge_attr, card, label, soft_card) in \
						enumerate(dataloader):
					if self.args.cuda:
						decomp_x, decomp_edge_index, decomp_edge_attr = \
							_to_cuda(decomp_x), _to_cuda(decomp_edge_index), _to_cuda(decomp_edge_attr)
						card, label, soft_card = card.cuda(), label.cuda(), soft_card.cuda()

					output, output_cla, hid_g = model(decomp_x, decomp_edge_index, decomp_edge_attr)
					output = output.squeeze()
					# batch_embeddings.append(hid_g)
					# batch_cards.append(card)

					loss = criterion(output, card) * (1 - self.distill_alpha)
					epoch_loss += loss.item()

					if soft_card is not None:
						distill_loss = criterion(output, soft_card)  # MSE for regression
						loss += self.distill_alpha * distill_loss
						epoch_loss_distill += distill_loss.item()

					if self.args.multi_task and self.args.coeff > 0:
						loss_cla = criterion_cal(output_cla, label)
						loss += loss_cla * self.args.coeff
						epoch_loss_cla += loss_cla.item()

					loss = loss / self.args.batch_size

					loss.backward(retain_graph=(self.args.contrastive_weight > 0))

					if self.args.contrastive_weight > 0 and hid_g.size(0) > 1:
						loss_con = contrastive_criterion(hid_g, card)
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
				print("{}-th QuerySet, {}-th Epoch: Reg. Loss={:.4f}, Cla. Loss={:.4f}"
					  .format(loader_idx, epoch, epoch_loss, epoch_loss_cla))

			# Evaluation the model
			all_eval_res = self.evaluate(model, criterion, val_datasets, print_res = True)
		end = datetime.datetime.now()
		elapse_time = (end - start).total_seconds()
		print("Training time: {:.4f}s".format(elapse_time))
		return model, elapse_time

	def evaluate(self, model, criterion, eval_datasets, print_res = False):
		if self.args.cuda:
			model.to(self.args.device)
		model.eval()
		all_eval_res = []
		eval_loaders = _to_dataloaders(datasets=eval_datasets)
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
				query = self.QD.get_query_by_index(self.args.pattern, self.args.size, query_idx)
                # 检查训练集中同构图
				train_pred = self.QD.get_train_prediction(query) if self.QD else None
				if train_pred is not None:
					print(f"Query {query_idx} in loader {loader_idx} found isomorphic match, using train_pred={train_pred}")
					output = torch.tensor([train_pred], device=card.device if self.args.cuda else 'cpu')
				else:
					if self.args.cuda:
						decomp_x, decomp_edge_index, decomp_edge_attr = \
							_to_cuda(decomp_x), _to_cuda(decomp_edge_index), _to_cuda(decomp_edge_attr)
						card, label, soft_card = card.cuda(), label.cuda(), soft_card.cuda()

					# print(decomp_x)
					output, output_cla, hid_g = model(decomp_x, decomp_edge_index, decomp_edge_attr)
					output = output.squeeze()
				loss += criterion(card, output).item()
				l1 += torch.abs(card - output).item()

				res.append((card.item(), output.item(), soft_card.item()))
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

				# print(decomp_x)
				# get the ensemble result
				outputs, losses, l1s = [], [], []
				for model in models:
					output, output_cla, hid_g = model(decomp_x, decomp_edge_index, decomp_edge_attr)
					output = output.squeeze()
					outputs.append(output.item())
					losses.append(criterion(card, output).item())
					l1s.append(torch.abs(card - output).item())
				#print(outputs, losses, l1s)
				geo_output = gmean(outputs)
				loss += np.mean(losses)
				l1 += np.mean(l1s)

				res.append((card.item(), geo_output, soft_card.item()))
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
				decomp_x, decomp_edge_index, decomp_edge_attr, _, _, _ = dataset[i]
				if self.args.cuda:
					decomp_x, decomp_edge_index, decomp_edge_attr = \
						_to_cuda(decomp_x), _to_cuda(decomp_edge_index), _to_cuda(decomp_edge_attr)

				outputs = []
				for model in models:
					output, output_cla, hid_g = model(decomp_x, decomp_edge_index, decomp_edge_attr)
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