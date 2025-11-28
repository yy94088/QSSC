import math
import torch.nn as nn
import torch
import torch.nn.functional as F
from .layers import MLP, FC
from torch_scatter import scatter_mean
from torch_geometric.nn import GINConv, GINEConv, NNConv, GATConv, GraphConv, SAGEConv
from .GINlayers import NNGINConv, NNGINConcatConv
from .improved_memory import ImprovedQueryMemoryBank

# ==================== QueryMemoryBank Definition ====================
class QueryMemoryBank(nn.Module):
	"""可学习的query记忆库"""
	def __init__(self, embedding_dim=128, memory_size=100, temperature=0.1, similarity_threshold=0.9):
		super(QueryMemoryBank, self).__init__()
		self.embedding_dim = embedding_dim
		self.memory_size = memory_size
		self.temperature = temperature
		self.similarity_threshold = similarity_threshold

		# 可学习的记忆embeddings和对应的基数
		self.memory_embeddings = nn.Parameter(torch.randn(memory_size, embedding_dim))
		self.memory_cardinalities = nn.Parameter(torch.randn(memory_size))
		self.memory_usage_count = torch.zeros(memory_size, dtype=torch.long)  # 记录每个记忆条目被多少其他查询匹配
		
		# 记录条目是否被使用
		self.register_buffer('memory_used', torch.zeros(memory_size, dtype=torch.bool))

	def forward(self, query_embedding, query_cardinality):
		used_mask = self.memory_used
		if not used_mask.any():
			return torch.zeros_like(query_embedding)
		
		used_indices = torch.where(used_mask)[0]
		used_embeddings = self.memory_embeddings[used_indices]
		used_cardinalities = self.memory_cardinalities[used_indices]

		similarities = torch.mm(query_embedding, used_embeddings.t())
		# 确保similarities是1D张量
		if similarities.dim() > 1:
			similarities = similarities.squeeze()
		elif similarities.dim() == 0:
			similarities = similarities.unsqueeze(0)

		# 处理card为None的情况（推理时）
		if query_cardinality is None:
			# 仅基于嵌入相似度检索（不考虑基数差异）
			valid_mask = similarities >= self.similarity_threshold
		else:
			# 确保query_cardinality是标量
			if hasattr(query_cardinality, 'numel') and query_cardinality.numel() > 1:
				query_cardinality = query_cardinality.mean()
			
			# 训练或验证阶段使用完整策略
			card_diff = torch.abs(query_cardinality - used_cardinalities)
			similarity_mask = similarities >= self.similarity_threshold
			cardinality_mask = card_diff <= 1.0
			valid_mask = similarity_mask & cardinality_mask
		
		if not valid_mask.any():
			return torch.zeros_like(query_embedding)
		
		valid_indices = torch.where(valid_mask)[0]
		# 处理维度问题
		if similarities.dim() == 0:
			valid_similarities = similarities.unsqueeze(0)
			valid_embeddings = used_embeddings
		else:
			valid_similarities = similarities[valid_indices]
			valid_embeddings = used_embeddings[valid_indices]
		
		attention_weights = F.softmax(valid_similarities / self.temperature, dim=-1 if valid_similarities.dim() > 0 else None)

		retrieved_embedding = torch.sum(attention_weights.unsqueeze(-1 if attention_weights.dim() > 0 else 0) * valid_embeddings, dim=0, keepdim=True)

		# 仅在训练阶段更新使用计数
		if self.training:
			for idx in valid_indices:
				original_idx = used_indices[idx]
				self.memory_usage_count[original_idx] += 1
		
		return retrieved_embedding

	def update_memory(self, query_embedding, query_cardinality):
		# 仅在训练阶段更新记忆库
		if not self.training:
			return False

		# 检查是否存在相似条目，避免重复插入
		if self._has_similar_entry(query_embedding, query_cardinality):
			return False
		
		# 尝试找到空闲插槽
		free_slot = self._find_free_slot()
		if free_slot is not None:
			self._add_to_slot(free_slot, query_embedding, query_cardinality)
			return True

		# 如果没有空闲插槽，则寻找需要替换的插槽
		slot_to_replace = self._find_slot_to_replace()
		if slot_to_replace is not None:
			self._add_to_slot(slot_to_replace, query_embedding, query_cardinality)
			return True
		
		# 无法插入新条目
		return False

	def _has_similar_entry(self, query_embedding, query_cardinality):
		if not self.memory_used.any():
			return False
			
		used_indices = torch.where(self.memory_used)[0]
		used_embeddings = self.memory_embeddings[used_indices]
		used_cardinalities = self.memory_cardinalities[used_indices]

		similarities = torch.mm(query_embedding, used_embeddings.t()).squeeze(0)

		card_diff = torch.abs(query_cardinality - used_cardinalities)

		similarity_mask = similarities >= self.similarity_threshold
		cardinality_mask = card_diff <= 1.0
		return (similarity_mask & cardinality_mask).any()

	def _find_free_slot(self):
		free_indices = torch.where(~self.memory_used)[0]
		if len(free_indices) > 0:
			return free_indices[0].item()
		return None

	def _find_slot_to_replace(self):
		if not self.memory_used.all():
			return self._find_free_slot()
		
		lower_threshold = self.similarity_threshold * 0.5
		
		used_indices = torch.where(self.memory_used)[0]
		usage_counts = torch.zeros(len(used_indices), dtype=torch.long)
		
		used_embeddings = self.memory_embeddings[used_indices]
		num_used = len(used_indices)
		
		for i in range(num_used):
			similarities = torch.mm(used_embeddings[i:i+1], used_embeddings.t()).squeeze(0)
			match_mask = similarities >= lower_threshold
			match_mask[i] = False
			usage_counts[i] = match_mask.sum()
		
		max_usage_idx = torch.argmax(usage_counts)
		return used_indices[max_usage_idx].item()

	def _add_to_slot(self, slot_idx, query_embedding, query_cardinality):
		with torch.no_grad():
			self.memory_embeddings[slot_idx].copy_(query_embedding.squeeze(0))
			if isinstance(query_cardinality, torch.Tensor):
				self.memory_cardinalities[slot_idx].copy_(query_cardinality.squeeze())
			else:
				self.memory_cardinalities[slot_idx].copy_(query_cardinality)
			self.memory_used[slot_idx] = True
			self.memory_usage_count[slot_idx] = 0

class Graph2Vec(nn.Module):
	def __init__(self, num_node_feat, num_edge_feat, g_hid, out_g_ch, dropout = True):
		super(Graph2Vec, self).__init__()
		nn1 = nn.Sequential(nn.Linear(num_node_feat, g_hid),
							nn.ReLU(),
							nn.Linear(g_hid, g_hid))
		nn2 = nn.Sequential(nn.Linear(g_hid, g_hid),
							nn.ReLU(),
							nn.Linear(g_hid, g_hid))
		nn3 = nn.Sequential(nn.Linear(g_hid, g_hid),
							nn.ReLU(),
							nn.Linear(g_hid, out_g_ch))
		self.cov1 = GINConv(nn = nn1)
		self.cov2 = GINConv(nn = nn2)
		self.cov3 = GINConv(nn = nn3)
		self.dropout = dropout


	def forward(self, x, edge_index, edge_attr):
		x = self.cov1(x = x, edge_index = edge_index)
		x = F.dropout(x, self.dropout, training=self.training)

		x = self.cov2(x=x, edge_index=edge_index)
		x = F.dropout(x, self.dropout, training=self.training)

		x = self.cov3(x=x, edge_index=edge_index)
		x = torch.unsqueeze(torch.sum(x, dim= 0), dim= 0)
		#x = scatter_mean(x, dim=0)
		return x


class DecomGNN(nn.Module):
	def __init__(self, args, num_node_feat, num_edge_feat):
		super(DecomGNN, self).__init__()
		self.num_node_feat = num_node_feat
		self.num_edge_feat = num_edge_feat
		self.num_layers = args.num_layers
		self.num_hid = args.num_g_hid
		self.num_e_hid = args.num_e_hid
		self.num_out = args.out_g_ch
		self.model_type = args.model_type
		self.dropout = args.dropout
		self.convs = nn.ModuleList()


		cov_layer = self.build_cov_layer(self.model_type)

		for l in range(self.num_layers):
			hidden_input_dim = self.num_node_feat if l == 0 else self.num_hid
			hidden_output_dim = self.num_out if l == self.num_layers - 1 else self.num_hid

			if self.model_type == "GIN" or self.model_type == "GINE" or self.model_type == "GAT" \
					or self.model_type == "GCN" or self.model_type == "SAGE":
				self.convs.append(cov_layer(hidden_input_dim, hidden_output_dim))
			elif self.model_type == "NN" or self.model_type == "NNGIN" or self.model_type == "NNGINConcat":
				self.convs.append(cov_layer(hidden_input_dim, hidden_output_dim, self.num_e_hid))
			elif self.model_type == "NNGINETransformer":
				# For NNGINETransformer, we need edge_dim parameter
				self.convs.append(cov_layer(hidden_input_dim, hidden_output_dim, self.num_edge_feat))
			else:
				print("Unsupported model type!")


	def build_cov_layer(self, model_type):
		if model_type == "GIN":
			return lambda in_ch, hid_ch : GINConv(nn= nn.Sequential(
				nn.Linear(in_ch, hid_ch), nn.ReLU(), nn.Linear(hid_ch, hid_ch)) )
		elif model_type == "GINE":
			return lambda in_ch, hid_ch : GINEConv(nn= nn.Sequential(
				nn.Linear(in_ch, hid_ch), nn.ReLU(), nn.Linear(hid_ch, hid_ch)) )
		elif model_type == "NN":
			return lambda in_ch, hid_ch, e_hid_ch : NNConv(in_ch, hid_ch,
				nn= nn.Sequential(nn.Linear(self.num_edge_feat, e_hid_ch),
								  nn.ReLU(), nn.Linear(e_hid_ch, in_ch * hid_ch)) )
		elif model_type == "NNGIN":
			return lambda in_ch, hid_ch, e_hid_ch : NNGINConv(
				edge_nn= nn.Sequential(nn.Linear(self.num_edge_feat, e_hid_ch), nn.ReLU(), nn.Linear(e_hid_ch, in_ch)),
				node_nn= nn.Sequential(nn.Linear(in_ch, hid_ch), nn.ReLU(), nn.Linear(hid_ch, hid_ch)) )
		elif model_type == "NNGINConcat":
			return lambda in_ch, hid_ch, e_hid_ch : NNGINConcatConv(
				edge_nn=nn.Sequential(nn.Linear(self.num_edge_feat, e_hid_ch), nn.ReLU(), nn.Linear(e_hid_ch, in_ch)),
				node_nn=nn.Sequential(nn.Linear(in_ch * 2, hid_ch), nn.ReLU(), nn.Linear(hid_ch, hid_ch)) )
		elif model_type == "NNGINETransformer":
			# Add support for NNGINETransformer
			from .GINlayers import NNGINETransformerConv
			return lambda in_ch, hid_ch, edge_dim: NNGINETransformerConv(
				in_channels=in_ch, out_channels=hid_ch, edge_dim=edge_dim)
		elif model_type == "GAT":
			return GATConv
		elif model_type == "SAGE":
			return SAGEConv
		elif model_type == "GCN":
			return GraphConv
		else:
			print("Unsupported model type!")


	def forward(self, x, edge_index, edge_attr = None):
		# 检查输入是否包含NaN
		if torch.isnan(x).any() or torch.isnan(edge_index).any():
			return torch.zeros(1, self.num_out, device=x.device)
			
		if edge_attr is not None and torch.isnan(edge_attr).any():
			return torch.zeros(1, self.num_out, device=x.device)

		for i in range(self.num_layers):
			try:
				if self.model_type == "GIN" or self.model_type == "GINE" or self.model_type == "GAT" \
						or self.model_type =="GCN" or self.model_type == "SAGE":
					x = self.convs[i](x, edge_index) # for GIN and GINE
				elif self.model_type == "NN" or self.model_type == "NNGIN" or self.model_type == "NNGINConcat":
					x = self.convs[i](x, edge_index, edge_attr)
				elif self.model_type == "NNGINETransformer":
					x = self.convs[i](x, edge_index, edge_attr)
				else:
					print("Unsupported model type!")
			except Exception as e:
				# 如果图卷积出错，返回零张量
				return torch.zeros(1, self.num_out, device=x.device)

			if i < self.num_layers - 1:
				x = F.dropout(x, p = self.dropout, training=self.training)
				
				# 检查输出是否包含NaN
				if torch.isnan(x).any() or torch.isinf(x).any():
					return torch.zeros(1, self.num_out, device=x.device)

		x = torch.unsqueeze(torch.sum(x, dim=0), dim=0)
		
		# 检查最终输出是否包含NaN
		if torch.isnan(x).any() or torch.isinf(x).any():
			return torch.zeros(1, self.num_out, device=x.device)
			
		return x


class Attention(nn.Module):
	"""
	Simple Attention layer
	"""
	def __init__(self, n_expert, n_hidden, v_hidden):
		super(Attention, self).__init__()
		self.n_expert = n_expert
		self.n_hidden = n_hidden
		self.v_hidden = v_hidden
		# Fix dimensions for proper matrix multiplication
		# w1: [n_hidden, v_hidden] - transforms from v_hidden to n_hidden
		self.w1 = nn.Parameter(torch.FloatTensor(self.n_hidden, self.v_hidden))
		# w2: [n_expert, n_hidden] - transforms from n_hidden to n_expert
		self.w2 = nn.Parameter(torch.FloatTensor(self.n_expert, self.n_hidden))
		self.reset_parameters()

	def reset_parameters(self):
		stdv = 1. / math.sqrt(self.w1.size(1))
		self.w1.data.uniform_(-stdv, stdv)
		stdv = 1. / math.sqrt(self.w2.size(1))
		self.w2.data.uniform_(-stdv, stdv)

	def forward(self, x):
		# x shape: [patch_num, out_g_ch]
		x = torch.transpose(x, 0, 1) # [out_g_ch, patch_num]
		
		# Check dimension compatibility
		if self.v_hidden != x.size(0):
			# If dimensions don't match, return uniform attention weights
			# This can happen when out_g_ch != v_hidden
			patch_num = x.size(1)
			# Return uniform weights
			return torch.ones(self.n_expert, patch_num, device=x.device) / patch_num

		# w1 is [n_hidden, v_hidden] and x is [v_hidden, patch_num]
		# support shape: [n_hidden, patch_num]
		support = F.tanh(self.w1.matmul(x)) # [n_hidden, patch_num]
		
		# Check for NaN in support
		if torch.isnan(support).any() or torch.isinf(support).any():
			patch_num = x.size(1)
			return torch.ones(self.n_expert, patch_num, device=x.device) / patch_num

		# w2 is [n_expert, n_hidden] and support is [n_hidden, patch_num]
		# output shape: [n_expert, patch_num]
		output = F.softmax(self.w2.matmul(support), dim = 1) #[n_expert, patch_num]
		
		# Check for NaN in output
		if torch.isnan(output).any() or torch.isinf(output).any():
			patch_num = x.size(1)
			return torch.ones(self.n_expert, patch_num, device=x.device) / patch_num
			
		return output



class CardNet(nn.Module):
	def __init__(self, args, num_node_feat, num_edge_feat):
		super(CardNet, self).__init__()
		self.num_node_feat = num_node_feat
		self.num_edge_feat = num_edge_feat
		self.num_expert = args.num_expert
		self.out_g_ch = args.out_g_ch
		self.num_att_hid = args.num_att_hid
		self.num_mlp_hid = args.num_mlp_hid
		self.num_classes = args.max_classes
		self.multi_task = args.multi_task
		self.pool_type = args.pool_type
		self.memory_size = args.memory_size
		self.similarity_threshold = args.similarity_threshold

		self.graph2vec = DecomGNN(args, self.num_node_feat, self.num_edge_feat)
		# Fix the Attention layer initialization to match dimensions properly
		# Attention(n_expert, n_hidden, v_hidden)
		# For proper dimension matching:
		# - v_hidden should match out_g_ch (dimension of graph embeddings)
		# - n_hidden can be any intermediate dimension
		self.att_layer = Attention(self.num_expert, self.num_att_hid, self.out_g_ch)
		self.mlp_in_ch = self.num_expert * self.out_g_ch if self.pool_type == "att" else self.out_g_ch
		
		if self.memory_size > 0:
			self.memory_bank = ImprovedQueryMemoryBank(
				embedding_dim=self.mlp_in_ch,
				memory_size=self.memory_size,
				high_quality_ratio=getattr(args, 'high_quality_ratio', 0.7),
				temperature=getattr(args, 'memory_temperature', 0.1),
				base_similarity_threshold=getattr(args, 'base_similarity_threshold', 0.85)
			)
			predictor_in_ch = self.mlp_in_ch * 2
		else:
			self.memory_bank = None
			predictor_in_ch = self.mlp_in_ch

		self.mlp = MLP(in_ch= predictor_in_ch, hid_ch= self.num_mlp_hid, out_ch= 1)

		self.fc_hid = FC(in_ch= predictor_in_ch, out_ch=self.num_mlp_hid)
		self.fc_reg = FC(in_ch= self.num_mlp_hid, out_ch= 1)
		self.fc_cla = FC(in_ch=self.num_mlp_hid, out_ch= self.num_classes)
		
		# 初始化所有参数，确保没有NaN值
		self.apply(self._init_weights)
		
	def _init_weights(self, module):
		if isinstance(module, nn.Linear):
			nn.init.xavier_uniform_(module.weight)
			if module.bias is not None:
				nn.init.zeros_(module.bias)
		elif isinstance(module, nn.Parameter):
			nn.init.uniform_(module, -0.1, 0.1)

	def iterative_predict(self, decomp_x, decomp_edge_index, decomp_edge_attr, max_iterations=5, tolerance=1e-4):
		"""
		迭代预测直到收敛
		Args:
			decomp_x: 查询图节点特征
			decomp_edge_index: 查询图边索引
			decomp_edge_attr: 查询图边属性
			max_iterations: 最大迭代次数
			tolerance: 收敛容差
		Returns:
			最终预测结果
		"""
		# 第一次预测（不使用记忆库）
		with torch.no_grad():
			initial_output, _, _ = self(decomp_x, decomp_edge_index, decomp_edge_attr)
			prev_card = initial_output.squeeze()
		
		# 如果prev_card是包含多个元素的张量，取平均值
		if hasattr(prev_card, 'numel') and prev_card.numel() > 1:
			prev_card = prev_card.mean()
		# 确保prev_card是标量
		elif prev_card.dim() > 0:
			prev_card = prev_card.squeeze()
		
		# 迭代优化
		for i in range(max_iterations):
			# 使用上一次的预测作为card参数
			output, _, _ = self(decomp_x, decomp_edge_index, decomp_edge_attr, card=prev_card)
			current_card = output.squeeze()
			
			# 如果current_card是包含多个元素的张量，取平均值
			if hasattr(current_card, 'numel') and current_card.numel() > 1:
				current_card = current_card.mean()
			# 确保current_card是标量
			elif current_card.dim() > 0:
				current_card = current_card.squeeze()
			
			# 检查收敛性
			diff = torch.abs(current_card - prev_card).item()
			if diff < tolerance:
				break
				
			prev_card = current_card
			
		return current_card, i + 1

	def forward(self, decomp_x, decomp_edge_index, decomp_edge_attr, card=None):
		g, output_cla = None, None

		for x, edge_index, edge_attr in zip(decomp_x, decomp_edge_index, decomp_edge_attr):
			x, edge_index, edge_attr = x.squeeze(), edge_index.squeeze(), edge_attr.squeeze()
			# 检查输入是否包含NaN
			if torch.isnan(x).any() or torch.isnan(edge_attr).any():
				# 如果输入包含NaN，返回零张量
				if g is None:
					g = torch.zeros(1, self.out_g_ch, device=x.device)
				else:
					g = torch.cat([g, torch.zeros(1, self.out_g_ch, device=x.device)], dim=0)
				continue
				
			if g is None:
				g = self.graph2vec(x, edge_index, edge_attr)
			else:
				g = torch.cat([g, self.graph2vec(x, edge_index, edge_attr)], dim = 0)
				
		# 检查g是否为空或包含NaN
		if g is None or torch.isnan(g).any() or torch.isinf(g).any():
			# 返回默认值
			x = torch.zeros(1, self.mlp_in_ch, device=next(self.parameters()).device)
		else:
			#print(g.shape)
			# g: [patch_num, out_g_ch]
			if self.pool_type == "sum":
				x = torch.sum(g, dim=0)
				x = x.unsqueeze(dim=0)
			elif self.pool_type == "mean":
				x = torch.mean(g, dim= 0)
				x = x.unsqueeze(dim=0)
			elif self.pool_type == "max":
				x, _ = torch.max(g, dim= 0, keepdim=True)
			else:
				att_wights = self.att_layer(g) # [num_expert, patch_num]
				g = att_wights.matmul(g) # g: [num_expert, out_g_ch]
				x = g.view((1, self.num_expert * self.out_g_ch))

		if self.memory_bank is not None:
			retrieved = self.memory_bank(x, card)
			x_combined = torch.cat([x, retrieved], dim=1)
			
			# Note: Memory update is handled externally in training loop 
			# after loss.backward() to avoid in-place operation errors
			
			x = x_combined
		else:
			x_combined = x

		# 检查x是否包含NaN
		if torch.isnan(x).any() or torch.isinf(x).any():
			x = torch.zeros_like(x)

		if self.multi_task:
			hid_g = F.relu(self.fc_hid(x))
			# 检查隐藏层输出
			if torch.isnan(hid_g).any() or torch.isinf(hid_g).any():
				hid_g = torch.zeros_like(hid_g)
				
			output = self.fc_reg(hid_g)
			output_cla = F.log_softmax(self.fc_cla(hid_g), dim= 1)
			
			# 检查输出是否包含NaN
			if torch.isnan(output).any() or torch.isinf(output).any():
				output = torch.tensor(0.0, device=output.device)
		else:
			output = self.mlp(x)
			# 检查输出是否包含NaN
			if torch.isnan(output).any() or torch.isinf(output).any():
				output = torch.tensor(0.0, device=output.device)

		return output, output_cla, x
