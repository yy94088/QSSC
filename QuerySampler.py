import os
import numpy as np
import networkx as nx
import random
import queue
import pickle
from util import make_dir, save_true_card
import tqdm
from multiprocessing import Pool, cpu_count
import functools
from FlowSC import generate_adjacency_matrix, bfs_layers, data_flow, query_flow, generate_query_adj
from FlowSC import Find_Candidates, Flow_Learner
import logging
import torch
import json
import time
import subprocess
import psutil
from tqdm import tqdm

def load_graph(fname):
	file = open(fname)
	nodes_list = []
	edges_list = []
	label_cardinality = {}

	for line in file:
		if line.strip().startswith("v"):
			tokens = line.strip().split()
			# v nodeID labelID degree
			id = int(tokens[1])
			labels = tokens[2]
			if labels not in label_cardinality.keys():
				label_cardinality[labels] = 1
			else:
				label_cardinality[labels] += 1
			nodes_list.append((id, {"labels": labels}))
		if line.strip().startswith("e"):
			tokens = line.strip().split()
			src, dst = int(tokens[1]), int(tokens[2])
			labels = [] #tokens[3:]
			edges_list.append((src, dst, {"labels" : labels}))

	graph = nx.Graph()
	graph.add_nodes_from(nodes_list)
	graph.add_edges_from(edges_list)

	print('number of nodes: {}'.format(graph.number_of_nodes()))
	print('number of edges: {}'.format(graph.number_of_edges()))
	file.close()
	return graph

def save_graph(graph, graph_save_path):
	with open(graph_save_path, 'w') as file:
		file.write("t {} {} \n".format(graph.number_of_nodes(), graph.number_of_edges()))
		for v in graph.nodes():
			file.write("v {} {} {} \n".format(v, graph.nodes[v]['labels'], graph.degree[v]))
		for (u, v) in graph.edges():
			file.write("e {} {} \n".format(u, v))


class QuerySampler(object):
	def __init__(self, graph):
		self.graph = graph


	def sample_chain(self, node_num):
		nodes_list = []
		edges_list = []

		src = random.randint(0, self.graph.number_of_nodes())
		Q = queue.Queue()
		Q.put(src)
		while not Q.empty():
			cur = Q.get()
			if len(nodes_list) > 0:
				edges_list.append((nodes_list[-1], cur))
			nodes_list.append(cur)
			if len(nodes_list) == node_num:
				break

			candidates = set(list(self.graph.neighbors(cur))).difference(set(nodes_list))
			if len(candidates) == 0:
				continue
			next = random.choice(list(candidates))
			Q.put(next)

		if len(nodes_list) < node_num:
			return None
		sample = self.node_reorder(nodes_list, edges_list)

		return sample

	def sample_star(self, node_num):
		nodes_list = []
		edges_list = []
		while True:
			src = random.randint(0, self.graph.number_of_nodes())
			if self.graph.degree[src] >= node_num - 1:
				break
		nodes_list.append(src)
		nexts = random.sample(list(self.graph.neighbors(src)), k= node_num - 1)
		for v in nexts:
			nodes_list.append(v)
			edges_list.append((src, v))
		sample = self.node_reorder(nodes_list, edges_list)
		return  sample

	def sample_tree(self, node_num):
		nodes_list = []
		edges_list = []
		parent = {}

		src = random.randint(0, self.graph.number_of_nodes())
		Q = queue.Queue()
		Q.put(src)
		while not Q.empty():
			cur = Q.get()
			if len(nodes_list) > 0:
				edges_list.append((parent[cur], cur))
			nodes_list.append(cur)
			if len(nodes_list) == node_num:
				break

			candidates = set(list(self.graph.neighbors(cur))).difference(set(nodes_list))
			if len(candidates) == 0:
				continue
			nexts = random.sample(list(self.graph.neighbors(src)), k = random.randint(1, min(len(candidates), node_num - len(nodes_list))))
			for v in nexts:
				Q.put(v)
				parent[v] = cur

		sample = self.node_reorder(nodes_list, edges_list)
		return sample

	def sample_cycle(self, node_num):
		nodes_list = [(0, {"labels" : random.randint(0, 7)})]
		edges_list = []

		for v in range(1, node_num):
			nodes_list.append((v, {"labels" : random.randint(0, 7)}))
			edges_list.append((v - 1, v))
		edges_list.append((node_num - 1, 0))
		sample = nx.Graph()
		sample.add_nodes_from(nodes_list)
		sample.add_edges_from(edges_list)
		return sample

	def sample_clique(self, node_num):
		nodes_list = []
		edges_list = []
		for v in range(0, node_num):
			nodes_list.append((v, {"labels": random.randint(0, 7)}))
			for u in range(0, v):
				edges_list.append((u, v))
		sample = nx.Graph()
		sample.add_nodes_from(nodes_list)
		sample.add_edges_from(edges_list)
		return sample

	def node_reorder(self, nodes_list, edges_list):
		idx_dict = {}
		node_cnt = 0
		for v in nodes_list:
			idx_dict[v] = node_cnt
			node_cnt += 1
		nodes_list = [(idx_dict[v], {"labels": self.graph.nodes[v]["labels"]})
					  for v in nodes_list]
		edges_list = [(idx_dict[u], idx_dict[v], {"labels": self.graph.edges[u, v]["labels"]})
					  for (u, v) in edges_list]
		sample = nx.Graph()
		sample.add_nodes_from(nodes_list)
		sample.add_edges_from(edges_list)
		return sample

def process_single_query_parallel(args_tuple):
    """
    并行处理单个查询的函数
    """
    (query_load_path, card_load_path, data_graph_path, teacher_model_path, 
     queryset_load_path, true_card_load_path, pattern, size, idx, 
     edge_embeds, edge_index_map, device, upper_card, lower_card, dataset) = args_tuple
    
    try:
        # 用dataset参数初始化QueryDecompose，确保加载正确的edge_embeds
        temp_qd = QueryDecompose(
            queryset_dir=queryset_load_path.replace(f"/{pattern}_{size}", ""),
            true_card_dir=true_card_load_path.replace(f"/{pattern}_{size}", ""),
            dataset=dataset,
            pattern=pattern,
            teacher_model_path=teacher_model_path,
            k=3,
            size=size,
            device=device
        )
        temp_qd.edge_embeds = edge_embeds
        temp_qd.edge_index_map = edge_index_map
        temp_qd.upper_card = upper_card
        temp_qd.lower_card = lower_card
        
        # 加载查询
        query, label_den = temp_qd.load_query(query_load_path)
        graphs = temp_qd.decompose(query)
        true_card = temp_qd.load_card(card_load_path)
        
        if true_card >= upper_card or true_card < lower_card:
            return None
        
        true_card = true_card + 1 if true_card == 0 else true_card
        
        # 加载教师模型进行预测
        device_torch = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 确保CUDA正确初始化
        if device_torch.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.set_device(device_torch)
        
        teacher_model = Flow_Learner(out_dim=128).to(device_torch)
        state_dict = torch.load(teacher_model_path, map_location=device_torch)
        teacher_model.load_state_dict(state_dict)
        teacher_model.eval()
        
        with torch.no_grad():
            query_generator, data_generator = temp_qd.generate_teacher_inputs(data_graph_path, query_load_path)
            pred_count = teacher_model(query_generator, data_generator, device_torch).item()
            soft_card = pred_count
        
        # 序列化图
        graph_key = temp_qd.serialize_graph(query)
        
        return {
            'idx': idx,
            'query': query,
            'graphs': graphs,
            'true_card': true_card,
            'soft_card': soft_card,
            'label_den': label_den,
            'graph_key': graph_key,
            'query_id': (pattern, size, idx)
        }
        
    except Exception as e:
        print(f"Error processing query {query_load_path}: {str(e)}")
        return None

class QueryDecompose(object):
	def __init__(self, queryset_dir: str, true_card_dir: str, dataset: str, pattern: str,teacher_model_path: str, k = 3, size = 4, device='cuda:0'):
		"""
		load the query graphs, true counts and perform query decomposition
		"""
		self.queryset = queryset_dir
		self.dataset = dataset
		self.queryset_load_path = os.path.join(queryset_dir, dataset)
		self.true_card_dir = true_card_dir
		self.true_card_load_path = os.path.join(true_card_dir, dataset)
		self.k = k
		self.pattern = pattern
		self.size = size
		self.teacher_model_path = teacher_model_path
		self.device = device
		self.edge_embeds = np.load(os.path.join("data/prone", dataset+"_edge.emb.npy"))
		map_path = os.path.join("data/prone", dataset+"_edge_index_map.json")
		with open(map_path, "r") as f:
			self.edge_index_map = json.load(f)
		self.data_graph_path = os.path.join("data/dataset", dataset, dataset+".txt")
		self.num_queries = 0
		self.all_subsets = {} # {(size, patten) -> [(decomp_graphs, true_card, softcard]}
		# preserve the undecomposed queries
		self.all_queries = {} # {(size, patten) -> [(graph, card, softcard)]}
		self.train_graphs = {} # 存储训练查询图和预测计数
		self.query_indices = {}  # 存储查询索引到原始query的映射
		self.lower_card = 10 ** 0
		self.upper_card = 10 ** 20

	def decompose_queries_parallel(self, num_workers=None):
		"""
		并行版本的查询分解方法
		"""
		if num_workers is None:
			num_workers = min(cpu_count(), 8)  # 限制最大进程数
		
		print(f"Starting parallel query decomposition with {num_workers} workers...")
		start_time = time.time()
		
		# 显示系统信息
		
		
		cpu_count_actual = cpu_count()
		memory_gb = psutil.virtual_memory().total / (1024**3)
		print(f"System info: {cpu_count_actual} CPUs, {memory_gb:.1f} GB RAM")
		print(f"Using {num_workers} worker processes")
		
		# 检查CUDA可用性
		if torch.cuda.is_available():
			print(f"CUDA available: {torch.cuda.get_device_name(0)}")
		else:
			print("CUDA not available, using CPU")
		
		# 验证模型文件
		if not os.path.exists(self.teacher_model_path):
			logging.error(f'Teacher model file not found: {self.teacher_model_path}')
			raise FileNotFoundError(f'Teacher model file not found: {self.teacher_model_path}')
		
		queries_dir = os.path.join(self.queryset_load_path, self.pattern+'_'+str(self.size))
		self.all_subsets[(self.pattern, self.size)] = []
		self.all_queries[(self.pattern, self.size)] = []
		
		# 准备并行处理的参数
		query_files = []
		for idx, query_dir in enumerate(os.listdir(queries_dir)):
			query_load_path = os.path.join(self.queryset_load_path, self.pattern+'_'+str(self.size), query_dir)
			card_load_path = os.path.join(self.true_card_load_path, self.pattern+'_'+str(self.size), query_dir)
			
			if not os.path.isfile(query_load_path) or os.path.splitext(query_load_path)[1] == ".pickle":
				continue
			
			query_files.append((query_load_path, card_load_path, idx))
		
		print(f"Found {len(query_files)} query files to process")
		
		# 准备并行处理的参数
		args_list = []
		for query_load_path, card_load_path, idx in query_files:
			args_tuple = (
				query_load_path, card_load_path, self.data_graph_path, self.teacher_model_path,
				self.queryset_load_path, self.true_card_load_path, self.pattern, self.size, idx,
				self.edge_embeds, self.edge_index_map, self.device, self.upper_card, self.lower_card, self.dataset
			)
			args_list.append(args_tuple)
		
		# 并行处理
		print(f"Processing {len(args_list)} queries in parallel...")
		with Pool(processes=num_workers) as pool:
			# 使用tqdm显示进度
			results = list(tqdm(
				pool.imap(process_single_query_parallel, args_list),
				total=len(args_list),
				desc="Processing queries"
			))
		
		# 收集结果
		avg_label_den = 0.0
		valid_results = [r for r in results if r is not None]
		
		for result in valid_results:
			idx = result['idx']
			query = result['query']
			graphs = result['graphs']
			true_card = result['true_card']
			soft_card = result['soft_card']
			label_den = result['label_den']
			graph_key = result['graph_key']
			query_id = result['query_id']
			
			avg_label_den += label_den
			
			self.all_subsets[(self.pattern, self.size)].append((graphs, true_card, soft_card))
			self.all_queries[(self.pattern, self.size)].append((query, true_card, soft_card))
			
			# 存储训练查询图（序列化后）
			self.train_graphs[graph_key] = {"graph": query, "pred": None}
			self.query_indices[(self.pattern, self.size, idx)] = query
			self.num_queries += 1
		
		end_time = time.time()
		print(f"Parallel processing completed in {end_time - start_time:.2f} seconds")
		print("num_queries", self.num_queries)
		
		if self.num_queries > 0:
			print("average label density: {}".format(avg_label_den/self.num_queries))
		else:
			print("Warning: No queries were successfully processed!")
			print(f"Total query files found: {len(query_files)}")
			print(f"Valid results: {len(valid_results)}")
			print("This suggests there may be an error in the parallel processing.")
			raise RuntimeError("No queries were successfully processed. Check the error messages above for details.")

	def decompose_queries(self):
		# 验证模型文件
		if not os.path.exists(self.teacher_model_path):
			logging.error(f'Teacher model file not found: {self.teacher_model_path}')
			raise FileNotFoundError(f'Teacher model file not found: {self.teacher_model_path}')
		    # 加载教师模型
		device = torch.device(self.device if torch.cuda.is_available() else 'cpu')
		teacher_model = Flow_Learner(out_dim=128).to(device)
		try:
			state_dict = torch.load(self.teacher_model_path, map_location=device)
			teacher_model.load_state_dict(state_dict)
			logging.info(f'Teacher model loaded successfully from {self.teacher_model_path}')
			logging.info(f'Sample parameter: {next(iter(teacher_model.parameters())).data[:5]}')
		except Exception as e:
			logging.error(f'Failed to load teacher model: {str(e)}')
			raise
		teacher_model.eval()
		avg_label_den = 0.0
		distinct_card = {}
		queries_dir = os.path.join(self.queryset_load_path, self.pattern+'_'+str(self.size))
		self.all_subsets[(self.pattern, self.size)] = []
		self.all_queries[(self.pattern, self.size)] = []
		for idx, query_dir in enumerate(os.listdir(queries_dir)):
			query_load_path = os.path.join(self.queryset_load_path, self.pattern+'_'+str(self.size), query_dir)
			card_load_path = os.path.join(self.true_card_load_path, self.pattern+'_'+str(self.size), query_dir)
			if not os.path.isfile(query_load_path) or os.path.splitext(query_load_path)[1] == ".pickle":
				continue
			# load, decompose the query
			query, label_den = self.load_query(query_load_path)
			avg_label_den += label_den
			graphs = self.decompose(query)
			true_card = self.load_card(card_load_path)
			if true_card >=  self.upper_card or true_card < self.lower_card:
				continue
			true_card = true_card + 1 if true_card == 0 else true_card
			
			with torch.no_grad():
				query_generator, data_generator = self.generate_teacher_inputs(self.data_graph_path,query_load_path)
				pred_count = teacher_model(query_generator, data_generator, self.device).item()
				soft_card = pred_count
				# soft_label = int(np.ceil(np.log10(pred_count + 1e-8)))  # Classification label
			query_id = (self.pattern, self.size, idx)
			self.all_subsets[(self.pattern, self.size)].append((graphs, true_card, soft_card))
			self.all_queries[(self.pattern, self.size)].append((query, true_card, soft_card))
			# 存储训练查询图（序列化后）
			graph_key = self.serialize_graph(query)
			self.train_graphs[graph_key] = {"graph": query, "pred": None}  # pred 在训练时更新
			self.query_indices[(self.pattern, self.size, idx)] = query  # 存储索引到query的映射
			self.num_queries += 1
			# save the decomposed query
			#query_save_path = os.path.splitext(query_load_path)[0] + ".pickle"
			#self.save_decomposed_query(graphs, true_card, query_save_path)
			#print("save decomposed query: {}".format(query_save_path))
		print("num_queries", self.num_queries)
		
		if self.num_queries > 0:
			print("average label density: {}".format(avg_label_den/self.num_queries))
		else:
			print("Warning: No queries were successfully processed!")
			raise RuntimeError("No queries were successfully processed in sequential mode.")

	def serialize_graph(self, g):
		"""生成图的唯一序列化表示"""
		nodes = sorted(g.nodes(data=True), key=lambda x: x[0])
		edges = sorted(g.edges(data=True), key=lambda x: (x[0], x[1]))
		node_str = ";".join(f"{n[0]}:{','.join(map(str, n[1]['labels']))}" for n in nodes)
		edge_str = ";".join(f"{e[0]}:{e[1]}:{','.join(map(str, e[2]['labels']))}" for e in edges)
		return f"{node_str}|{edge_str}"
	
	def is_isomorphic(self, g1, g2):
		"""检查两个图是否同构，考虑标签列表"""
		def node_match(n1, n2):
			return n1.get('labels', []) == n2.get('labels', [])
		def edge_match(e1, e2):
			return e1.get('labels', []) == e2.get('labels', [])
		return nx.is_isomorphic(g1, g2, node_match=node_match, edge_match=edge_match)
	
	def get_train_prediction(self, query):
		"""查找训练集中同构图的预测计数"""
		query_key = self.serialize_graph(query)
		if query_key in self.train_graphs and self.train_graphs[query_key]["pred"] is not None:
			return self.train_graphs[query_key]["pred"]
		for train_key, train_data in self.train_graphs.items():
			if self.is_isomorphic(query, train_data["graph"]):
				self.train_graphs[query_key] = train_data  # 缓存序列化
				return train_data["pred"]
		return None

	def update_train_prediction(self, graph, pred):
		"""更新训练图的预测计数"""
		graph_key = self.serialize_graph(graph)
		if graph_key in self.train_graphs:
			self.train_graphs[graph_key]["pred"] = pred
	
	def get_query_by_index(self, pattern, size, idx):
		"""根据索引获取原始查询图"""
		return self.query_indices.get((pattern, size, idx))

	def decompose(self, query):
		graphs = []
		for src in query.nodes():
			G = self.k_hop_spanning_tree(query, src)
			graphs.append(G)
		return graphs

	def k_hop_spanning_tree(self, query, src):
		nodes_list = [src]
		edges_list = []
		Q = queue.Queue()
		Q.put(src)
		depth = 0
		while not Q.empty():
			s = Q.qsize()
			for _ in range(s):
				cur = Q.get()
				for next in query.neighbors(cur):
					if next in nodes_list:
						continue
					Q.put(next)
					nodes_list.append(next)
					edges_list.append((cur, next))
			depth += 1
			if depth >= self.k:
				break

		G = self.node_reorder(query, nodes_list, edges_list)
		return G

	def k_hop_induced_subgraph(self, query, src):
		nodes_list = [src]
		Q = queue.Queue()
		Q.put(src)
		depth = 0
		while not Q.empty():
			s = Q.qsize()
			for _ in range(s):
				cur = Q.get()
				for next in query.neighbors(cur):
					if next in nodes_list:
						continue
					Q.put(next)
					nodes_list.append(next)
			depth += 1
			if depth >= self.k:
				break
		edges_list = query.subgraph(nodes_list).edges()
		G = self.node_reorder(query, nodes_list, edges_list)
		return G

	def node_reorder(self, query, nodes_list, edges_list):
		idx_dict = {}
		node_cnt = 0
		for v in nodes_list:
			idx_dict[v] = node_cnt
			node_cnt += 1
		nodes_list = [(idx_dict[v], {"labels": query.nodes[v]["labels"]})
					  for v in nodes_list]
		edges_list = [(idx_dict[u], idx_dict[v], {"labels": query.edges[u, v]["labels"]})
					  for (u, v) in edges_list]
		sample = nx.Graph()
		sample.add_nodes_from(nodes_list)
		sample.add_edges_from(edges_list)
		return sample

	def load_query(self, query_load_path):
		file = open(query_load_path)
		nodes_list = []
		edges_list = []
		label_cnt = 0
		node_label_dict = dict()

		for line in file:
			if line.strip().startswith("v"):
				tokens = line.strip().split()
				# v nodeID labelID
				id = int(tokens[1])
				tmp_labels = [int(tokens[2])] # (only one label in the query node)
				label = int(tmp_labels[0])
				#tmp_labels = [int(token) for token in tokens[2 : ]]
				labels = [] if -1 in tmp_labels else tmp_labels
				label_cnt += len(labels)
				nodes_list.append((id, {"labels": labels}))
				node_label_dict[id] = label

			if line.strip().startswith("e"):
				# e srcID dstID labelID1 labelID2....
				tokens = line.strip().split()
				src, dst = int(tokens[1]), int(tokens[2])
				key = (node_label_dict[src], node_label_dict[dst])
				# print("key:",key)
				if str(key) in self.edge_index_map:
					edge_label = self.edge_index_map[str(key)]
				else:
					edge_label = [int(tokens[3])]
				tmp_labels = edge_label
				# tmp_labels = [int(tokens[3])]
				#tmp_labels = [int(token) for token in tokens[3 : ]]
				labels = [] if -1 in tmp_labels else tmp_labels
				edges_list.append((src, dst, {"labels": labels}))

		query = nx.Graph()
		query.add_nodes_from(nodes_list)
		query.add_edges_from(edges_list)

		file.close()
		label_den = float(label_cnt) / query.number_of_nodes()
		return query, label_den
	
	def generate_teacher_inputs(self, data_graph_path, query_graph_path):

		# 调用 Find_Candidates
		finder = Find_Candidates(data_graph_path, query_graph_path)
		output_q_info, output_g_info = finder.cpp_GQL()
			
		cs_size, candidates_size, g_nodes, g_edges = output_g_info
		q_vertices, q_labels, q_degrees, q_edges= output_q_info

		root = cs_size.index(max(cs_size))
		layers = bfs_layers(q_edges, root)             

		query_adj = generate_query_adj(q_edges, len(q_vertices))
		query_generator = query_flow(query_adj, q_labels, layers)

		adj = generate_adjacency_matrix(g_edges, candidates_size)
		data_generator = data_flow(q_labels, g_nodes, layers, root, adj)
		return query_generator, data_generator

	def load_card(self, card_load_path):
		with open(card_load_path, "r") as in_file:
			card = in_file.readline().strip()
			card = int(card)
			in_file.close()
		return card

	def save_decomposed_query(self, graphs, card, save_path):
		with open(save_path, "wb") as out_file:
			obj = {"graphs": graphs, "card": card}
			pickle.dump(obj = obj, file=out_file, protocol=3)
			out_file.close()



def get_true_cardinality(card_estimator_path, query_load_path, graph_load_path, timeout_sec = 7200):
	
	est_path = os.path.join(card_estimator_path, "SubgraphMatching.out")
	cmd = "timeout %d %s -d %s -q %s -filter GQL -order GQL -engine LFTJ -num MAX"\
		  %(timeout_sec, est_path, graph_load_path, query_load_path)
	print(cmd)
	popen = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=True)
	popen.wait()
	card, run_time = None, None
	for line in iter(popen.stdout.readline, b''):
		line = line.decode("utf-8")
		# if line.startswith("#Embeddings:"):
		# 	card = line.partition("#Embeddings:")[-1].strip()
		if line.startswith("Call Count:"):
			card = line.partition("Call Count:")[-1].strip()
		elif line.startswith("Enumerate time (seconds):"):
			run_time = line.partition("Enumerate time (seconds):")[-1].strip()
	return card, run_time


def get_batch_true_card(card_estimator_path, queries_load_path, graph_load_path, card_save_dir):
	queries_dir = os.listdir(queries_load_path)
	for query_dir in queries_dir:
		query_load_path = os.path.join(queries_load_path, query_dir)
		pattern, size, = str(query_dir.split("_")[1]), str(query_dir.split("_")[2])

		card, run_time = get_true_cardinality(card_estimator_path, query_load_path, graph_load_path)
		if card is not None:
			card_save_path = os.path.join(card_save_dir, '_'.join([pattern, size]))
			make_dir(card_save_path)
			card_save_path = os.path.join(card_save_path, os.path.splitext(query_dir)[0] + '.txt')
			save_true_card(card, card_save_path, run_time)
			print("save card {} in {}".format(card, card_save_path))


def get_save_true_card(card_estimator_path, queries_load_path, graph_load_path, card_save_dir, query_dir):
	query_load_path = os.path.join(queries_load_path, query_dir)
	pattern, size, = str(query_dir.split("_")[1]), str(query_dir.split("_")[2])

	card, run_time = get_true_cardinality(card_estimator_path, query_load_path, graph_load_path)
	if card is not None:
		card_save_path = os.path.join(card_save_dir, '_'.join([pattern, size]))
		make_dir(card_save_path)
		card_save_path = os.path.join(card_save_path, os.path.splitext(query_dir)[0] + '.txt')
		save_true_card(card, card_save_path, run_time)
		print("save card {} in {}".format(card, card_save_path))


def process_batch_true_card(card_estimator_path, queries_load_path, graph_load_path, card_save_dir, num_workers = 10):
	"""
	parallel version of get_batch_true_card
	"""
	pro = functools.partial(get_save_true_card, card_estimator_path,
											 graph_load_path, card_save_dir)
	queries_dir = os.listdir(queries_load_path)
	p = Pool(num_workers)
	p.map(pro, queries_dir)