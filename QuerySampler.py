import os
import numpy as np
import networkx as nx
import queue
from multiprocessing import Pool, cpu_count
import torch
import json
import time
import psutil
from tqdm import tqdm


def serialize_query_graph(query):
	"""Serialize original query graph to JSON-friendly structure."""
	nodes = []
	for nid, attrs in query.nodes(data=True):
		labels = attrs.get("labels", [])
		nodes.append({"id": int(nid), "labels": [int(x) for x in labels]})

	edges = []
	for src, dst, attrs in query.edges(data=True):
		labels = attrs.get("labels", [])
		edges.append({"src": int(src), "dst": int(dst), "labels": [int(x) for x in labels]})

	return {
		"num_nodes": int(query.number_of_nodes()),
		"num_edges": int(query.number_of_edges()),
		"nodes": nodes,
		"edges": edges,
	}


def process_single_query_parallel(args_tuple):
	"""
	并行处理单个查询的函数
	"""
	(query_load_path, card_load_path, data_graph_path, queryset_load_path, true_card_load_path, pattern, size, idx,edge_embeds, edge_index_map, upper_card, lower_card, dataset) = args_tuple

	try:
		# 用 dataset 参数初始化 QueryDecompose
		temp_qd = QueryDecompose(
			queryset_dir=queryset_load_path.replace(f"/{dataset}", ""),
			true_card_dir=true_card_load_path.replace(f"/{dataset}", ""),
			dataset=dataset,
			pattern='query',  # Fixed to 'query'
			k=3,
			size=size
		)
		temp_qd.edge_embeds = edge_embeds
		temp_qd.edge_index_map = edge_index_map
		temp_qd.upper_card = upper_card
		temp_qd.lower_card = lower_card

		# 加载查询
		query, label_den = temp_qd.load_query(query_load_path)
		graphs, subgraph_distances = temp_qd.decompose(query)
		true_card, soft_card = temp_qd.load_card(card_load_path)

		if true_card >= upper_card or true_card < lower_card:
			return None

		true_card = true_card + 1 if true_card == 0 else true_card

		return {
			'idx': idx,
			'query': query,
			'graphs': graphs,
			'subgraph_distances': subgraph_distances,
			'true_card': true_card,
			'soft_card': soft_card,
			'label_den': label_den,
			'query_id': (pattern, size, idx),
			'file_name': os.path.basename(query_load_path),
			'query_load_path': query_load_path,
			'card_load_path': card_load_path,
			'query_graph': serialize_query_graph(query),
		}

	except Exception as e:
		print(f"Error processing query {query_load_path}: {str(e)}")
		return None


class QueryDecompose(object):
	def __init__(self, queryset_dir: str, true_card_dir: str, dataset: str, pattern: str = 'query', k = 3, size = 4):
		"""
		load the query graphs, true counts and perform query decomposition
		"""
		self.queryset = queryset_dir
		self.dataset = dataset
		self.queryset_load_path = os.path.join(queryset_dir, dataset)
		self.true_card_dir = true_card_dir
		self.true_card_load_path = os.path.join(true_card_dir, dataset)
		self.k = k
		# Fixed pattern to 'query'
		self.pattern = 'query'
		self.size = size
		self.edge_embeds = np.load(os.path.join("data/prone", dataset+"_edge.emb.npy"))
		map_path = os.path.join("data/prone", dataset+"_edge_index_map.json")
		with open(map_path, "r") as f:
			self.edge_index_map = json.load(f)
		self.data_graph_path = os.path.join("data/dataset", dataset, dataset+".txt")
		self.num_queries = 0
		self.all_subsets = {} # {(size, patten) -> [(decomp_graphs, true_card, soft_card]}
		# preserve the undecomposed queries
		self.all_queries = {} # {(size, patten) -> [(graph, card, softcard)]}
		self.query_indices = {}  # 存储查询索引到原始query的映射
		self.query_file_names = {}  # 存储查询索引到文件名的映射
		
		self.lower_card = 10 ** 0
		self.upper_card = 10 ** 20

	def _resolve_size_dirs(self):
		"""Return a list of (size, query_dir, card_dir) to process."""
		size_dirs = []
		if self.size == -1:
			for entry in sorted(os.listdir(self.queryset_load_path)):
				query_dir = os.path.join(self.queryset_load_path, entry)
				if not os.path.isdir(query_dir):
					continue
				prefix = self.pattern + '_'
				if not entry.startswith(prefix):
					continue
				try:
					size_val = int(entry[len(prefix):])
				except ValueError:
					continue
				card_dir = os.path.join(self.true_card_load_path, entry)
				if os.path.isdir(card_dir):
					size_dirs.append((size_val, query_dir, card_dir))
		else:
			dir_name = self.pattern + '_' + str(self.size)
			query_dir = os.path.join(self.queryset_load_path, dir_name)
			card_dir = os.path.join(self.true_card_load_path, dir_name)
			size_dirs.append((self.size, query_dir, card_dir))
		return size_dirs

	def decompose_queries_parallel(self, num_workers=None):
		"""
		并行版本的查询分解方法
		"""
		if num_workers is None:
			num_workers = min(cpu_count(), 8)  # 限制最大进程数
		
		print(f"Starting parallel query decomposition with {num_workers} workers...")
		start_time = time.time()
		
		cpu_count_actual = cpu_count()
		memory_gb = psutil.virtual_memory().total / (1024**3)
		print(f"System info: {cpu_count_actual} CPUs, {memory_gb:.1f} GB RAM")
		print(f"Using {num_workers} worker processes")
		
		# 检查CUDA可用性
		if torch.cuda.is_available():
			print(f"CUDA available: {torch.cuda.get_device_name(0)}")
		else:
			print("CUDA not available, using CPU")
		
		size_dirs = self._resolve_size_dirs()
		for size_val, _, _ in size_dirs:
			self.all_subsets[(self.pattern, size_val)] = []
			self.all_queries[(self.pattern, size_val)] = []
		
		# 准备并行处理的参数
		query_files = []
		for size_val, queries_dir, cards_dir in size_dirs:
			if not os.path.isdir(queries_dir) or not os.path.isdir(cards_dir):
				continue
			for idx, query_dir in enumerate(os.listdir(queries_dir)):
				query_load_path = os.path.join(queries_dir, query_dir)
				card_load_path = os.path.join(cards_dir, query_dir)
				
				if not os.path.isfile(query_load_path) or os.path.splitext(query_load_path)[1] == ".pickle":
					continue
				if not os.path.isfile(card_load_path):
					continue
				
				query_files.append((query_load_path, card_load_path, idx, size_val))
		
		print(f"Found {len(query_files)} query files to process")
		
		# 准备并行处理的参数
		args_list = []
		for query_load_path, card_load_path, idx, size_val in query_files:
			args_tuple = (
				query_load_path, card_load_path, self.data_graph_path, self.queryset_load_path, 
				self.true_card_load_path, self.pattern, size_val, idx,
				self.edge_embeds, self.edge_index_map, self.upper_card, self.lower_card, self.dataset
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
			subgraph_distances = result['subgraph_distances']
			true_card = result['true_card']
			soft_card = result['soft_card']
			label_den = result['label_den']
			query_id = result['query_id']
			size_val = query_id[1]
			file_name = result.get('file_name', f"{query_id[0]}_{query_id[1]}_{idx}")
			query_meta = {
				"pattern": str(query_id[0]),
				"size": int(query_id[1]),
				"local_idx": int(idx),
				"file_name": file_name,
				"query_load_path": result.get('query_load_path', ''),
				"card_load_path": result.get('card_load_path', ''),
				"query_graph": result.get('query_graph', None),
			}
			
			avg_label_den += label_den
			
			self.all_subsets[(self.pattern, size_val)].append((graphs, subgraph_distances, true_card, soft_card, query_meta))
			self.all_queries[(self.pattern, size_val)].append((query, true_card, soft_card))
			
			self.query_indices[(self.pattern, size_val, idx)] = query
			self.query_file_names[(self.pattern, size_val, idx)] = file_name
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
		avg_label_den = 0.0
		size_dirs = self._resolve_size_dirs()
		for size_val, queries_dir, cards_dir in size_dirs:
			if not os.path.isdir(queries_dir) or not os.path.isdir(cards_dir):
				continue
			self.all_subsets[(self.pattern, size_val)] = []
			self.all_queries[(self.pattern, size_val)] = []
			for idx, query_dir in enumerate(os.listdir(queries_dir)):
				query_load_path = os.path.join(queries_dir, query_dir)
				card_load_path = os.path.join(cards_dir, query_dir)
				if not os.path.isfile(query_load_path) or os.path.splitext(query_load_path)[1] == ".pickle":
					continue
				if not os.path.isfile(card_load_path):
					continue
				# load, decompose the query
				query, label_den = self.load_query(query_load_path)
				avg_label_den += label_den
				graphs, subgraph_distances = self.decompose(query)
				true_card, soft_card = self.load_card(card_load_path)
				if true_card >=  self.upper_card or true_card < self.lower_card:
					continue
				true_card = true_card + 1 if true_card == 0 else true_card
				
				query_id = (self.pattern, size_val, idx)
				query_meta = {
					"pattern": str(self.pattern),
					"size": int(size_val),
					"local_idx": int(idx),
					"file_name": query_dir,
					"query_load_path": query_load_path,
					"card_load_path": card_load_path,
					"query_graph": serialize_query_graph(query),
				}
				self.all_subsets[(self.pattern, size_val)].append((graphs, subgraph_distances, true_card, soft_card, query_meta))
				self.all_queries[(self.pattern, size_val)].append((query, true_card, soft_card))
				
				self.query_indices[(self.pattern, size_val, idx)] = query
				self.query_file_names[(self.pattern, size_val, idx)] = query_dir
				self.num_queries += 1
		
		print("num_queries", self.num_queries)
		
		if self.num_queries > 0:
			print("average label density: {}".format(avg_label_den/self.num_queries))
		else:
			print("Warning: No queries were successfully processed!")
			raise RuntimeError("No queries were successfully processed in sequential mode.")

	def decompose(self, query):
		graphs = []
		node_order = list(query.nodes())
		for src in node_order:
			G = self.k_hop_induced_subgraph(query, src)
			graphs.append(G)

		# Build subgraph-to-subgraph shortest-path distances on the original query graph.
		# The i-th subgraph is centered at node_order[i].
		subgraph_distances = self.compute_subgraph_distances(query, node_order)
		return graphs, subgraph_distances

	def compute_subgraph_distances(self, query, node_order):
		num_nodes = len(node_order)
		distances = np.zeros((num_nodes, num_nodes), dtype=np.int64)
		all_pairs = dict(nx.all_pairs_shortest_path_length(query))

		for i, src in enumerate(node_order):
			src_dists = all_pairs.get(src, {})
			for j, dst in enumerate(node_order):
				if i == j:
					continue
				# Use a large finite distance for disconnected pairs to keep tensors numeric.
				distances[i, j] = int(src_dists.get(dst, num_nodes))

		return distances

	def k_hop_induced_subgraph(self, query, src):
		nodes_list = [src]
		q = queue.Queue()
		q.put(src)
		visited = {src}
		depth = 0
		
		head = 0
		while head < len(nodes_list) and depth < self.k:
			curr_node = nodes_list[head]
			head += 1
			for neighbor in query.neighbors(curr_node):
				if neighbor not in visited:
					visited.add(neighbor)
					nodes_list.append(neighbor)
			
			if head == len(nodes_list): # Finished a level
				depth += 1

		subgraph = query.subgraph(nodes_list)
		edges_list = list(subgraph.edges())
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
				labels = [] if -1 in tmp_labels else tmp_labels
				label_cnt += len(labels)
				nodes_list.append((id, {"labels": labels}))
				node_label_dict[id] = label

			if line.strip().startswith("e"):
				# e srcID dstID labelID1 labelID2....
				tokens = line.strip().split()
				src, dst = int(tokens[1]), int(tokens[2])
				key = (node_label_dict[src], node_label_dict[dst])
				if str(key) in self.edge_index_map:
					edge_label = self.edge_index_map[str(key)]
				else:
					edge_label = [int(tokens[3])]
				tmp_labels = edge_label
				labels = [] if -1 in tmp_labels else tmp_labels
				edges_list.append((src, dst, {"labels": labels}))

		query = nx.Graph()
		query.add_nodes_from(nodes_list)
		query.add_edges_from(edges_list)

		file.close()
		label_den = float(label_cnt) / query.number_of_nodes()
		return query, label_den
	
	def load_card(self, card_load_path):
		with open(card_load_path, "r") as in_file:
			true_card = float(in_file.readline().strip())
			soft_card = float(in_file.readline().strip())
			in_file.close()
		return true_card, soft_card
