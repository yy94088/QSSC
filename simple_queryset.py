import os
import torch
import networkx as nx
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from util import make_dir  # 复用util.py中的make_dir
import math

class SimpleQueryset(object):
    def __init__(self, args, all_queries):
        """
        all_queries: {(pattern, size) -> [(query, true_card, soft_card)]}
        """
        self.args = args
        self.data_dir = args.data_dir
        self.dataset = args.dataset
        self.data_graph = DataGraph(data_dir=self.data_dir, dataset=self.dataset)
        self.node_label_card = self.data_graph.node_label_card
        self.edge_label_card = self.data_graph.edge_label_card
        self.num_queries = 0

        self.num_nodes = self.data_graph.num_nodes
        self.num_edges = self.data_graph.num_edges
        self.node_label_fre = 0
        self.edge_label_fre = 0


        self.label_dict = self.load_label_mapping() if self.dataset == 'hprd' or self.dataset == 'hprd_gen' else \
			{key: key for key in self.node_label_card.keys()}  # label_id -> embed_id
        embed_feat_path = os.path.join(args.embed_feat_dir, "{}.emb.npy".format(args.dataset))
        embed_feat = np.load(embed_feat_path)

		#assert embed_feat.shape[0] == len(self.node_label_card) + 1, "prone embedding size error!"
        self.embed_dim = embed_feat.shape[1]
        self.embed_feat = torch.from_numpy(embed_feat)
        if self.args.embed_type == "freq":
            self.num_node_feat = len(self.node_label_card)
        elif self.args.embed_type == "n2v" or self.args.embed_type == "prone" or self.args.embed_type == "nrp":
             self.num_node_feat = self.embed_dim
        else:
            self.num_node_feat = self.embed_dim + len(self.node_label_card)

        if self.args.edge_embed_type == "freq":
            self.edge_embed_feat = None
            self.edge_embed_dim = 0
            self.num_edge_feat = len(self.edge_label_card)
        else:
            edge_embed_feat_path = os.path.join(args.embed_feat_dir, "{}_edge.emb.npy".format(args.dataset))
            edge_embed_feat = np.load(edge_embed_feat_path)
            self.edge_embed_dim = edge_embed_feat.shape[1]
            self.edge_embed_feat = torch.from_numpy(edge_embed_feat)
            self.num_edge_feat = self.edge_embed_dim

        # 转换查询图为张量
        self.queries = self.transform_query_to_tensors(all_queries)

        # 分裂数据集
        self.all_query = []
        self.all_queries = {}
        self.all_patterns = {}
        for (pattern, size), graphs_card_pairs in self.queries.items():
            self.all_query += graphs_card_pairs
            if pattern not in self.all_patterns.keys():
                self.all_patterns[pattern] = []
            self.all_patterns[pattern] += graphs_card_pairs
            if size not in self.all_queries.keys():
                self.all_queries[size] = []
            self.all_queries[size] += graphs_card_pairs

        train_sets, val_sets, test_sets, all_train_sets = self.data_split(self.all_queries)
        self.train_loaders = self.to_dataloader(train_sets)
        self.val_loaders = self.to_dataloader(val_sets)
        self.test_loaders = self.to_dataloader(test_sets)
        self.train_sets, self.val_sets, self.test_sets = train_sets, val_sets, test_sets

    def data_split(self, all_sets, train_ratio=0.8, val_ratio=0.1, seed=1):
        random.seed(seed)
        train_sets, val_sets, test_sets = [], [], []
        all_train_sets = [[]]
        for key in sorted(all_sets.keys()):
            num_instances = len(all_sets[key])
            random.shuffle(all_sets[key])
            train_sets.append(all_sets[key][:int(num_instances * train_ratio)])
            all_train_sets[-1] += train_sets[-1]
            val_sets.append(all_sets[key][int(num_instances * train_ratio):int(num_instances * (train_ratio + val_ratio))])
            test_sets.append(all_sets[key][int(num_instances * (train_ratio + val_ratio)):])
        return train_sets, val_sets, test_sets, all_train_sets

    def to_dataloader(self, all_sets, batch_size=1, shuffle=True):
        datasets = [QueryDataset(queries=queries) for queries in all_sets]
        return [DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle) for dataset in datasets]

    def transform_query_to_tensors(self, all_queries):
        tmp_queries = {}
        for (pattern, size), graphs_card_pairs in all_queries.items():
            tmp_queries[(pattern, size)] = []
            for (query, card, _) in graphs_card_pairs:
                x, edge_index, edge_attr = self._get_graph_data(query)
                tmp_queries[(pattern, size)].append((x, edge_index, edge_attr, card))
        return tmp_queries

    def _get_graph_data(self, graph):
        if self.args.embed_type == "freq":
            node_attr = self._get_nodes_attr_freq(graph)
        elif self.args.embed_type == "n2v" or self.args.embed_type == "prone" or self.args.embed_type == "nrp":
               node_attr = self._get_nodes_attr_embed(graph)
        else:
               node_attr_freq, node_attr_embed = self._get_nodes_attr_freq(graph), self._get_nodes_attr_embed(graph)
               node_attr = torch.cat([node_attr_freq, node_attr_embed], dim=1)

        if self.args.edge_embed_type == "freq":
               edge_index, edge_attr = self._get_edges_index_freq(graph)
        else:
               edge_index, edge_attr = self._get_edges_index_embed(graph)
        return node_attr, edge_index, edge_attr
    
    def _get_nodes_attr_freq(self, graph):
        node_attr = torch.ones(size=(graph.number_of_nodes(), len(self.node_label_card)), dtype=torch.float)
        for v in graph.nodes():
            for label in graph.nodes[v]["labels"]:
                node_attr[v][self.label_dict[label]] = self.node_label_card[label]
                self.node_label_fre += 1
        return node_attr

    def _get_nodes_attr_embed(self, graph):
        node_attr = torch.zeros(size=(graph.number_of_nodes(), self.embed_dim), dtype=torch.float)
        for v in graph.nodes():
            if len(graph.nodes[v]["labels"]) == 0:
                continue
            for label in graph.nodes[v]["labels"]:
                node_attr[v] += self.embed_feat[self.label_dict[label]]
                self.node_label_fre += 1
        return node_attr


    def _get_edges_index_freq(self, graph):
        edge_index = torch.ones(size= (2, graph.number_of_edges()), dtype = torch.long)
        edge_attr = torch.zeros(size= (graph.number_of_edges(), len(self.edge_label_card)), dtype=torch.float)
        cnt = 0
        for e in graph.edges():
            edge_index[0][cnt], edge_index[1][cnt] = e[0], e[1]
            for label in graph.edges[e]["labels"]:
                edge_attr[cnt][label] = self.edge_label_card[label]
                self.edge_label_fre += 1
            cnt += 1
        return edge_index, edge_attr

    def _get_edges_index_embed(self, graph):
        edge_index = torch.ones(size= (2, graph.number_of_edges()), dtype = torch.long)
        edge_attr = torch.zeros(size= (graph.number_of_edges(), self.edge_embed_dim), dtype=torch.float)
        cnt = 0
        for e in graph.edges():
            edge_index[0][cnt], edge_index[1][cnt] = e[0], e[1]
            for label in graph.edges[e]["labels"]:
                edge_attr[cnt] += self.edge_embed_feat[label]
                self.edge_label_fre += 1
            cnt += 1
        return edge_index, edge_attr

    def load_label_mapping(self):
        map_load_path = os.path.join(self.args.embed_feat_dir, "{}_mapping.txt".format(self.dataset))
        assert os.path.exists(map_load_path), "The label mapping file is not exists!"
        label_dict = {} # label_id -> embed_id
        cnt = 0
        with open(map_load_path, "r") as in_file:
            for line in in_file:
                label_id = int(line.strip())
                label_dict[label_id] = cnt
                cnt += 1
            in_file.close()
        return label_dict

    def print_queryset_info(self):
        print("<" * 80)
        print("Query Set Profile:")
        print("# Total Queries: {}".format(self.num_queries))
        print("# Train Queries: {}".format(self.num_train_queries))
        print("# Val Queries: {}".format(self.num_val_queries))
        print("# Test Queries: {}".format(self.num_test_queries))
        print("# Node Feat: {}".format(self.num_node_feat))
        print("# Edge Feat: {}".format(self.num_edge_feat))
        print("# Node label fre: {}".format(self.node_label_fre))
        print("# Edge label fre: {}".format(self.edge_label_fre))
        print(">" * 80)

class DataGraph(object):  # 复用Queryset.py中的DataGraph
    def __init__(self, data_dir, dataset):
        self.data_dir = data_dir
        self.data_set = dataset
        self.data_load_path = os.path.join(data_dir, dataset, f"{dataset}.txt")
        self.graph, self.node_label_card, self.edge_label_card = self.load_graph()
        self.num_nodes = self.graph.number_of_nodes()
        self.num_edges = self.graph.number_of_edges()
        

    def load_graph(self):
        file = open(self.data_load_path)
        nodes_list = []
        edges_list = []
        node_label_card = {}
        edge_label_card = {}
        edge_index = 0
        for line in file:
            if line.strip().startswith("v"):
                tokens = line.strip().split()
                id = int(tokens[1])
                labels = [int(token) for token in tokens[2:]]
                for label in labels:
                    node_label_card[label] = node_label_card.get(label, 0) + 1.0
                nodes_list.append((id, {"labels": labels}))

            if line.strip().startswith("e"):
                tokens = line.strip().split()
                src, dst = int(tokens[1]), int(tokens[2])
                labels = [edge_index]  # 每条边按顺序编号作为标签
				# labels = [0]
				#labels = [] if -1 in tmp_labels else tmp_labels
                for label in labels:
                    if label not in edge_label_card.keys():
                        edge_label_card[label] = 1.0
                    else:
                        edge_label_card[label] += 1.0
                edges_list.append((src, dst, {"labels": labels}))
                edge_index += 1
                    
        graph = nx.Graph()
        graph.add_nodes_from(nodes_list)
        graph.add_edges_from(edges_list)
        file.close()
        return graph, node_label_card, edge_label_card

class QueryDataset(Dataset):
    def __init__(self, queries, num_classes = 10):
        self.queries = queries
        self.num_classes = num_classes
        self.label_base = 10

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, index):
        x, edge_index, edge_attr, card = self.queries[index]
        idx = math.ceil(math.log(card, self.label_base))
        idx = self.num_classes - 1 if idx >= self.num_classes else idx
        card = torch.tensor(math.log2(card + 1e-8), dtype=torch.float)  # 对数基数
        return x, edge_index, edge_attr, card