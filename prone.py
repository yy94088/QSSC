import numpy as np
import argparse
import os
import networkx as nx
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
from collections import defaultdict
import logging
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Generate ProNE embeddings")
    parser.add_argument("--graph", type=str, required=True, help="Input graph file")
    parser.add_argument("--output", type=str, required=True, help="Output embedding file")
    parser.add_argument("--dim", type=int, default=128, help="Node embedding dimension")
    parser.add_argument("--edge_dim", type=int, default=16, help="Edge embedding dimension")
    parser.add_argument("--mapping_output", type=str, default=None, help="Output node label mapping file")
    return parser.parse_args()

def load_graph(graph_path, output_path, mapping_output):
    """Load graph: v nodeID label1 label2..., e srcID dstID label1 label2..."""
    G = nx.Graph()
    label_set = set()
    try:
        with open(graph_path, 'r') as f:
            for line in f:
                tokens = line.strip().split()
                if tokens[0] == 't':
                    continue
                elif tokens[0] == 'v':
                    node_id = int(tokens[1])
                    labels = [int(tokens[2])]  # Support multiple labels
                    G.add_node(node_id, labels=labels)
                    label_set.update(labels)
                elif tokens[0] == 'e':
                    src, dst = map(int, tokens[1:3])
                    labels = [int(token) for token in tokens[3:]] or [0]
                    G.add_edge(src, dst, labels=labels)
        
        # Generate label mapping
        label_dict = {label: idx for idx, label in enumerate(sorted(label_set))}
        if mapping_output:
            mapping_path = mapping_output
        else:
            output_dir = os.path.dirname(output_path)
            dataset = os.path.splitext(os.path.basename(output_path))[0].replace(".emb", "")
            mapping_path = os.path.join(output_dir, f"{dataset}_mapping.txt")
        with open(mapping_path, 'w') as f:
            for label in sorted(label_set):
                f.write(f"{label}\n")
        logging.info(f"Node label mapping saved to {mapping_path}, labels: {len(label_set)}")
        logging.info(f"Label dict sample: {dict(list(label_dict.items())[:5])}")
        
        logging.info(f"Graph info: nodes={G.number_of_nodes()}, edges={G.number_of_edges()}, unique labels={len(label_set)}")
        return G, label_dict
    except Exception as e:
        logging.error(f"Error loading graph or saving mapping: {e}")
        exit(1)

def generate_node_embeddings(G, label_dict, dim):
    """Generate node embeddings for unique labels using simplified ProNE"""
    try:
        # Convert to adjacency matrix
        adj_matrix = nx.to_scipy_sparse_array(G, nodelist=sorted(G.nodes()), dtype=np.float32)
        num_nodes = G.number_of_nodes()
        
        # Compute Laplacian matrix
        laplacian = csgraph.laplacian(adj_matrix, normed=True)
        
        # Spectral decomposition: compute top-k eigenvectors
        k = min(32, num_nodes - 1)  # Reduce k for stability
        try:
            _, eigenvectors = eigsh(laplacian, k=k, which='SM', maxiter=5000, tol=1e-4, ncv=2*k+1)
            node_embeds = eigenvectors[:, :k]
        except Exception as e:
            logging.warning(f"eigsh failed: {e}. Falling back to random embeddings.")
            node_embeds = np.random.rand(num_nodes, k).astype(np.float32)
        
        # Pad to ensure at least dim columns
        if node_embeds.shape[1] < dim:
            pad_width = ((0, 0), (0, dim - node_embeds.shape[1]))
            node_embeds = np.pad(node_embeds, pad_width, mode='constant')
        elif node_embeds.shape[1] > dim:
            node_embeds = node_embeds[:, :dim]
        
        # Normalize node embeddings
        node_embeds = node_embeds / (np.linalg.norm(node_embeds, axis=1, keepdims=True) + 1e-10)
        
        # Generate label embeddings by averaging node embeddings with same label
        label_embeds = np.zeros((len(label_dict), dim), dtype=np.float32)
        label_counts = np.zeros(len(label_dict), dtype=np.int32)
        for node in G.nodes():
            labels = G.nodes[node].get('labels', [])
            for label in labels:
                idx = label_dict[label]
                label_embeds[idx] += node_embeds[node]
                label_counts[idx] += 1
        label_embeds = label_embeds / (label_counts[:, np.newaxis] + 1e-10)
        
        # Normalize label embeddings
        label_embeds = label_embeds / (np.linalg.norm(label_embeds, axis=1, keepdims=True) + 1e-10)
        
        # Verify mapping alignment
        for label, idx in list(label_dict.items())[:5]:
            logging.info(f"Label {label} mapped to index {idx}, embed norm: {np.linalg.norm(label_embeds[idx])}")
        
        logging.info(f"Generated label embeddings, shape={label_embeds.shape}, mean norm={np.linalg.norm(label_embeds, axis=1).mean()}")
        return label_embeds
    except Exception as e:
        logging.error(f"Error generating node embeddings: {e}")
        exit(1)

def generate_edge_embeddings(G, label_embeds, label_dict, edge_dim):
    """返回格式与原始版本兼容的边嵌入"""
    try:
        # 预计算所有唯一边类型的嵌入
        edge_types = set()
        edge_embeds = []
        edge_index_map = defaultdict(list)
        
        # 首先收集所有边类型
        for src, dst in G.edges():
            src_label = G.nodes[src]['labels'][0]
            dst_label = G.nodes[dst]['labels'][0]
            edge_type = (src_label, dst_label)
            edge_types.add(edge_type)
        
        # 为每种边类型计算嵌入
        for idx, (src_label, dst_label) in enumerate(edge_types):
            edge_type = (src_label, dst_label)
            src_vec = label_embeds[label_dict[src_label]]
            dst_vec = label_embeds[label_dict[dst_label]]
            edge_vec = (src_vec + dst_vec) / 2
            
            # 维度调整
            if edge_vec.shape[0] > edge_dim:
                projection = np.random.rand(edge_vec.shape[0], edge_dim).astype(np.float32)
                edge_vec = np.dot(edge_vec, projection)
            elif edge_vec.shape[0] < edge_dim:
                edge_vec = np.pad(edge_vec, (0, edge_dim - edge_vec.shape[0]), mode='constant')
            
            edge_embeds.append(edge_vec)
            edge_index_map[edge_type].append(idx)
        
        edge_embeds = np.array(edge_embeds)
        edge_embeds = edge_embeds / (np.linalg.norm(edge_embeds, axis=1, keepdims=True) + 1e-10)
        
        return edge_embeds, edge_index_map
    except Exception as e:
        logging.error(f"Error generating edge embeddings: {e}")
        exit(1)

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    args = parse_args()
    G, label_dict = load_graph(args.graph, args.output, args.mapping_output)
    
    # Generate label embeddings
    label_embeds = generate_node_embeddings(G, label_dict, args.dim)
    np.save(args.output, label_embeds)
    logging.info(f"Label embeddings saved to {args.output}, shape={label_embeds.shape}")
    
    # Generate edge embeddings
    edge_embeds, edge_index_map = generate_edge_embeddings(G, label_embeds, label_dict, args.edge_dim)
    edge_output = args.output.replace(".emb.npy", "_edge.emb.npy")
    np.save(edge_output, edge_embeds)
    logging.info(f"Edge embeddings saved to {edge_output}, shape={edge_embeds.shape}")
    edge_index = args.output.replace(".emb.npy", "_edge_index_map.json")
    with open(edge_index, "w") as f:
        json.dump({str(k): v for k, v in edge_index_map.items()}, f, indent=2)
    logging.info(f"Edge index saved to {edge_index}")

if __name__ == "__main__":
    main()