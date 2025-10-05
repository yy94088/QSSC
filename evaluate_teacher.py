
import torch
import numpy as np
import pickle
import os
import logging
import time
import pandas as pd
from read_queries import AllDataset
from learner import Flow_Learner, MSLELoss
from compute import q_error
from utils import Find_Candidates, bfs_layers, generate_query_adj, generate_adjacency_matrix, query_flow, data_flow

def setup_logging(log_path):
    logging.basicConfig(filename=log_path, level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')

def generate_teacher_inputs(data_graph_path, query_graph_path, device):

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
def evaluate_teacher(dataset_name='wordnet', query_size=8, device='cuda:0'):
    # 设置路径
    root_path = './dataset'
    teacher_model_path = '/home/wangchichu/SC/FlowSC_model_8.pt'
    data_graph_path = f'/home/wangchichu/SC/dataset/{dataset_name}/data_graph/{dataset_name}.graph'
    query_graph_dir = f'{root_path}/{dataset_name}/query_graph/query_{query_size}/'
    result_dir = f'./results/{dataset_name}_q{query_size}_teacher_eval'
    os.makedirs(result_dir, exist_ok=True)
    
    # 日志设置
    log_path = f'{result_dir}/teacher_eval.log'
    setup_logging(log_path)
    
    # 验证模型文件
    if not os.path.exists(teacher_model_path):
        logging.error(f'Teacher model file not found: {teacher_model_path}')
        raise FileNotFoundError(f'Teacher model file not found: {teacher_model_path}')
    
    # 加载数据集
    dataset = AllDataset(dataset_name, query_size)
    num_graphs = len(dataset.data)
    indices = np.arange(num_graphs)
    np.random.seed(41)
    np.random.shuffle(indices)
    train_split = int(0.7 * num_graphs)
    val_split = int(0.9 * num_graphs)
    train_indices = indices[:train_split]
    val_indices = indices[train_split:val_split]
    test_indices = indices[val_split:]
    
    # 加载教师模型
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    teacher_model = Flow_Learner(out_dim=128).to(device)
    try:
        state_dict = torch.load(teacher_model_path, map_location=device)
        teacher_model.load_state_dict(state_dict)
        logging.info(f'Teacher model loaded successfully from {teacher_model_path}')
        logging.info(f'Sample parameter: {next(iter(teacher_model.parameters())).data[:5]}')
    except Exception as e:
        logging.error(f'Failed to load teacher model: {str(e)}')
        raise
    teacher_model.eval()
    
    # 初始化评估指标
    msle_loss_fn = MSLELoss()
    results = {'filename': [], 'true_count': [], 'pred_count': [], 'q_error': []}
    splits = [('train', train_indices), ('val', val_indices), ('test', test_indices)]
    
    # 测试单图预测
    try:
        test_query = os.path.join(query_graph_dir, 'query_dense_8_135.graph')
        query_generator, data_generator = generate_teacher_inputs(data_graph_path, test_query, device)
        with torch.no_grad():
            pred = teacher_model(query_generator, data_generator, device)
        logging.info(f'Test prediction for query_dense_8_135.graph: {pred.item()}')
    except Exception as e:
        logging.error(f'Test prediction failed: {str(e)}')
    
    for split_name, split_indices in splits:
        logging.info(f'Evaluating on {split_name} set...')
        total_msle_loss = 0
        total_q_error = 0
        num_samples = 0
        
        for idx in split_indices:
            graph = dataset.data[idx]
            filename = graph['filename']
            true_count = graph['count']
            query_graph_path = os.path.join(query_graph_dir, filename)
            
            # 生成输入
            try:
                query_generator, data_generator = generate_teacher_inputs(
                    data_graph_path, query_graph_path, device)
                
                # 预测
                with torch.no_grad():
                    pred = teacher_model(query_generator, data_generator, device)
                pred_count = pred.item()
                print("pred",pred_count)
                
                # 计算指标
                msle_loss = msle_loss_fn(
                    torch.tensor([pred_count], dtype=torch.float32),
                    torch.tensor([true_count], dtype=torch.float32)
                ).item()
                q_err = q_error(pred_count, true_count)
                
                # 记录结果
                results['filename'].append(filename)
                results['true_count'].append(true_count)
                results['pred_count'].append(pred_count)
                results['q_error'].append(q_err)
                
                total_msle_loss += msle_loss
                total_q_error += q_err
                num_samples += 1
                
                logging.info(f'Graph: {filename}, True: {true_count:.2f}, Pred: {pred_count:.2f}, '
                             f'MSLE: {msle_loss:.6f}, Q-Error: {q_err:.6f}')
                
            except Exception as e:
                logging.error(f'Error processing {filename}: {str(e)}')
                continue
        
        # 统计平均指标
        avg_msle_loss = total_msle_loss / num_samples if num_samples > 0 else 0
        avg_q_error = total_q_error / num_samples if num_samples > 0 else 0
        logging.info(f'{split_name.capitalize()} set: Avg MSLE Loss: {avg_msle_loss:.6f}, '
                     f'Avg Q-Error: {avg_q_error:.6f}, Samples: {num_samples}')
    
    # 保存结果
    results_df = pd.DataFrame(results)
    results_csv = f'{result_dir}/teacher_eval_results.csv'
    results_df.to_csv(results_csv, index=False)
    logging.info(f'Results saved to {results_csv}')
    
    # 保存统计信息
    with open(f'{result_dir}/teacher_eval_metadata.txt', 'w') as f:
        for split_name, split_indices in splits:
            split_results = results_df[results_df['filename'].isin(
                [dataset.data[idx]['filename'] for idx in split_indices])]
            f.write(f'{split_name.capitalize()} Set:\n')
            f.write(f'Avg MSLE Loss: {split_results["q_error"].mean():.6f}\n')
            f.write(f'Avg Q-Error: {split_results["q_error"].mean():.6f}\n')
            f.write(f'Max Q-Error: {split_results["q_error"].max():.6f}\n')
            f.write(f'Min Q-Error: {split_results["q_error"].min():.6f}\n')
            f.write(f'Samples: {len(split_indices)}\n\n')
    
if __name__ == '__main__':
    evaluate_teacher()
