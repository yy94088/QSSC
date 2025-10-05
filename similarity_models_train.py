import torch
import torch.nn as nn
import argparse
import json
import numpy as np
import os

from simple_queryset import SimpleQueryset
from active import _to_cuda, print_eval_res, save_eval_res
from QuerySampler import QueryDecompose
from torch.utils.data import DataLoader
from simple_queryset import QueryDataset
from util import model_checkpoint
import torch.multiprocessing as mp

# 导入我们的相似性模型
from query_similarity_net import QSimNet, ContrastiveLoss

mp.set_start_method('spawn', force=True)


# 简单的KFold实现，避免sklearn依赖
class KFold:
    def __init__(self, n_splits=5, shuffle=True, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
    
    def split(self, data):
        n_samples = len(data)
        indices = list(range(n_samples))
        
        if self.shuffle:
            if self.random_state is not None:
                import random
                random.seed(self.random_state)
            import random
            random.shuffle(indices)
        
        fold_size = n_samples // self.n_splits
        for i in range(self.n_splits):
            start = i * fold_size
            if i == self.n_splits - 1:
                # 最后一个fold包含剩余的所有样本
                test_indices = indices[start:]
            else:
                test_indices = indices[start:start + fold_size]
            
            train_indices = [idx for idx in indices if idx not in test_indices]
            yield train_indices, test_indices

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed_type", default="prone", type=str,
						help="the node feature encoding type")
    parser.add_argument("--edge_embed_type", default="prone", type=str,
						help="the edge feature encoding type")
    parser.add_argument("--full_data_dir", type=str, default="./data/")
    parser.add_argument('--dataset', type=str, default='hprd')
    parser.add_argument('--data_dir', type=str, default='data/dataset')
    parser.add_argument('--queryset_dir', type=str, default='data/queryset')
    parser.add_argument('--true_card_dir', type=str, default='data/true_cardinality')
    parser.add_argument('--model_save_dir', type=str, default='models')
    parser.add_argument('--result_save_dir', type=str, default='result')
    
    # 模型选择
    
    parser.add_argument('--num_hidden', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)  # 增大batch size for similarity learning
    parser.add_argument("--size", type=int, default=8)
    parser.add_argument("--pattern", type=str, default='query')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument("--cumulative", default=False, type=bool, help='For compatibility with model_checkpoint')
    
    # 相似性学习特定参数
    parser.add_argument('--memory_size', type=int, default=200, help='Memory bank size for QSimNet')
    parser.add_argument('--contrastive_weight', type=float, default=0.2, help='Weight for contrastive loss')
    parser.add_argument('--use_temporal', action='store_true', default=False, help='Use temporal similarity')
    
    # 训练参数
    parser.add_argument('--cross_validation', action='store_true', default=True)
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--fold_seed', type=int, default=42)
    parser.add_argument('--use_parallel', action='store_true', default=True)
    parser.add_argument('--num_workers', type=int, default=8)
    
    args = parser.parse_args()
    
    args.teacher_model_path = os.path.join("FlowSC_model", args.dataset, "FlowSC_model_"+str(args.size)+".pt")
    args.prone_feat_dir = os.path.join(args.full_data_dir, "prone")
    args.n2v_feat_dir = os.path.join(args.full_data_dir, "n2v")
    args.nrp_feat_dir = os.path.join(args.full_data_dir, "nrp")
    args.embed_feat_dir = args.n2v_feat_dir if args.embed_type == "n2v" or args.embed_type == "n2v_concat" else \
		args.prone_feat_dir
    if args.embed_type == "nrp":
         args.embed_feat_dir = args.nrp_feat_dir

    return args

def create_similarity_model(args, num_node_feat, num_edge_feat):
    """根据参数创建相应的相似性模型"""
    model = QSimNet(
        num_node_feat=num_node_feat,
        num_edge_feat=num_edge_feat,
        hidden_dim=args.num_hidden,
        memory_size=args.memory_size
    )
    criterion = nn.MSELoss()
    contrastive_criterion = ContrastiveLoss()
    
    return model, criterion, contrastive_criterion

def prepare_batch_data(dataloader, args):
    """准备batch数据用于相似性模型"""
    batch_data = []
    for x, edge_index, edge_attr, card in dataloader:
        if args.cuda:
            x, edge_index, edge_attr, card = _to_cuda([x, edge_index, edge_attr, card])
        
        # 移除DataLoader添加的batch维度
        x = x.squeeze(0)
        edge_index = edge_index.squeeze(0) 
        edge_attr = edge_attr.squeeze(0)
        card = card.squeeze(0)
        
        batch_data.append((x, edge_index, edge_attr, card))
    
    return batch_data

def train_similarity_model(model, criterion, contrastive_criterion, train_loaders, val_loaders, optimizer, args):
    """训练相似性模型"""
    device = torch.device(args.device if args.cuda and torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()
    
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    
    # 学习率调度
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    best_state_dict = None
    
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        # 训练循环
        for loader_idx, dataloader in enumerate(train_loaders):
            # 收集一个batch的数据
            batch_data = prepare_batch_data(dataloader, args)
            
            if len(batch_data) == 0:
                continue
            
            # 前向传播
            model_output = model(batch_data, training=True)
            predictions = model_output['predictions']
            true_cardinalities = torch.stack([item[3] for item in batch_data])
            
            # 主损失
            main_loss = criterion(predictions, true_cardinalities)
            
            # 对比损失
            if contrastive_criterion is not None:
                contrastive_loss = contrastive_criterion(
                    model_output['contrastive_embeddings'],
                    true_cardinalities,
                    model_output['similarity_matrix']
                )
                total_loss = main_loss + args.contrastive_weight * contrastive_loss
            else:
                total_loss = main_loss
            
            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += total_loss.item()
            num_batches += 1
        
                    # 验证 (每个epoch不打印详细结果)
            val_loss = evaluate_similarity_model(model, criterion, val_loaders, args, training=False, print_res=False)
            if isinstance(val_loss, tuple):
                val_loss = val_loss[0]
        
        # 打印损失
        avg_train_loss = epoch_loss / max(num_batches, 1)
        print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        # 早停和学习率调度
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state_dict = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        scheduler.step(val_loss)
    
    # 加载最佳权重
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    
    return model

def evaluate_similarity_model(model, criterion, eval_loaders, args, training=False, print_res=False):
    """评估相似性模型"""
    device = torch.device(args.device if args.cuda and torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    total_loss = 0.0
    total_l1 = 0.0
    num_batches = 0
    all_results = []
    all_eval_res = []  # 用于print_eval_res的数据结构
    
    with torch.no_grad():
        for loader_idx, dataloader in enumerate(eval_loaders):
            batch_data = prepare_batch_data(dataloader, args)
            if len(batch_data) == 0:
                continue
            # 模型预测
            model_output = model(batch_data, training=training)
            predictions = model_output['predictions']
            true_cardinalities = torch.stack([item[3] for item in batch_data])
            # 计算损失
            loss = criterion(predictions, true_cardinalities)
            l1_error = torch.mean(torch.abs(predictions - true_cardinalities))
            total_loss += loss.item()
            total_l1 += l1_error.item()
            num_batches += 1
            # 收集结果用于详细分析
            res = []  # 用于all_eval_res的数据
            for pred, true_card in zip(predictions, true_cardinalities):
                all_results.append((true_card.item(), pred.item()))
                res.append((true_card.item(), pred.item()))
            all_eval_res.append((res, loss.item(), l1_error.item(), 0.0))  # elapse_time设为0
    
    avg_loss = total_loss / max(num_batches, 1)
    avg_l1 = total_l1 / max(num_batches, 1)
    
    if print_res and all_results:
        true_values = [r[0] for r in all_results]
        pred_values = [r[1] for r in all_results]
        
        q_errors = []
        detailed_results = []
        
        for i, (true_val, pred_val) in enumerate(zip(true_values, pred_values)):
            mse_loss = (pred_val - true_val) ** 2
            l1_error = abs(pred_val - true_val)
            
            true_card_raw = 2 ** true_val
            pred_card_raw = 2 ** pred_val
            if true_card_raw > 0 and pred_card_raw > 0:
                q_error = max(pred_card_raw / true_card_raw, true_card_raw / pred_card_raw)
            else:
                q_error = float('inf') if pred_card_raw > 0 else 1.0
            
            # 计算相对误差
            relative_error = abs(pred_val - true_val) / max(abs(true_val), 1e-8)
            
            if q_error != float('inf'):
                q_errors.append(q_error)
            
            # 存储详细结果
            detailed_result = {
                'sample_id': i,
                'true_cardinality': true_val,
                'predicted_cardinality': pred_val,
                'mse_loss': mse_loss,
                'l1_error': l1_error,
                'q_error': q_error,
                'relative_error': relative_error
            }
            detailed_results.append(detailed_result)
        
        if q_errors:
            avg_q_error = np.mean(q_errors)
            median_q_error = np.median(q_errors)
            
            # 打印详细结果（按照simple_train.py的格式）
            print("\n" + "="*80)
            print("详细测试集预测结果")
            print("="*80)
            
            # 打印表头
            print(f"{'样本ID':<8} {'真实值(log)':<12} {'预测值(log)':<12} {'MSE Loss':<12} {'L1 Error':<12} {'Q-Error':<12} {'相对误差':<12}")
            print("-"*80)
            
            # 打印每个样本的详细结果
            total_mse = 0.0
            total_l1_detailed = 0.0
            total_relative_error = 0.0
            
            for result in detailed_results:
                print(f"{result['sample_id']:<8} {result['true_cardinality']:<12.2f} {result['predicted_cardinality']:<12.2f} "
                      f"{result['mse_loss']:<12.4f} {result['l1_error']:<12.2f} {result['q_error']:<12.2f} "
                      f"{result['relative_error']:<12.2f}")
                
                total_mse += result['mse_loss']
                total_l1_detailed += result['l1_error']
                total_relative_error += result['relative_error']
            
            # 打印统计信息
            print("-"*80)
            num_samples = len(detailed_results)
            avg_mse = total_mse / num_samples
            avg_l1_detailed = total_l1_detailed / num_samples
            avg_relative_error = total_relative_error / num_samples
             
            print(f"平均MSE Loss: {avg_mse:.4f}")
            print(f"平均L1 Error: {avg_l1_detailed:.2f}")
            print(f"平均Q-Error: {avg_q_error:.2f}")
            print(f"中位数Q-Error: {median_q_error:.2f}")
            print(f"平均相对误差: {avg_relative_error:.2f}")
            print("="*80)
            
            # 保存详细结果到文件
            os.makedirs(args.result_save_dir, exist_ok=True)
            detailed_results_file = os.path.join(args.result_save_dir, f"detailed_predictions_{args.dataset}_qsimnet.json")
            
            # 添加统计信息到保存的数据中
            save_data = {
                'detailed_results': detailed_results,
                'statistics': {
                    'num_samples': num_samples,
                    'avg_mse': avg_mse,
                    'avg_l1': avg_l1_detailed,
                    'avg_relative_error': avg_relative_error,
                    'avg_q_error': avg_q_error,
                    'median_q_error': median_q_error
                }
            }
            
            with open(detailed_results_file, 'w') as f:
                json.dump(save_data, f, indent=2, default=str)
            
            print(f"详细预测结果已保存到: {detailed_results_file}")
            
            # 调用原有的print_eval_res函数
            print_eval_res(all_eval_res)
        else:
            print(f"Average Q-Error: inf")
            print(f"Median Q-Error: inf")
    
    return avg_loss, all_eval_res

def cross_validation_similarity_train(args, all_queries, num_node_feat, num_edge_feat):
    """交叉验证训练相似性模型"""
    print(f"Starting {args.n_folds}-fold cross validation for qsimnet...")
    
    # 准备数据
    all_data = []
    for (pattern, size), graphs_card_pairs in all_queries.items():
        all_data.extend(graphs_card_pairs)
    
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.fold_seed)
    fold_results = []
    fold_models = []
    # 获取全局test_loaders，保证每个fold都能访问
    test_queryset = SimpleQueryset(args, all_queries)
    test_loaders = getattr(test_queryset, 'test_loaders', None)

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_data)):
        print(f"\n=== Fold {fold + 1}/{args.n_folds} ===")
        
        # 分割数据
        train_data = [all_data[i] for i in train_idx]
        val_data = [all_data[i] for i in val_idx]
        
        # 创建数据加载器
        train_dataset, val_dataset = create_datasets(train_data, val_data, args, all_data)
        train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False)
        
        # 创建模型
        model, criterion, contrastive_criterion = create_similarity_model(args, num_node_feat, num_edge_feat)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        
        # 训练
        model = train_similarity_model(
            model, criterion, contrastive_criterion, [train_loader], [val_loader], optimizer, args
        )
        
        # 每个fold结束时，用最佳模型打印一次详细结果
        print(f"\n=== Fold {fold + 1}/{args.n_folds} 验证集详细评估结果 ===")
        val_loss, val_eval_res = evaluate_similarity_model(model, criterion, [val_loader], args, print_res=True)
        print(f"Fold {fold + 1} - Val Loss: {val_loss:.4f}")
        # 用active_util的print_eval_res打印详细预测结果
        try:
            from active.active_util import print_eval_res
            print_eval_res(val_eval_res)
        except Exception as e:
            print(f"[Warning] print_eval_res import/exec failed: {e}")
        # 每个fold结束时，对测试集打印一次详细结果
        if test_loaders is not None:
            print(f"\n=== Fold {fold + 1}/{args.n_folds} 测试集详细评估结果 ===")
            test_loss, test_eval_res = evaluate_similarity_model(model, criterion, test_loaders, args, print_res=True)
            try:
                from active.active_util import print_eval_res
                print_eval_res(test_eval_res)
            except Exception as e:
                print(f"[Warning] print_eval_res import/exec failed: {e}")
        fold_results.append(val_loss)
        fold_models.append(model)
    if len(fold_results) > 0:
        avg_loss = np.mean(fold_results)
        std_loss = np.std(fold_results)
    else:
        avg_loss = 0.0
        std_loss = 0.0

    print(f"\n=== Cross Validation Results for QSIMNET ===")
    print(f"Average Loss: {avg_loss:.4f} ± {std_loss:.4f}")
    return fold_models, {'avg_loss': avg_loss, 'std_loss': std_loss, 'fold_results': fold_results}

def create_datasets(train_data, val_data, args, all_data):
    """创建数据集"""
    # 创建临时的SimpleQueryset来获取数据转换功能
    temp_queryset = SimpleQueryset(args, {('temp', 0): all_data})
    
    # 转换训练数据
    train_tensors = []
    for query, card, _ in train_data:
        x, edge_index, edge_attr = temp_queryset._get_graph_data(query)
        train_tensors.append((x, edge_index, edge_attr, card))
    
    # 转换验证数据
    val_tensors = []
    for query, card, _ in val_data:
        x, edge_index, edge_attr = temp_queryset._get_graph_data(query)
        val_tensors.append((x, edge_index, edge_attr, card))
    
    train_dataset = QueryDataset(queries=train_tensors)
    val_dataset = QueryDataset(queries=val_tensors)
    
    return train_dataset, val_dataset

def main():
    args = parse_args()
    
    print(f"Training QSIMNET model...")
    print(f"Model parameters: hidden_dim={args.num_hidden}, batch_size={args.batch_size}")
    
    # 加载数据
    QD = QueryDecompose(
        queryset_dir=args.queryset_dir,
        true_card_dir=args.true_card_dir,
        dataset=args.dataset,
        pattern=args.pattern,
        size=args.size,
        teacher_model_path=args.teacher_model_path
    )
    
    if args.use_parallel:
        print(f"Using parallel processing with {args.num_workers} workers...")
        QD.decompose_queries_parallel(num_workers=args.num_workers)
    else:
        print("Using sequential processing...")
        QD.decompose_queries()
    
    # 获取特征维度
    temp_queryset = SimpleQueryset(args, QD.all_queries)
    num_node_feat = temp_queryset.num_node_feat
    num_edge_feat = temp_queryset.num_edge_feat
    
    print(f"Feature dimensions: node_feat={num_node_feat}, edge_feat={num_edge_feat}")
    
    if args.cross_validation:
        # 交叉验证
        fold_models, cv_results = cross_validation_similarity_train(
            args, QD.all_queries, num_node_feat, num_edge_feat
        )
        
        # 保存结果
        result_file = os.path.join(args.result_save_dir, f"qsimnet_cv_results_{args.dataset}.json")
        os.makedirs(args.result_save_dir, exist_ok=True)
        with open(result_file, 'w') as f:
            json.dump(cv_results, f, indent=2)
        
        # 保存最佳模型
        best_fold = np.argmin(cv_results['fold_results'])
        best_model = fold_models[best_fold]
        temp_optimizer = torch.optim.Adam(best_model.parameters(), lr=args.lr)
        model_checkpoint(args, best_model, temp_optimizer, 
                        model_dir=f"qsimnet_{args.dataset}_best.pth")
        print(f"Best model from fold {best_fold + 1} saved.")
        
    else:
        # 单次训练
        queryset = SimpleQueryset(args, QD.all_queries)
        model, criterion, contrastive_criterion = create_similarity_model(args, num_node_feat, num_edge_feat)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        # 训练
        model = train_similarity_model(
            model, criterion, contrastive_criterion, 
            queryset.train_loaders, queryset.val_loaders, optimizer, args
        )
        # 评估
        test_loss, test_eval_res = evaluate_similarity_model(model, criterion, queryset.test_loaders, args, print_res=True)
        try:
            from active.active_util import print_eval_res
            print_eval_res(test_eval_res)
        except Exception as e:
            print(f"[Warning] print_eval_res import/exec failed: {e}")
        # 保存模型
        model_checkpoint(args, model, optimizer, model_dir=f"qsimnet_{args.dataset}.pth")

if __name__ == "__main__":
    main()
