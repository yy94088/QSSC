import torch
import torch.nn as nn
import argparse
import json
import numpy as np
import os
from sklearn.model_selection import KFold
from simple_queryset import SimpleQueryset
from simple_gin import SimpleGIN
from active import _to_cuda, print_eval_res, save_eval_res
from QuerySampler import QueryDecompose
from torch.utils.data import DataLoader
from simple_queryset import QueryDataset
from util import model_checkpoint
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed_type", default="prone", type=str,
						help="the node feature encoding type") # freq, n2v, prone, n2v_concat, prone_concat, nrp are tested
    parser.add_argument("--edge_embed_type", default="prone", type=str,
						help="the edge feature encoding type")
    parser.add_argument("--full_data_dir", type=str, default="./data/")
    parser.add_argument('--dataset', type=str, default='hprd')
    parser.add_argument('--data_dir', type=str, default='data/dataset')
    parser.add_argument('--queryset_dir', type=str, default='data/queryset')
    parser.add_argument('--true_card_dir', type=str, default='data/true_cardinality')
    parser.add_argument('--model_save_dir', type=str, default='models')
    parser.add_argument('--result_save_dir', type=str, default='result')
    parser.add_argument('--model_type', type=str, default='simple_gin')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--num_hidden', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument("--size", type=int, default=8)
    parser.add_argument("--pattern", type=str, default='query', help="Specific pattern_size to load (e.g., query_4 query_8)")
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--k', type=int, default=3,	help='decompose hop number.')
    parser.add_argument("--cumulative", default=False, type=bool,help='Whether or not to enable cumulative learning')
    parser.add_argument('--device', type=str, default='cuda:0')
    # 添加交叉验证相关参数
    parser.add_argument('--cross_validation', action='store_true', default=True, help='Enable cross validation')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of folds for cross validation')
    parser.add_argument('--fold_seed', type=int, default=42, help='Random seed for fold splitting')
    # 添加并行处理相关参数
    parser.add_argument('--use_parallel', action='store_true', default=True, help='Enable parallel processing for query decomposition')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of worker processes for parallel processing')
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

def train(model, criterion, train_loaders, val_loaders, optimizer, args):
    device = torch.device(args.device if args.cuda and torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    # 学习率调度与最优权重缓存
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    best_state_dict = None
    
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for loader_idx, dataloader in enumerate(train_loaders):
            for i, (x, edge_index, edge_attr, card) in enumerate(dataloader):
                if args.cuda:
                    x, edge_index, edge_attr, card = _to_cuda([x, edge_index, edge_attr, card])
                # 仅移除 DataLoader 引入的 batch 维度，避免误删其他尺寸为1的维度
                x = x.squeeze(0)
                edge_index = edge_index.squeeze(0)
                edge_attr = edge_attr.squeeze(0)
                card = card.squeeze(0)
                output = model(x, edge_index, edge_attr=edge_attr, batch=torch.zeros(x.size(0), dtype=torch.long, device=x.device))
                loss = criterion(output, card)
                epoch_loss += loss.item()
                loss.backward()
                # 梯度裁剪，提升训练稳定性
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                # 对于batch_size=1，每个样本都更新一次
                optimizer.step()
                optimizer.zero_grad()
        
        # 验证损失
        val_loss = 0.0
        val_count = 0
        model.eval()
        with torch.no_grad():
            for loader_idx, dataloader in enumerate(val_loaders):
                for x, edge_index, edge_attr, card in dataloader:
                    if args.cuda:
                        x, edge_index, edge_attr, card = _to_cuda([x, edge_index, edge_attr, card])
                    x = x.squeeze(0)
                    edge_index = edge_index.squeeze(0)
                    edge_attr = edge_attr.squeeze(0)
                    card = card.squeeze(0)
                    output = model(x, edge_index, edge_attr=edge_attr, batch=torch.zeros(x.size(0), dtype=torch.long, device=x.device))
                    val_loss += criterion(output, card).item()
                    val_count += 1
        model.train()
        
        # 打印平均损失，便于观察趋势
        num_train_samples = sum(len(dl.dataset) for dl in train_loaders)
        avg_train_loss = epoch_loss / max(num_train_samples, 1)
        avg_val_loss = val_loss / max(val_count, 1)
        print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        # 早停机制
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # 记录最佳权重
            best_state_dict = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        # 学习率调度（基于验证损失）
        scheduler.step(avg_val_loss)
    
    # 回滚到验证集最优权重
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    return model

def evaluate(model, criterion, eval_loaders, args, print_res=True):
    device = torch.device(args.device if args.cuda and torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    all_eval_res = []
    
    for loader_idx, dataloader in enumerate(eval_loaders):
        res, loss, l1 = [], 0.0, 0.0
        
        for x, edge_index, edge_attr, card in dataloader:
            if args.cuda:
                x, edge_index, edge_attr, card = _to_cuda([x, edge_index, edge_attr, card])
            x = x.squeeze(0)
            edge_index = edge_index.squeeze(0)
            edge_attr = edge_attr.squeeze(0)
            card = card.squeeze(0)

            output = model(x, edge_index, edge_attr=edge_attr, batch=torch.zeros(x.size(0), dtype=torch.long, device=x.device))
            
            true_card = card.item()
            pred_card = output.item()
            
            mse_loss = criterion(output, card).item()
            l1_error = torch.abs(output - card).item()
            
            loss += mse_loss
            l1 += l1_error
            res.append((true_card, pred_card))
        
        all_eval_res.append((res, loss, l1, 0.0))
    
    if print_res:
        print_eval_res(all_eval_res)
    
    return all_eval_res

def cross_validation_train(args, all_queries, num_node_feat, num_edge_feat):
    """
    执行交叉验证训练
    """
    print(f"Starting {args.n_folds}-fold cross validation...")
    
    # 准备所有数据
    all_data = []
    for (pattern, size), graphs_card_pairs in all_queries.items():
        all_data.extend(graphs_card_pairs)
    
    # 创建KFold分割
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.fold_seed)
    fold_results = []
    fold_models = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(all_data)):
        print(f"\n=== Fold {fold + 1}/{args.n_folds} ===")
        
        # 分割数据
        train_data = [all_data[i] for i in train_idx]
        val_data = [all_data[i] for i in val_idx]
        
        # 转换数据为张量格式
        train_tensors = []
        val_tensors = []
        
        # 创建临时的SimpleQueryset来获取数据转换功能
        temp_queryset = SimpleQueryset(args, {('temp', 0): train_data + val_data})
        
        # 转换训练数据
        for query, card, _ in train_data:
            x, edge_index, edge_attr = temp_queryset._get_graph_data(query)
            train_tensors.append((x, edge_index, edge_attr, card))
        
        # 转换验证数据
        for query, card, _ in val_data:
            x, edge_index, edge_attr = temp_queryset._get_graph_data(query)
            val_tensors.append((x, edge_index, edge_attr, card))
        
        train_dataset = QueryDataset(queries=train_tensors)
        val_dataset = QueryDataset(queries=val_tensors)
        
        train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False)
        
        # 初始化模型
        model = SimpleGIN(
            num_node_feat=num_node_feat,
            num_edge_feat=num_edge_feat,
            num_hidden=args.num_hidden,
            num_layers=args.num_layers,
            dropout=args.dropout
        )
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        
        # 训练模型
        model = train(model, criterion, [train_loader], [val_loader], optimizer, args)
        
        # 评估模型 - 在最后一个fold显示详细结果
        if fold == args.n_folds - 1:
            # 最后一个fold显示详细预测结果
            val_results = evaluate(model, criterion, [val_loader], args, print_res=True)
        else:
            val_results = evaluate(model, criterion, [val_loader], args, print_res=False)
        
        # 计算验证指标
        total_loss = sum(result[1] for result in val_results)
        total_l1 = sum(result[2] for result in val_results)
        avg_loss = total_loss / len(val_results) if val_results else 0
        avg_l1 = total_l1 / len(val_results) if val_results else 0
        
        print(f"Fold {fold + 1} - Avg Loss: {avg_loss:.4f}, Avg L1: {avg_l1:.4f}")
        fold_results.append((avg_loss, avg_l1))
        fold_models.append(model)
    
    # 计算平均结果
    avg_loss = np.mean([result[0] for result in fold_results])
    avg_l1 = np.mean([result[1] for result in fold_results])
    std_loss = np.std([result[0] for result in fold_results])
    std_l1 = np.std([result[1] for result in fold_results])
    
    print(f"\n=== Cross Validation Results ===")
    print(f"Average Loss: {avg_loss:.4f} ± {std_loss:.4f}")
    print(f"Average L1: {avg_l1:.4f} ± {std_l1:.4f}")
    
    # 保存交叉验证结果
    cv_results = {
        'fold_results': fold_results,
        'avg_loss': avg_loss,
        'avg_l1': avg_l1,
        'std_loss': std_loss,
        'std_l1': std_l1
    }
    
    result_file = os.path.join(args.result_save_dir, f"cv_results_{args.dataset}.json")
    os.makedirs(args.result_save_dir, exist_ok=True)
    with open(result_file, 'w') as f:
        json.dump(cv_results, f, indent=2)
    
    return fold_models, cv_results

def main():
    args = parse_args()
    # 加载查询图数据
    QD = QueryDecompose(
        queryset_dir = args.queryset_dir,
        true_card_dir = args.true_card_dir,
        dataset = args.dataset,
        pattern = args.pattern,
        size = args.size,
        teacher_model_path = args.teacher_model_path
    )
    # 根据参数选择是否使用并行处理
    if args.use_parallel:
        print(f"Using parallel processing for query decomposition with {args.num_workers} workers...")
        QD.decompose_queries_parallel(num_workers=args.num_workers)
    else:
        print("Using sequential processing for query decomposition...")
        QD.decompose_queries()
    
    if args.cross_validation:
        # 为了获取正确的特征维度，先创建一个临时的queryset
        temp_queryset = SimpleQueryset(args, QD.all_queries)
        
        # 执行交叉验证
        fold_models, cv_results = cross_validation_train(args, QD.all_queries, 
                                                       num_node_feat=temp_queryset.num_node_feat, 
                                                       num_edge_feat=temp_queryset.num_edge_feat)
        
        # 保存最佳模型（这里选择验证损失最低的模型）
        best_fold = np.argmin([result[0] for result in cv_results['fold_results']])
        best_model = fold_models[best_fold]
        
        # 为保存模型创建一个临时的optimizer
        temp_optimizer = torch.optim.Adam(best_model.parameters(), lr=args.lr)     
        model_checkpoint(args, best_model, temp_optimizer, model_dir=f"simple_gin_{args.dataset}_cv_best.pth")
        print(f"Best model from fold {best_fold + 1} saved.")
        
    else:
        # 原有的训练流程
        queryset = SimpleQueryset(args, QD.all_queries)

        # 初始化模型
        model = SimpleGIN(
            num_node_feat=queryset.num_node_feat,
            num_edge_feat=queryset.num_edge_feat,
            num_hidden=args.num_hidden,
            num_layers=args.num_layers,
            dropout=args.dropout
        )
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # 训练
        model = train(model, criterion, queryset.train_loaders, queryset.val_loaders, optimizer, args)

        # 评估
        all_eval_res = evaluate(model, criterion, queryset.test_loaders, args, print_res=True)
        save_eval_res(args, sorted(queryset.all_queries.keys()), all_eval_res, args.result_save_dir)

        # 保存模型
        model_checkpoint(args, model, optimizer, model_dir=f"simple_gin_{args.dataset}.pth")

if __name__ == "__main__":
    main()