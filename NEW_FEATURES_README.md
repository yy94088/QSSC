# 新功能使用说明

本次更新添加了三个主要功能：

## 1. 知识蒸馏（Knowledge Distillation）

### 功能描述
实现了基于置信度的知识蒸馏功能，可以利用软标签（教师模型预测）来辅助训练。系统会自动评估软标签的质量，并根据其与真实标签的差异动态调整蒸馏权重。

### 设计特点
- **自适应权重调整**：软标签质量好时增加蒸馏权重，质量差时依赖硬标签
- **温度缩放**：使用temperature参数平滑预测分布，促进知识迁移
- **质量监控**：训练时会打印软标签质量权重，便于观察蒸馏效果

### 使用方法
```bash
python active_train.py \
    --use_distillation \
    --distillation_alpha 0.5 \
    --distillation_temperature 2.0 \
    --confidence_threshold 1.0 \
    --dataset dblp
```

### 参数说明
- `--use_distillation`: 启用知识蒸馏功能
- `--distillation_alpha`: 硬标签和软标签的平衡参数（0~1）
  - 0 = 只使用硬标签（真实标签）
  - 1 = 只使用软标签（教师预测）
  - 0.5 = 平衡使用（推荐）
- `--distillation_temperature`: 温度参数，控制预测分布的平滑程度
  - 较高的温度（如2.0~4.0）使分布更平滑，有利于知识迁移
  - 较低的温度接近原始预测
- `--confidence_threshold`: 软标签质量阈值（log空间）
  - 软标签误差超过此阈值时，会降低其权重
  - 建议设置为1.0~2.0

### 实现文件
- `cardnet/loss.py`: `DistillationLoss` 类
- `active/ActiveLearner.py`: 集成蒸馏损失到训练循环

## 2. 子图关系网络（Subgraph Relation Network）

### 功能描述
设计了专门的网络模块来学习分解后子图之间的关系和交互。系统提供两种实现方式：
1. **注意力机制**（推荐）：使用Multi-Head Self-Attention捕获子图间依赖
2. **图神经网络**：将子图视为元图节点，使用GNN建模关系

### 设计特点
- **自动关系学习**：无需手动定义子图间连接，自动学习依赖关系
- **多头注意力**：可以捕获不同类型的子图交互模式
- **残差连接**：保留原始子图信息的同时增强关系建模
- **灵活切换**：支持两种关系建模方式，可根据数据特点选择

### 使用方法

#### 方式1：注意力机制（推荐）
```bash
python active_train.py \
    --use_subgraph_relation \
    --relation_type attention \
    --relation_hidden_dim 128 \
    --dataset dblp
```

#### 方式2：图神经网络
```bash
python active_train.py \
    --use_subgraph_relation \
    --relation_type graph \
    --relation_hidden_dim 128 \
    --dataset dblp
```

### 参数说明
- `--use_subgraph_relation`: 启用子图关系网络
- `--relation_type`: 关系网络类型
  - `attention`: 基于Multi-Head Self-Attention（推荐，计算效率高）
  - `graph`: 基于图神经网络（适合有明确子图连接关系的场景）
- `--relation_hidden_dim`: 关系网络的隐藏层维度（建议128或256）

### 实现文件
- `cardnet/relation.py`: `SubgraphRelationNet` 和 `GraphBasedRelationNet` 类
- `cardnet/model.py`: 在CardNet中集成关系网络
- `cardnet/encoder.py`: 修改编码器支持返回单个子图embedding

### 工作原理
1. 编码器对每个子图生成embedding
2. 关系网络对所有子图embeddings进行交互建模
3. 增强后的embeddings重新池化得到最终图表示
4. 预测器基于增强表示进行基数估计

## 3. Q-Error符号保留

### 功能描述
修改了Q-Error的计算和打印逻辑，不再使用绝对值，而是保留原始符号信息，用于区分过估计和欠估计。

### 设计特点
- **符号含义**：
  - 正值（+）：模型过估计（预测值 > 真实值）
  - 负值（-）：模型欠估计（预测值 < 真实值）
- **详细统计**：打印时会显示带符号的误差分布
- **保持兼容**：L1 loss计算仍使用绝对值，只有输出显示改变

### 输出示例
```
Student Model Q-Error (pred-true, +overest/-underest) of 1000 Queries:
Min/Max: -2.5431 / 3.2156
Mean: 0.1234
Median: 0.0567
25%/75% Quantiles: -0.4523 / 0.6234
```

### 解读方法
- Mean > 0：模型整体倾向于过估计
- Mean < 0：模型整体倾向于欠估计
- Mean ≈ 0：模型预测相对平衡

### 实现文件
- `active/active_util.py`: 修改 `print_eval_res` 函数
- `active/ActiveLearner.py`: 添加注释说明L1仍使用绝对值

## 综合使用示例

同时启用所有三个功能：

```bash
python active_train.py \
    --dataset dblp \
    --use_distillation \
    --distillation_alpha 0.5 \
    --distillation_temperature 2.0 \
    --confidence_threshold 1.0 \
    --use_subgraph_relation \
    --relation_type attention \
    --relation_hidden_dim 128 \
    --epochs 30 \
    --batch_size 16 \
    --learning_rate 1e-4
```

## 注意事项

1. **知识蒸馏**：
   - 需要数据中包含soft_card字段（软标签）
   - 如果soft_card为NaN或缺失，会自动退化为标准训练
   - 建议先用小的alpha（如0.3）测试效果

2. **子图关系网络**：
   - 会增加模型参数量和计算成本
   - 对于子图数量较少（<3）的情况，效果可能不明显
   - attention类型内存占用较小，推荐首选

3. **Q-Error符号**：
   - 不影响模型训练，只改变输出显示
   - 可以帮助分析模型是否存在系统性偏差
   - 结合loss曲线一起观察效果更佳

## 代码结构

```
QSSC/
├── cardnet/
│   ├── loss.py           # 添加了DistillationLoss类
│   ├── relation.py       # 新增：子图关系网络模块
│   ├── model.py          # 修改：集成子图关系网络
│   └── encoder.py        # 修改：支持返回子图embeddings
├── active/
│   ├── ActiveLearner.py  # 修改：集成蒸馏损失
│   └── active_util.py    # 修改：Q-Error符号保留
└── active_train.py       # 修改：添加新参数
```

## 性能建议

1. **首次使用**：建议从单独测试每个功能开始
   ```bash
   # 只测试蒸馏
   python active_train.py --use_distillation
   
   # 只测试关系网络
   python active_train.py --use_subgraph_relation
   ```

2. **参数调优顺序**：
   - 先找到好的基础模型参数（learning_rate, batch_size等）
   - 再调整蒸馏参数（alpha, temperature）
   - 最后调整关系网络参数（hidden_dim）

3. **计算资源**：
   
   - 关系网络会增加约20-30%的训练时间
   - 如果GPU内存不足，可以减小relation_hidden_dim或batch_size

## 技术细节

### 蒸馏损失公式
```
quality_weight = sigmoid((threshold - |soft_label - hard_label|) / T)
adaptive_alpha = alpha * mean(quality_weight)
loss = (1 - adaptive_alpha) * MSE(pred, hard) + adaptive_alpha * T² * MSE(pred/T, soft/T)
```

### 关系网络架构
```
Input: [num_subgraphs, embedding_dim]
  ↓
Input Projection: Linear(embedding_dim, hidden_dim)
  ↓
Multi-Head Attention (4 heads)
  ↓
LayerNorm + Residual
  ↓
Feed-Forward Network
  ↓
LayerNorm + Residual
  ↓
Output Projection: Linear(hidden_dim, embedding_dim)
  ↓
Output: [num_subgraphs, embedding_dim] (enhanced)
```
