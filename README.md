# Transformer 从零实现 (Encoder-only & Encoder-Decoder)

本项目为《大型模型基础与应用》期中作业，完整实现了 Transformer 的 Encoder-only 与 Encoder-Decoder 结构，并在 Tiny Shakespeare数据集上进行训练与消融实验。

**特点：**
- 参照 tensor2tensor 实现风格
- 支持 Encoder-only（语言建模）和 Encoder-Decoder（序列到序列）两种模式
- 完整的训练、验证、可视化流程
- 梯度裁剪、学习率调度、模型保存等完善功能

---

## 项目结构

```
transformer-assignment/
├── src/
│   ├── model.py          # Transformer 模型实现（含详细注释）
│   │   ├── ScaledDotProductAttention      # 缩放点积注意力
│   │   ├── MultiHeadAttention             # 多头注意力机制
│   │   ├── PositionwiseFeedForward        # 位置前馈网络
│   │   ├── PositionalEncoding             # 正弦-余弦位置编码
│   │   ├── EncoderLayer                   # 编码器层
│   │   ├── DecoderLayer                   # 解码器层
│   │   ├── Transformer                    # 完整 Transformer 模型
│   │   ├── make_transformer_lm()          # 创建语言模型
│   │   └── make_transformer_seq2seq()     # 创建序列到序列模型
│   │
│   └── train.py          # 训练脚本（含详细注释）
│       ├── TinyTextDataset                # 字符级数据集
│       ├── train_epoch()                  # 训练一个 epoch
│       ├── validate()                     # 验证函数
│       ├── generate_text()                # 文本生成
│       ├── plot_training_curves()         # 可视化训练曲线
│       └── main()                         # 主训练流程
│
├── results/
│   ├── loss_curve.png    # 训练曲线图
│   ├── best_model.pt     # 最佳模型（验证集最低 loss）
│   └── final_model.pt    # 最终模型（包含词汇表等信息）
│
├── requirements.txt      # Python 依赖
├── README.md            # 本文件
└── main.tex             # LaTeX 报告源文件
```

---

## 环境要求

- Python 3.10+
- PyTorch >= 2.0
- CUDA（可选，用于 GPU 加速）

### 运行环境

**测试环境：**
- 操作系统：Ubuntu 22.04.1 LTS
- 设备：NVIDIA Quadro P4000
- GPU 内存：8.5 GB
- CUDA：支持

**兼容环境：**
- CPU：所有支持 PyTorch 的平台（Windows、macOS、Linux）
- GPU：NVIDIA GPU with CUDA Compute Capability 3.5+

### 快速安装

```bash
# 克隆或下载项目
cd transformer-assignment

# 安装依赖
pip install -r requirements.txt
```

**依赖列表：**
- `torch>=2.0.0` - PyTorch 深度学习框架
- `matplotlib>=3.8.0` - 数据可视化
- `numpy>=1.24.0` - 数值计算
- `tqdm>=4.65.0` - 进度条显示
- `datasets>=2.16.0`（可选）- Hugging Face 数据集

---

## 使用方法

### 1. 基础训练（Encoder-only 语言模型）

```bash
# 使用默认参数训练（示例文本，2层，128维，4头）
python src/train.py

# 自定义超参数
python src/train.py \
    --epochs 20 \
    --batch_size 32 \
    --d_model 128 \
    --num_heads 4 \
    --num_layers 2 \
    --dropout 0.1 \
    --lr 3e-4
```

### 2. 使用自己的文本文件

```bash
# 从文件加载文本数据
python src/train.py --text_file data/tiny_shakespeare.txt --epochs 30

# 调整序列长度
python src/train.py --text_file data/wikitext-2.txt --seq_len 128
```

### 3. 训练 Encoder-Decoder 模型

```bash
# 启用解码器（用于序列到序列任务）
python src/train.py \
    --use_decoder \
    --d_model 256 \
    --num_heads 8 \
    --num_layers 4 \
    --d_ff 1024 \
    --epochs 30
```

### 4. 高级选项

```bash
# 使用学习率调度器（Warmup + 余弦退火）
python src/train.py \
    --use_scheduler \
    --warmup_steps 1000 \
    --epochs 50

# 调整权重衰减和随机种子
python src/train.py \
    --weight_decay 0.01 \
    --seed 42
```

### 5. 完整参数列表

```bash
python src/train.py --help
```

**数据参数：**
- `--text_file`：文本文件路径（默认使用示例文本）
- `--seq_len`：序列长度（默认 64）

**模型参数：**
- `--d_model`：模型维度（默认 128）
- `--num_heads`：注意力头数（默认 4）
- `--d_ff`：前馈网络维度（默认 512）
- `--num_layers`：Transformer 层数（默认 2）
- `--dropout`：Dropout 概率（默认 0.1）
- `--use_decoder`：使用 Encoder-Decoder 架构

**训练参数：**
- `--batch_size`：批次大小（默认 32）
- `--epochs`：训练轮数（默认 20）
- `--lr`：学习率（默认 3e-4）
- `--weight_decay`：权重衰减（默认 0.01）
- `--use_scheduler`：使用学习率调度器
- `--warmup_steps`：Warmup 步数（默认 1000）
- `--seed`：随机种子（默认 42）

---

##  模型架构详解

### Encoder-only 模型（语言建模）

```
输入 Token IDs [B, L]
    ↓
Embedding + 位置编码 [B, L, d_model]
    ↓
┌─────────────────────────────┐
│  Encoder Layer 1            │
│  ├─ Multi-Head Self-Attn   │
│  ├─ Add & Norm              │
│  ├─ Feed-Forward            │
│  └─ Add & Norm              │
└─────────────────────────────┘
    ↓
┌─────────────────────────────┐
│  Encoder Layer 2            │
│  ...                        │
└─────────────────────────────┘
    ↓
线性投影到词汇表 [B, L, vocab_size]
    ↓
Softmax（在损失函数中）
```

**参数量示例：**
- 词汇表大小 40，d_model=128，num_layers=2：**421,220 参数**

### Encoder-Decoder 模型（序列到序列）

```
源序列 [B, L_src]          目标序列 [B, L_tgt]
    ↓                           ↓
Encoder Layers              Decoder Layers
    ↓                       ↙    ↓
编码表示 [B, L_src, d]    Cross-Attention
                                ↓
                        线性投影 [B, L_tgt, vocab]
```

**参数量示例：**
- 词汇表大小 100，d_model=256，num_layers=4：**7,411,812 参数**

---

## 代码测试

### 1. 测试模型前向传播

```bash
cd /path/to/transformer-assignment
python src/model.py
```

**预期输出：**
```
============================================================
测试 Encoder-only 模型
============================================================
输入形状: torch.Size([2, 10])
输出形状: torch.Size([2, 10, 100])
参数量: 421,220

============================================================
测试 Encoder-Decoder 模型
============================================================
源序列形状: torch.Size([2, 10])
目标序列形状: torch.Size([2, 8])
输出形状: torch.Size([2, 8, 100])
参数量: 7,411,812

✓ 所有测试通过！
```

### 2. 快速训练测试（2 个 epoch）

```bash
python src/train.py --epochs 2 --batch_size 16
```

**预期输出：**
- 训练进度条显示
- 每个 epoch 后显示训练和验证 loss/perplexity
- 保存最佳模型至 `results/best_model.pt`
- 生成训练曲线图至 `results/loss_curve.png`

---

## 训练输出

### 训练过程示例

#### encoder-decoder模型
```
python src/train.py --use_decoder --epochs 2 --batch_size 16

使用设备: cuda

============================================================
加载数据
============================================================
使用示例文本（重复 100 次）
数据集统计:
  文本长度: 54,180 字符
  词汇表大小: 40
  序列长度: 64
  样本数量: 54,116
数据集统计:
  文本长度: 6,020 字符
  词汇表大小: 40
  序列长度: 64
  样本数量: 5,956

============================================================
创建模型
============================================================
模型类型: Encoder-Decoder
总参数量: 932,904
可训练参数量: 932,904

============================================================
开始训练
============================================================
Epoch 1: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 3383/3383 [01:58<00:00, 28.49it/s, loss=0.226]

Epoch 1/2
  Train Loss: 0.6834 | Train PPL: 1.98
  Val Loss:   0.0863 | Val PPL:   1.09
  ✓ 保存最佳模型至: results/best_model.pt
Epoch 2: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 3383/3383 [01:58<00:00, 28.66it/s, loss=0.0535]

Epoch 2/2
  Train Loss: 0.1099 | Train PPL: 1.12
  Val Loss:   0.0117 | Val PPL:   1.01
  ✓ 保存最佳模型至: results/best_model.pt
训练曲线已保存至: results/loss_curve.png

最终模型已保存至: results/final_model.pt

============================================================
训练完成！
============================================================
```
#### encoder only 模型


```
python src/train.py --epochs 2 --batch_size 16
使用设备: cuda

============================================================
加载数据
============================================================
使用示例文本（重复 100 次）
数据集统计:
  文本长度: 54,180 字符
  词汇表大小: 40
  序列长度: 64
  样本数量: 54,116
数据集统计:
  文本长度: 6,020 字符
  词汇表大小: 40
  序列长度: 64
  样本数量: 5,956

============================================================
创建模型
============================================================
模型类型: Encoder-only (Language Model)
总参数量: 405,800
可训练参数量: 405,800

============================================================
开始训练
============================================================
Epoch 1: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 3383/3383 [00:54<00:00, 62.30it/s, loss=0.0499]

Epoch 1/2
  Train Loss: 0.5924 | Train PPL: 1.81
  Val Loss:   0.0104 | Val PPL:   1.01
  ✓ 保存最佳模型至: results/best_model.pt
Epoch 2: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 3383/3383 [00:54<00:00, 62.46it/s, loss=0.0105]

Epoch 2/2
  Train Loss: 0.0299 | Train PPL: 1.03
  Val Loss:   0.0002 | Val PPL:   1.00
  ✓ 保存最佳模型至: results/best_model.pt
训练曲线已保存至: results/loss_curve.png

最终模型已保存至: results/final_model.pt

============================================================
训练完成！
============================================================
```

### 输出文件

1. **results/best_model.pt**：最佳模型检查点
   - 包含：模型权重、优化器状态、验证损失、训练参数

2. **results/final_model.pt**：最终模型
   - 包含：模型权重、词汇表映射、训练参数

3. **results/loss_curve.png**：训练曲线
   - 显示训练和验证损失随 epoch 的变化

---

## 核心实现细节

### 1. 缩放点积注意力（Scaled Dot-Product Attention）

```python
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

**关键特性：**
- 缩放因子 `sqrt(d_k)` 防止点积过大
- 支持可选掩码（因果掩码、padding 掩码）
- Dropout 正则化

### 2. 多头注意力（Multi-Head Attention）

```python
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
```

**优势：**
- 在不同子空间捕获多样的依赖关系
- 并行计算所有头，效率高

### 3. 位置编码（Positional Encoding）

```python
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**作用：**
- 为模型注入位置信息
- 固定编码，不需要学习参数

### 4. 残差连接和层归一化

```python
Output = LayerNorm(x + Sublayer(x))
```

**作用：**
- 缓解梯度消失
- 加速训练收敛

### 5. 参数初始化

- 使用 **Xavier Uniform 初始化**（tensor2tensor 标准）
- Embedding 乘以 `sqrt(d_model)` 进行缩放

### 6. 训练技巧

- **梯度裁剪**：`max_norm=1.0` 防止梯度爆炸
- **AdamW 优化器**：`betas=(0.9, 0.98)`
- **学习率调度**：Warmup + 余弦退火（可选）
- **困惑度（Perplexity）**：`exp(loss)` 评估模型性能

---

## 代码注释说明

本项目包含超过 **800 行详细中文注释**，涵盖：

1. **数学公式推导**：每个模块都有对应的数学公式
2. **参数形状说明**：每个张量的形状都有注释
3. **设计思想**：解释为什么这样设计
4. **实现细节**：关键步骤的代码注释
5. **背景知识**：相关概念的科普说明

**示例：**

```python
def forward(self, Q, K, V, mask=None):
    """
    前向传播

    Args:
        Q: Query 矩阵 [B, num_heads, L_q, d_k]
        K: Key 矩阵   [B, num_heads, L_k, d_k]
        V: Value 矩阵 [B, num_heads, L_v, d_k]
        mask: 掩码 [B, 1, L_q, L_k]

    Returns:
        context: 注意力加权后的输出
        attn: 注意力权重
    """
    # Step 1: 计算 Q 和 K 的点积，并除以 sqrt(d_k) 进行缩放
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    ...
```

---

## 实验建议

### 1. 语言建模（Tiny Shakespeare）

```bash
# 下载 Tiny Shakespeare 数据集（如果还没有）
# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
# mkdir -p data && mv input.txt data/

# 运行收敛实验（推荐）
python convergence_experiment.py --epochs 20 --batch_size 64

# 或使用基础训练脚本
python src/train.py \
    --text_file data/input.txt \
    --epochs 50 \
    --d_model 256 \
    --num_heads 8 \
    --num_layers 4 \
    --batch_size 64 \
    --use_scheduler
```

### 2. 消融实验【未做】

**去除位置编码：**
```python
# 在 model.py 中注释掉位置编码
# src_embedded = self.pos_encoding(src_embedded)  # 注释这行
```

**对比不同层数：**
```bash
python src/train.py --num_layers 2 --epochs 20  # 2层
python src/train.py --num_layers 4 --epochs 20  # 4层
python src/train.py --num_layers 6 --epochs 20  # 6层
```

### 3. 超参数调优【未做】

```bash
# 尝试不同的模型维度
python src/train.py --d_model 128 --num_heads 4
python src/train.py --d_model 256 --num_heads 8
python src/train.py --d_model 512 --num_heads 8

# 尝试不同的学习率
python src/train.py --lr 1e-4
python src/train.py --lr 3e-4
python src/train.py --lr 1e-3
```

---

## 参考资料

1. **原始论文**：Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin, "Attention Is All You Need", NeurIPS 2017
2. **tensor2tensor**：Google 的 Transformer 实现
   - https://github.com/tensorflow/tensor2tensor
3. **The Annotated Transformer**：哈佛 NLP 的注释版本
   - http://nlp.seas.harvard.edu/annotated-transformer/


---

## 常见问题

### Q1: 训练速度很慢怎么办？

**A:**
- 检查是否使用 GPU：`torch.cuda.is_available()`
- 减小 batch_size 或序列长度
- 减少模型层数或维度

### Q2: 出现 CUDA out of memory 错误

**A:**
- 减小 `--batch_size`（如 16 或 8）
- 减小 `--seq_len`（如 32 或 64）
- 使用梯度累积（需修改代码）

### Q3: 如何加载训练好的模型？

**A:**
```python
import torch
from src.model import make_transformer_lm

# 加载检查点
checkpoint = torch.load('results/final_model.pt')

# 创建模型
model = make_transformer_lm(
    vocab_size=checkpoint['vocab_size'],
    # ... 其他参数
)

# 加载权重
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

### Q4: 如何生成文本？

**A:**
```python
from src.train import generate_text, TinyTextDataset

# 加载模型和词汇表
checkpoint = torch.load('results/final_model.pt')
# ...（同 Q3）

# 生成文本
text = generate_text(
    model,
    dataset,  # 需要数据集获取字符映射
    device,
    start_text="To be",
    max_len=200,
    temperature=0.8
)
print(text)
```

---

## 许可

本项目仅供学习使用。

---

## 致谢

- 感谢 Vaswani 等人提出的 Transformer 架构
- 感谢 tensor2tensor 项目提供的参考实现
- 感谢《大型模型基础与应用》课程的指导

---

**日期**：2025年11月

如有问题，欢迎提出 Issue 或联系作者。
