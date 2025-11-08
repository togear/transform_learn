"""
Transformer 训练脚本
支持 Encoder-only 和 Encoder-Decoder 模型的训练
包含数据加载、训练循环、验证、可视化等完整功能
"""

import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

from model import make_transformer_lm, make_transformer_seq2seq


# ======================================================================
# 数据集类
# ======================================================================
class TinyTextDataset(Dataset):
    """
    字符级语言模型数据集

    用于 Tiny Shakespeare 或其他文本数据的字符级建模
    将文本转换为字符索引序列，并创建滑动窗口样本

    特点：
        - 字符级建模（每个字符作为一个 token）
        - 自回归方式：输入为 [t, t+1, ..., t+seq_len-1]
                      目标为 [t+1, t+2, ..., t+seq_len]
    """

    def __init__(self, text, seq_len=64):
        """
        初始化数据集

        Args:
            text: 输入文本字符串
            seq_len: 序列长度（上下文窗口大小）
        """
        # 构建词汇表：提取所有唯一字符并排序
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)

        # 创建字符到索引、索引到字符的映射
        self.char2idx = {ch: i for i, ch in enumerate(chars)}
        self.idx2char = {i: ch for i, ch in enumerate(chars)}

        # 将文本转换为索引序列
        self.data = torch.tensor([self.char2idx[c] for c in text], dtype=torch.long)
        self.seq_len = seq_len

        print(f"数据集统计:")
        print(f"  文本长度: {len(text):,} 字符")
        print(f"  词汇表大小: {self.vocab_size}")
        print(f"  序列长度: {seq_len}")
        print(f"  样本数量: {len(self):,}")

    def __len__(self):
        """返回数据集大小（可以生成的样本数）"""
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        """
        获取一个训练样本

        Args:
            idx: 样本索引

        Returns:
            x: 输入序列 [seq_len]
            y: 目标序列 [seq_len]（相对输入向右移动一位）
        """
        # 输入：从 idx 开始的 seq_len 个字符
        x = self.data[idx : idx + self.seq_len]
        # 目标：从 idx+1 开始的 seq_len 个字符（即预测下一个字符）
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        return x, y


# ======================================================================
# 训练函数
# ======================================================================
def train_epoch(model, dataloader, optimizer, criterion, device, epoch, use_decoder=False):
    """
    训练一个 epoch

    Args:
        model: Transformer 模型
        dataloader: 数据加载器
        optimizer: 优化器
        criterion: 损失函数
        device: 设备（cuda/cpu）
        epoch: 当前 epoch 编号
        use_decoder: 是否使用解码器模式

    Returns:
        avg_loss: 平均训练损失
        perplexity: 困惑度（Perplexity）
    """
    model.train()  # 设置为训练模式（启用 dropout 等）
    total_loss = 0
    total_tokens = 0

    # 使用 tqdm 显示进度条
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, (x, y) in enumerate(pbar):
        # 将数据移到指定设备
        x, y = x.to(device), y.to(device)
        batch_size = x.size(0)

        # 梯度清零
        optimizer.zero_grad()

        # 前向传播
        if use_decoder:
            # Encoder-Decoder 模式：需要创建目标序列的因果掩码
            tgt = y[:, :-1]  # 目标输入（去掉最后一个 token）
            tgt_y = y[:, 1:]  # 目标输出（去掉第一个 token）

            # 生成因果掩码
            tgt_len = tgt.size(1)
            tgt_mask = model.generate_square_subsequent_mask(tgt_len).to(device)
            tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, tgt_len, tgt_len]

            # 前向传播
            logits = model(x, tgt, tgt_mask=tgt_mask)

            # 计算损失
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_y.reshape(-1))
        else:
            # Encoder-only 模式（语言建模）
            logits = model(x)

            # 计算损失
            # logits: [batch_size, seq_len, vocab_size]
            # y: [batch_size, seq_len]
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))

        # 反向传播
        loss.backward()

        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 更新参数
        optimizer.step()

        # 累积损失
        total_loss += loss.item() * batch_size
        total_tokens += batch_size

        # 更新进度条
        pbar.set_postfix({'loss': loss.item()})

    # 计算平均损失和困惑度
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(min(avg_loss, 20))  # 限制最大值防止溢出

    return avg_loss, perplexity


# ======================================================================
# 验证函数
# ======================================================================
def validate(model, dataloader, criterion, device, use_decoder=False):
    """
    在验证集上评估模型

    Args:
        model: Transformer 模型
        dataloader: 验证数据加载器
        criterion: 损失函数
        device: 设备
        use_decoder: 是否使用解码器模式

    Returns:
        avg_loss: 平均验证损失
        perplexity: 困惑度
    """
    model.eval()  # 设置为评估模式（禁用 dropout 等）
    total_loss = 0
    total_tokens = 0

    # 不计算梯度（节省内存和计算）
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            batch_size = x.size(0)

            # 前向传播
            if use_decoder:
                tgt = y[:, :-1]
                tgt_y = y[:, 1:]
                tgt_len = tgt.size(1)
                tgt_mask = model.generate_square_subsequent_mask(tgt_len).to(device)
                tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0)
                logits = model(x, tgt, tgt_mask=tgt_mask)
                loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_y.reshape(-1))
            else:
                logits = model(x)
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))

            total_loss += loss.item() * batch_size
            total_tokens += batch_size

    # 计算平均损失和困惑度
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(min(avg_loss, 20))

    return avg_loss, perplexity


# ======================================================================
# 文本生成函数（用于测试模型）
# ======================================================================
def generate_text(model, dataset, device, start_text="To be", max_len=100, temperature=1.0):
    """
    使用训练好的模型生成文本

    Args:
        model: 训练好的模型
        dataset: 数据集（用于字符映射）
        device: 设备
        start_text: 起始文本
        max_len: 生成的最大长度
        temperature: 温度参数（控制随机性，越大越随机）

    Returns:
        generated_text: 生成的文本
    """
    model.eval()

    # 将起始文本转换为索引
    chars = [dataset.char2idx.get(c, 0) for c in start_text]
    input_seq = torch.tensor(chars, dtype=torch.long).unsqueeze(0).to(device)

    generated = start_text

    with torch.no_grad():
        for _ in range(max_len):
            # 前向传播
            logits = model(input_seq)

            # 获取最后一个位置的 logits
            next_token_logits = logits[0, -1, :] / temperature

            # 采样下一个 token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # 转换为字符并添加到生成文本
            next_char = dataset.idx2char[next_token.item()]
            generated += next_char

            # 更新输入序列（添加新生成的 token）
            next_token = next_token.unsqueeze(0)
            input_seq = torch.cat([input_seq, next_token], dim=1)

            # 限制输入序列长度（保留最后 seq_len 个 token）
            if input_seq.size(1) > dataset.seq_len:
                input_seq = input_seq[:, -dataset.seq_len:]

    return generated


# ======================================================================
# 可视化训练曲线
# ======================================================================
def plot_training_curves(train_losses, val_losses, save_path="results/loss_curve.png"):
    """
    绘制并保存训练和验证损失曲线

    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        save_path: 保存路径
    """
    plt.figure(figsize=(10, 6))

    epochs = range(1, len(train_losses) + 1)

    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss Curves', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # 确保保存目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"训练曲线已保存至: {save_path}")
    plt.close()


# ======================================================================
# 主训练流程
# ======================================================================
def main(args):
    """
    主训练函数

    Args:
        args: 命令行参数
    """
    # 设置随机种子（保证可复现性）
    torch.manual_seed(args.seed)

    # 选择设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # ========== 加载数据 ==========
    print("\n" + "="*60)
    print("加载数据")
    print("="*60)

    # 加载数据的多种方式
    if args.dataset == 'wikitext2':
        # WikiText-2 数据集
        train_file = args.train_file or 'data/wikitext2_train.txt'
        val_file = args.val_file or 'data/wikitext2_val.txt'

        if not os.path.exists(train_file) or not os.path.exists(val_file):
            raise FileNotFoundError(
                f"WikiText-2 数据集文件未找到！\n"
                f"请确保以下文件存在：\n"
                f"  - {train_file}\n"
                f"  - {val_file}\n"
                f"可以运行以下命令下载：\n"
                f"  cd data && wget https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt -O wikitext2_train.txt"
            )

        with open(train_file, 'r', encoding='utf-8') as f:
            train_text = f.read()
        with open(val_file, 'r', encoding='utf-8') as f:
            val_text = f.read()

        print(f"从 WikiText-2 加载:")
        print(f"  训练集: {train_file} ({len(train_text):,} 字符)")
        print(f"  验证集: {val_file} ({len(val_text):,} 字符)")

    elif args.train_file and args.val_file:
        # 自定义分开的训练集和验证集
        with open(args.train_file, 'r', encoding='utf-8') as f:
            train_text = f.read()
        with open(args.val_file, 'r', encoding='utf-8') as f:
            val_text = f.read()
        print(f"从文件加载:")
        print(f"  训练集: {args.train_file}")
        print(f"  验证集: {args.val_file}")

    elif args.text_file and os.path.exists(args.text_file):
        # 单个文本文件，自动划分训练集和验证集
        with open(args.text_file, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"从文件加载: {args.text_file}")

        # 划分训练集和验证集
        split_idx = int(len(text) * 0.9)
        train_text = text[:split_idx]
        val_text = text[split_idx:]

    else:
        # 使用示例文本
        text = """To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die—to sleep,
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep, perchance to dream—ay, there's the rub:
For in that sleep of death what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause—there's the respect
That makes calamity of so long life.""" * 100  # 重复以增加数据量
        print("使用示例文本（重复 100 次）")

        # 划分训练集和验证集
        split_idx = int(len(text) * 0.9)
        train_text = text[:split_idx]
        val_text = text[split_idx:]

    # 创建数据集
    train_dataset = TinyTextDataset(train_text, seq_len=args.seq_len)
    val_dataset = TinyTextDataset(val_text, seq_len=args.seq_len)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Windows 上设置为 0
        pin_memory=True if device.type == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )

    # ========== 创建模型 ==========
    print("\n" + "="*60)
    print("创建模型")
    print("="*60)

    if args.use_decoder:
        print("模型类型: Encoder-Decoder")
        model = make_transformer_seq2seq(
            vocab_size=train_dataset.vocab_size,
            d_model=args.d_model,
            num_heads=args.num_heads,
            d_ff=args.d_ff,
            num_encoder_layers=args.num_layers,
            num_decoder_layers=args.num_layers,
            dropout=args.dropout
        )
    else:
        print("模型类型: Encoder-only (Language Model)")
        model = make_transformer_lm(
            vocab_size=train_dataset.vocab_size,
            d_model=args.d_model,
            num_heads=args.num_heads,
            d_ff=args.d_ff,
            num_layers=args.num_layers,
            dropout=args.dropout
        )

    model = model.to(device)

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # ========== 设置优化器和损失函数 ==========
    # 使用 AdamW 优化器（带权重衰减的 Adam）
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.98),  # tensor2tensor 推荐的参数
        eps=1e-9,
        weight_decay=args.weight_decay
    )

    # 使用交叉熵损失
    criterion = nn.CrossEntropyLoss()

    # 学习率调度器（可选）
    if args.use_scheduler:
        # Warmup + 余弦退火
        def lr_lambda(step):
            warmup_steps = args.warmup_steps
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (args.epochs * len(train_loader) - warmup_steps)
                return 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = None

    # ========== 训练循环 ==========
    print("\n" + "="*60)
    print("开始训练")
    print("="*60)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        # 训练
        train_loss, train_ppl = train_epoch(
            model, train_loader, optimizer, criterion,
            device, epoch, use_decoder=args.use_decoder
        )

        # 验证
        val_loss, val_ppl = validate(
            model, val_loader, criterion,
            device, use_decoder=args.use_decoder
        )

        # 记录损失
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # 打印信息
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Train PPL: {train_ppl:.2f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val PPL:   {val_ppl:.2f}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs("results", exist_ok=True)
            model_path = "results/best_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'args': args
            }, model_path)
            print(f"  ✓ 保存最佳模型至: {model_path}")

        # 更新学习率
        if scheduler:
            scheduler.step()

        # 生成样本文本（每 5 个 epoch）
        if epoch % 5 == 0 and not args.use_decoder:
            print("\n生成样本文本:")
            sample = generate_text(
                model, train_dataset, device,
                start_text="To be",
                max_len=200,
                temperature=0.8
            )
            print(f"  {sample}")

    # ========== 保存训练曲线 ==========
    plot_training_curves(train_losses, val_losses)

    # ========== 保存最终模型 ==========
    final_model_path = "results/final_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': train_dataset.vocab_size,
        'char2idx': train_dataset.char2idx,
        'idx2char': train_dataset.idx2char,
        'args': args
    }, final_model_path)
    print(f"\n最终模型已保存至: {final_model_path}")

    print("\n" + "="*60)
    print("训练完成！")
    print("="*60)


# ======================================================================
# 命令行参数解析
# ======================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='训练 Transformer 模型')

    # 数据参数
    parser.add_argument('--text_file', type=str, default=None,
                        help='文本文件路径（如果不提供则使用示例文本）')
    parser.add_argument('--train_file', type=str, default=None,
                        help='训练集文件路径')
    parser.add_argument('--val_file', type=str, default=None,
                        help='验证集文件路径')
    parser.add_argument('--dataset', type=str, default='custom',
                        choices=['custom', 'wikitext2'],
                        help='数据集类型')
    parser.add_argument('--seq_len', type=int, default=64,
                        help='序列长度')

    # 模型参数
    parser.add_argument('--d_model', type=int, default=128,
                        help='模型维度')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='注意力头数')
    parser.add_argument('--d_ff', type=int, default=512,
                        help='前馈网络维度')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Transformer 层数')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout 概率')
    parser.add_argument('--use_decoder', action='store_true',
                        help='使用 Encoder-Decoder 架构')

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--epochs', type=int, default=20,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='权重衰减')
    parser.add_argument('--use_scheduler', action='store_true',
                        help='使用学习率调度器')
    parser.add_argument('--warmup_steps', type=int, default=1000,
                        help='Warmup 步数')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')

    args = parser.parse_args()

    # 运行训练
    main(args)
