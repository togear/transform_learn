"""
Tiny Shakespeare 数据集完整收敛实验
使用 Transformer 模型进行字符级语言建模的收敛性分析
包含详细的训练监控、可视化和收敛性指标
"""

import os
import sys
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import json
import argparse
from datetime import datetime

# 添加 src 目录到路径
sys.path.append('src')
from model import make_transformer_lm
from train import TinyTextDataset


def setup_experiment(args):
    """设置实验环境和参数"""
    print("="*80)
    print(" Tiny Shakespeare 收敛实验")
    print("="*80)

    # 设置随机种子确保可复现性
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 使用设备: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # 创建实验目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = f"experiments/shakespeare_convergence_{timestamp}"
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(f"{exp_dir}/plots", exist_ok=True)
    os.makedirs(f"{exp_dir}/models", exist_ok=True)

    print(f" 实验目录: {exp_dir}")

    return device, exp_dir


def load_shakespeare_data(args):
    """加载和预处理 Tiny Shakespeare 数据"""
    print("\n 加载 Tiny Shakespeare 数据集")
    print("-" * 50)

    data_path = "data/input.txt"
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"未找到 Tiny Shakespeare 数据集！\n"
            f"请确保文件存在: {data_path}\n"
            f"可以运行: curl -O https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        )

    # 读取文本
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"  数据统计:")
    print(f"   总字符数: {len(text):,}")
    print(f"   唯一字符数: {len(set(text))}")

    # 数据划分：训练集(90%), 验证集(10%)
    split_idx = int(len(text) * 0.9)
    train_text = text[:split_idx]
    val_text = text[split_idx:]

    print(f"   训练集: {len(train_text):,} 字符")
    print(f"   验证集: {len(val_text):,} 字符")

    # 创建数据集
    train_dataset = TinyTextDataset(train_text, seq_len=args.seq_len)
    val_dataset = TinyTextDataset(val_text, seq_len=args.seq_len)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    return train_dataset, val_dataset, train_loader, val_loader


def create_model(vocab_size, args, device):
    """创建和初始化模型"""
    print(f"\n  创建 Transformer 模型")
    print("-" * 50)

    model = make_transformer_lm(
        vocab_size=vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        dropout=args.dropout
    )

    model = model.to(device)

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f" 模型架构:")
    print(f"   总参数量: {total_params:,}")
    print(f"   可训练参数: {trainable_params:,}")
    print(f"   模型大小: {total_params * 4 / 1e6:.2f} MB (float32)")

    return model


def setup_training(model, args):
    """设置训练组件（优化器、损失函数、调度器）"""
    print(f"\n  配置训练组件")
    print("-" * 50)

    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),  # GPT-3 的设置
        eps=1e-8,
        weight_decay=args.weight_decay
    )

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # 学习率调度器（余弦退火 + warmup）
    def get_lr_scheduler(optimizer, warmup_steps, total_steps):
        def lr_lambda(step):
            if step < warmup_steps:
                # Warmup phase: 线性增长
                return step / warmup_steps
            else:
                # 余弦退火
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return 0.5 * (1 + math.cos(math.pi * progress))

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    total_steps = args.epochs * args.steps_per_epoch if hasattr(args, 'steps_per_epoch') else args.epochs * 1000
    scheduler = get_lr_scheduler(optimizer, args.warmup_steps, total_steps)

    print(f"  训练配置:")
    print(f"   优化器: AdamW")
    print(f"   初始学习率: {args.lr}")
    print(f"   权重衰减: {args.weight_decay}")
    print(f"   Warmup 步数: {args.warmup_steps}")

    return optimizer, criterion, scheduler


class ConvergenceTracker:
    """收敛性跟踪器"""

    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.wait = 0
        self.stopped_epoch = 0

        # 记录各种指标
        self.train_losses = []
        self.val_losses = []
        self.train_ppls = []
        self.val_ppls = []
        self.learning_rates = []
        self.grad_norms = []
        self.step_times = []

    def update(self, train_loss, val_loss, train_ppl, val_ppl, lr, grad_norm, step_time):
        """更新跟踪指标"""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_ppls.append(train_ppl)
        self.val_ppls.append(val_ppl)
        self.learning_rates.append(lr)
        self.grad_norms.append(grad_norm)
        self.step_times.append(step_time)

        # 检查是否有改进
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.wait = 0
            return False  # 未收敛
        else:
            self.wait += 1
            return self.wait >= self.patience  # 是否达到早停条件

    def get_convergence_metrics(self):
        """计算收敛性指标"""
        if len(self.val_losses) < 10:
            return {}

        # 最近 10 个 epoch 的损失变化
        recent_val_losses = self.val_losses[-10:]
        loss_variance = np.var(recent_val_losses)
        loss_trend = np.polyfit(range(len(recent_val_losses)), recent_val_losses, 1)[0]

        # 损失平滑性（相邻 epoch 之间的平均变化）
        loss_smoothness = np.mean([abs(self.val_losses[i] - self.val_losses[i-1])
                                  for i in range(1, len(self.val_losses))])

        return {
            'best_val_loss': self.best_loss,
            'final_val_loss': self.val_losses[-1],
            'loss_variance': loss_variance,
            'loss_trend': loss_trend,
            'loss_smoothness': loss_smoothness,
            'converged_at_epoch': len(self.val_losses) - self.wait if self.wait >= self.patience else None
        }


def train_epoch(model, train_loader, optimizer, criterion, device, epoch, tracker, scheduler=None):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    total_tokens = 0
    grad_norms = []

    start_time = time.time()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}")

    for batch_idx, (x, y) in enumerate(pbar):
        x, y = x.to(device), y.to(device)
        batch_size = x.size(0)

        # 前向传播
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))

        # 反向传播
        loss.backward()

        # 梯度裁剪和记录
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        grad_norms.append(grad_norm.item())

        optimizer.step()
        if scheduler:
            scheduler.step()

        # 累积统计
        total_loss += loss.item() * batch_size
        total_tokens += batch_size

        # 更新进度条
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'lr': f'{current_lr:.2e}',
            'grad_norm': f'{grad_norm.item():.3f}'
        })

    epoch_time = time.time() - start_time
    avg_loss = total_loss / total_tokens
    avg_grad_norm = np.mean(grad_norms)
    perplexity = math.exp(min(avg_loss, 20))

    return avg_loss, perplexity, avg_grad_norm, epoch_time


def validate(model, val_loader, criterion, device):
    """验证模型"""
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            batch_size = x.size(0)

            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))

            total_loss += loss.item() * batch_size
            total_tokens += batch_size

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(min(avg_loss, 20))

    return avg_loss, perplexity


def generate_text(model, dataset, device, start_text="First Citizen:", max_len=200, temperature=0.8):
    """生成样本文本"""
    model.eval()

    # 将起始文本转换为索引
    chars = [dataset.char2idx.get(c, 0) for c in start_text]
    input_seq = torch.tensor(chars, dtype=torch.long).unsqueeze(0).to(device)

    generated = start_text

    with torch.no_grad():
        for _ in range(max_len):
            if input_seq.size(1) > dataset.seq_len:
                input_seq = input_seq[:, -dataset.seq_len:]

            logits = model(input_seq)
            next_token_logits = logits[0, -1, :] / temperature

            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            next_char = dataset.idx2char[next_token.item()]
            generated += next_char

            input_seq = torch.cat([input_seq, next_token.unsqueeze(0)], dim=1)

    return generated


def plot_convergence_analysis(tracker, exp_dir):
    """绘制收敛性分析图表"""
    epochs = range(1, len(tracker.train_losses) + 1)

    # 创建多子图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Tiny Shakespeare 收敛性分析', fontsize=16, fontweight='bold')

    # 1. 损失曲线
    axes[0, 0].plot(epochs, tracker.train_losses, 'b-', label='Training Loss', linewidth=2)
    axes[0, 0].plot(epochs, tracker.val_losses, 'r-', label='Validation Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('训练和验证损失')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. 困惑度曲线
    axes[0, 1].plot(epochs, tracker.train_ppls, 'b-', label='Training PPL', linewidth=2)
    axes[0, 1].plot(epochs, tracker.val_ppls, 'r-', label='Validation PPL', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Perplexity')
    axes[0, 1].set_title('困惑度 (Perplexity)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_yscale('log')

    # 3. 学习率曲线
    axes[0, 2].plot(epochs, tracker.learning_rates, 'g-', linewidth=2)
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Learning Rate')
    axes[0, 2].set_title('学习率调度')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_yscale('log')

    # 4. 梯度范数
    axes[1, 0].plot(epochs, tracker.grad_norms, 'purple', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Gradient Norm')
    axes[1, 0].set_title('梯度范数')
    axes[1, 0].grid(True, alpha=0.3)

    # 5. 训练时间
    axes[1, 1].plot(epochs, tracker.step_times, 'orange', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Time (seconds)')
    axes[1, 1].set_title('每轮训练时间')
    axes[1, 1].grid(True, alpha=0.3)

    # 6. 收敛性指标
    if len(tracker.val_losses) >= 10:
        # 计算移动平均以显示趋势
        window_size = min(10, len(tracker.val_losses) // 4)
        val_loss_ma = np.convolve(tracker.val_losses,
                                 np.ones(window_size)/window_size, mode='valid')
        ma_epochs = epochs[window_size-1:]

        axes[1, 2].plot(epochs, tracker.val_losses, 'r-', alpha=0.5, label='原始验证损失')
        axes[1, 2].plot(ma_epochs, val_loss_ma, 'r-', linewidth=3, label=f'移动平均 (窗口={window_size})')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Validation Loss')
        axes[1, 2].set_title('验证损失趋势')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图表
    plot_path = f"{exp_dir}/plots/convergence_analysis.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f" 收敛性分析图表已保存: {plot_path}")
    plt.close()


def save_experiment_results(args, tracker, model, dataset, exp_dir):
    """保存实验结果"""
    print(f"\n 保存实验结果")
    print("-" * 50)

    # 收敛性指标
    convergence_metrics = tracker.get_convergence_metrics()

    # 实验配置和结果
    results = {
        'experiment_config': {
            'dataset': 'tiny_shakespeare',
            'model_type': 'transformer_encoder_only',
            'vocab_size': dataset.vocab_size,
            'seq_len': args.seq_len,
            'd_model': args.d_model,
            'num_heads': args.num_heads,
            'd_ff': args.d_ff,
            'num_layers': args.num_layers,
            'dropout': args.dropout,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'weight_decay': args.weight_decay,
            'warmup_steps': args.warmup_steps,
            'epochs': args.epochs,
            'seed': args.seed
        },
        'training_history': {
            'train_losses': tracker.train_losses,
            'val_losses': tracker.val_losses,
            'train_perplexities': tracker.train_ppls,
            'val_perplexities': tracker.val_ppls,
            'learning_rates': tracker.learning_rates,
            'gradient_norms': tracker.grad_norms,
            'epoch_times': tracker.step_times
        },
        'convergence_metrics': convergence_metrics,
        'final_results': {
            'total_epochs': len(tracker.train_losses),
            'best_val_loss': min(tracker.val_losses),
            'best_val_perplexity': min(tracker.val_ppls),
            'final_val_loss': tracker.val_losses[-1],
            'final_val_perplexity': tracker.val_ppls[-1],
            'total_training_time': sum(tracker.step_times),
            'average_epoch_time': np.mean(tracker.step_times)
        }
    }

    # 保存 JSON 结果
    results_path = f"{exp_dir}/experiment_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # ��存模型
    model_path = f"{exp_dir}/models/final_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': dataset.vocab_size,
        'char2idx': dataset.char2idx,
        'idx2char': dataset.idx2char,
        'model_config': {
            'd_model': args.d_model,
            'num_heads': args.num_heads,
            'd_ff': args.d_ff,
            'num_layers': args.num_layers,
            'dropout': args.dropout
        }
    }, model_path)

    # 生成并保存样本文本
    sample_text = generate_text(model, dataset, torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                               start_text="First Citizen:", max_len=500, temperature=0.8)

    sample_path = f"{exp_dir}/generated_sample.txt"
    with open(sample_path, 'w', encoding='utf-8') as f:
        f.write("Generated Text Sample:\n")
        f.write("=" * 50 + "\n\n")
        f.write(sample_text)

    print(f" 实验结果已保存:")
    print(f"   结果文件: {results_path}")
    print(f"   模型文件: {model_path}")
    print(f"   样本文本: {sample_path}")

    return results


def print_experiment_summary(results):
    """打印实验摘要"""
    print("\n" + "="*80)
    print(" 实验结果摘要")
    print("="*80)

    config = results['experiment_config']
    final = results['final_results']
    convergence = results['convergence_metrics']

    print(f" 模型配置:")
    print(f"   词汇表大小: {config['vocab_size']}")
    print(f"   序列长度: {config['seq_len']}")
    print(f"   模型维度: {config['d_model']}")
    print(f"   注意力头数: {config['num_heads']}")
    print(f"   层数: {config['num_layers']}")

    print(f"\n 训练结果:")
    print(f"   总训练轮数: {final['total_epochs']}")
    print(f"   最佳验证损失: {final['best_val_loss']:.4f}")
    print(f"   最佳验证困惑度: {final['best_val_perplexity']:.2f}")
    print(f"   最终验证损失: {final['final_val_loss']:.4f}")
    print(f"   最终验证困惑度: {final['final_val_perplexity']:.2f}")

    print(f"\n  训练时间:")
    print(f"   总训练时间: {final['total_training_time']:.1f} 秒")
    print(f"   平均每轮时间: {final['average_epoch_time']:.1f} 秒")

    if convergence:
        print(f"\n 收敛性分析:")
        print(f"   损失方差（最近10轮）: {convergence.get('loss_variance', 'N/A'):.6f}")
        print(f"   损失趋势（最近10轮）: {convergence.get('loss_trend', 'N/A'):.6f}")
        print(f"   损失平滑性: {convergence.get('loss_smoothness', 'N/A'):.6f}")
        if convergence.get('converged_at_epoch'):
            print(f"   收敛轮数: {convergence['converged_at_epoch']}")
        else:
            print(f"   状态: 训练完成但未检测到收敛")


def main():
    """主实验函数"""
    parser = argparse.ArgumentParser(description='Tiny Shakespeare 收敛实验')

    # 模型参数
    parser.add_argument('--d_model', type=int, default=256, help='模型维度')
    parser.add_argument('--num_heads', type=int, default=8, help='注意力头数')
    parser.add_argument('--d_ff', type=int, default=1024, help='前馈网络维度')
    parser.add_argument('--num_layers', type=int, default=6, help='Transformer 层数')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout 概率')
    parser.add_argument('--seq_len', type=int, default=128, help='序列长度')

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=3e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='权重衰减')
    parser.add_argument('--warmup_steps', type=int, default=2000, help='Warmup 步数')

    # 实验参数
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--patience', type=int, default=10, help='早停耐心值')
    parser.add_argument('--min_delta', type=float, default=1e-4, help='最小改进阈值')

    args = parser.parse_args()

    # 设置实验环境
    device, exp_dir = setup_experiment(args)

    # 加载数据
    train_dataset, val_dataset, train_loader, val_loader = load_shakespeare_data(args)

    # 创建模型
    model = create_model(train_dataset.vocab_size, args, device)

    # 设置训练组件
    args.steps_per_epoch = len(train_loader)
    optimizer, criterion, scheduler = setup_training(model, args)

    # 初始化收敛跟踪器
    tracker = ConvergenceTracker(patience=args.patience, min_delta=args.min_delta)

    print(f"\n 开始训练")
    print("-" * 50)

    best_val_loss = float('inf')

    # 训练循环
    for epoch in range(1, args.epochs + 1):
        # 训练
        train_loss, train_ppl, grad_norm, epoch_time = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch, tracker, scheduler
        )

        # 验证
        val_loss, val_ppl = validate(model, val_loader, criterion, device)

        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']

        # 更新跟踪器
        early_stop = tracker.update(train_loss, val_loss, train_ppl, val_ppl,
                                   current_lr, grad_norm, epoch_time)

        # 打印进度
        print(f"\nEpoch {epoch:3d}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Train PPL: {train_ppl:7.2f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val PPL:   {val_ppl:7.2f}")
        print(f"  LR: {current_lr:.2e} | Grad Norm: {grad_norm:.3f} | Time: {epoch_time:.1f}s")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = f"{exp_dir}/models/best_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'args': args
            }, best_model_path)
            print(f"   最佳模型已保存!")

        # 每10轮生成样本文本
        if epoch % 10 == 0:
            print(f"\n 生成样本 (Epoch {epoch}):")
            sample = generate_text(model, train_dataset, device,
                                 start_text="ROMEO:", max_len=150, temperature=0.8)
            print(f"   {sample[:100]}...")

        # 检查早停
        if early_stop:
            print(f"\n  Early stopping at epoch {epoch}")
            print(f"    No improvement for {args.patience} epochs")
            break

    # 绘制和保存结果
    plot_convergence_analysis(tracker, exp_dir)
    results = save_experiment_results(args, tracker, model, train_dataset, exp_dir)
    print_experiment_summary(results)

    print(f"\n 实验完成！结果保存在: {exp_dir}")


if __name__ == "__main__":
    main()