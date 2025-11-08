# 快速开始指南

本文档提供最简洁的上手步骤，帮助您快速开始使用 Transformer 实现。

##  30 秒快速开始

```bash
# 1. 安装依赖
pip install torch matplotlib tqdm

# 2. 测试模型
python src/model.py

# 3. 快速训练（2个epoch）
python src/train.py --epochs 2 --batch_size 16

# 4. 查看结果
ls results/
```

##  常用命令

### 基础训练
```bash
# 默认配置（CPU，示例文本）
python src/train.py

# 使用GPU
python src/train.py  # 自动检测CUDA

# 完整训练（20 epochs）
python src/train.py --epochs 20 --batch_size 32
```

### 使用自己的数据
```bash
# 从文件读取
python src/train.py --text_file your_text.txt --epochs 30

# Tiny Shakespeare 收敛实验
python convergence_experiment.py --epochs 20 --batch_size 64
```

### Encoder-Decoder 模型
```bash
# 序列到序列任务
python src/train.py --use_decoder --d_model 256 --num_heads 8
```

##  输出文件

训练完成后，在 `results/` 目录下：
- `loss_curve.png` - 训练曲线图
- `best_model.pt` - 最佳模型
- `final_model.pt` - 最终模型

## 查看详细文档

更多详细信息请参考 [README.md](README.md)

## 性能优化

```bash
# 减少batch size以节省内存
python src/train.py --batch_size 8

# 减少模型大小
python src/train.py --d_model 64 --num_layers 2

# 使用学习率调度器
python src/train.py --use_scheduler --epochs 50
```

## 提示

- 第一次运行会使用示例文本（莎士比亚片段重复100次）
- 训练时会每5个epoch生成样本文本
- 所有超参数都可以通过命令行调整
- 使用 `--help` 查看所有可用参数

## 常见问题快速解决

**问题：模块找不到**
```bash
pip install torch matplotlib tqdm
```

**问题：CUDA out of memory**
```bash
python src/train.py --batch_size 8 --seq_len 32
```

**问题：训练太慢**
```bash
# 减少数据量和模型大小
python src/train.py --epochs 5 --num_layers 2
```

---

Happy Training! 
