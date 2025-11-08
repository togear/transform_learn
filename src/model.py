"""
Transformer 模型实现
参照 tensor2tensor 风格实现 Encoder-only 和 Encoder-Decoder 架构
包含详细的数学推导和实现注释
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# Scaled Dot-Product Attention
# ======================================================================
class ScaledDotProductAttention(nn.Module):
    """
    缩放点积注意力机制

    数学公式：
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    参数说明：
        - Q (Query): [batch_size, num_heads, seq_len_q, d_k]
        - K (Key):   [batch_size, num_heads, seq_len_k, d_k]
        - V (Value): [batch_size, num_heads, seq_len_v, d_k]
        - mask:      可选的掩码张量，用于屏蔽某些位置的注意力

    返回：
        - context:   加权后的值向量 [batch_size, num_heads, seq_len_q, d_k]
        - attn:      注意力权重矩阵 [batch_size, num_heads, seq_len_q, seq_len_k]
    """

    def __init__(self, dropout=0.1):
        """
        初始化缩放点积注意力

        Args:
            dropout: Dropout 概率，用于注意力权重的正则化
        """
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        """
        前向传播

        Args:
            Q: Query 矩阵 [B, num_heads, L_q, d_k]
            K: Key 矩阵   [B, num_heads, L_k, d_k]
            V: Value 矩阵 [B, num_heads, L_v, d_k]
            mask: 掩码 [B, 1, L_q, L_k] 或 [B, num_heads, L_q, L_k]

        Returns:
            context: 注意力加权后的输出
            attn: 注意力权重
        """
        # 获取 Key 的维度 d_k
        d_k = Q.size(-1)

        # Step 1: 计算 Q 和 K 的点积，并除以 sqrt(d_k) 进行缩放
        # scores: [B, num_heads, L_q, L_k]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        # Step 2: 如果提供了 mask，则将被屏蔽的位置设置为一个很大的负数
        # 这样经过 softmax 后，这些位置的权重接近 0
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Step 3: 对最后一维（Key 维度）应用 softmax，得到注意力权重
        # attn: [B, num_heads, L_q, L_k]
        attn = F.softmax(scores, dim=-1)

        # Step 4: 应用 dropout 进行正则化（训练时随机丢弃部分连接）
        attn = self.dropout(attn)

        # Step 5: 将注意力权重与 Value 相乘，得到加权后的输出
        # context: [B, num_heads, L_q, d_k]
        context = torch.matmul(attn, V)

        return context, attn


# ======================================================================
# Multi-Head Attention
# ======================================================================
class MultiHeadAttention(nn.Module):
    """
    多头注意力机制

    核心思想：
        将 d_model 维度分成 num_heads 个子空间，每个子空间独立计算注意力，
        最后将所有头的输出拼接起来并通过线性变换。

    数学公式：
        MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
        where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)

    优势：
        - 允许模型在不同的表示子空间捕获信息
        - 不同的头可以关注不同的依赖关系（如局部/全局、语法/语义）
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        """
        初始化多头注意力

        Args:
            d_model: 模型维度（必须能被 num_heads 整除）
            num_heads: 注意力头的数量
            dropout: Dropout 概率
        """
        super(MultiHeadAttention, self).__init__()

        # 确保 d_model 能被 num_heads 整除
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # 每个头的维度
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.d_model = d_model

        # 定义 Q, K, V 的线性变换矩阵
        # 注意：这里使用单个大矩阵而不是 num_heads 个小矩阵，计算效率更高
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)

        # 输出的线性变换矩阵 W^O
        self.fc_out = nn.Linear(d_model, d_model, bias=False)

        # 注意力层和输出层的 dropout
        self.attn_layer = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        """
        多头注意力的前向传播

        Args:
            Q: Query [batch_size, seq_len_q, d_model]
            K: Key   [batch_size, seq_len_k, d_model]
            V: Value [batch_size, seq_len_v, d_model]
            mask: 可选的掩码

        Returns:
            output: 多头注意力的输出 [batch_size, seq_len_q, d_model]
        """
        batch_size = Q.size(0)

        # Step 1: 通过线性变换得到 Q, K, V
        # 然后重塑为 [batch_size, num_heads, seq_len, d_k] 的形状
        # 这样可以并行计算所有头的注意力

        # [B, L, d_model] -> [B, L, d_model] -> [B, L, num_heads, d_k] -> [B, num_heads, L, d_k]
        Q = self.W_Q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Step 2: 如果提供了 mask，需要扩展其维度以匹配多头结构
        if mask is not None:
            # mask: [B, 1, L, L] 广播到所有头
            mask = mask.unsqueeze(1)  # [B, 1, L, L]

        # Step 3: 计算缩放点积注意力
        # context: [B, num_heads, L_q, d_k]
        # attn_weights: [B, num_heads, L_q, L_k]
        context, attn_weights = self.attn_layer(Q, K, V, mask)

        # Step 4: 将多头的输出拼接起来
        # [B, num_heads, L, d_k] -> [B, L, num_heads, d_k] -> [B, L, d_model]
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )

        # Step 5: 通过输出线性层 W^O
        output = self.fc_out(context)

        # Step 6: 应用 dropout
        output = self.dropout(output)

        return output


# ======================================================================
# Position-wise Feed-Forward Network
# ======================================================================
class PositionwiseFeedForward(nn.Module):
    """
    逐位置前馈网络（Position-wise Feed-Forward Network）

    数学公式：
        FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
             = ReLU(xW_1 + b_1)W_2 + b_2

    特点：
        - 对序列中的每个位置独立应用相同的前馈网络
        - 包含两个线性变换和一个 ReLU 激活函数
        - 中间层维度 d_ff 通常是 d_model 的 4 倍（如 d_model=512, d_ff=2048）

    作用：
        - 增加模型的非线性表达能力
        - 对注意力输出进行进一步的特征变换
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        初始化前馈网络

        Args:
            d_model: 输入和输出的维度
            d_ff: 中间层的维度（通常为 d_model 的 4 倍）
            dropout: Dropout 概率
        """
        super(PositionwiseFeedForward, self).__init__()

        # 第一个线性层：d_model -> d_ff（维度扩展）
        self.linear1 = nn.Linear(d_model, d_ff)

        # 第二个线性层：d_ff -> d_model（维度还原）
        self.linear2 = nn.Linear(d_ff, d_model)

        # Dropout 层，用于正则化
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量 [batch_size, seq_len, d_model]

        Returns:
            output: 输出张量 [batch_size, seq_len, d_model]
        """
        # x: [B, L, d_model]
        # -> [B, L, d_ff] 经过第一个线性层和 ReLU 激活
        # -> [B, L, d_ff] 应用 dropout
        # -> [B, L, d_model] 经过第二个线性层
        output = self.linear2(self.dropout(F.relu(self.linear1(x))))

        return output


# ======================================================================
# Positional Encoding
# ======================================================================
class PositionalEncoding(nn.Module):
    """
    位置编码（Positional Encoding）

    问题背景：
        Transformer 中的注意力机制是置换不变的（permutation-invariant），
        即改变序列顺序不会影响输出。但对于序列建模，位置信息至关重要。

    解决方案：
        在输入 embedding 上加上位置编码，使模型能够利用序列的顺序信息。

    编码公式（正弦-余弦编码）：
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    其中：
        - pos: 位置索引（0 到 max_len-1）
        - i: 维度索引（0 到 d_model/2-1）
        - 偶数维度使用 sin，奇数维度使用 cos

    优点：
        - 可以外推到训练时未见过的序列长度
        - 不同位置的编码之间存在固定的线性关系
        - 不需要学习参数
    """

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        """
        初始化位置编码

        Args:
            d_model: 模型维度
            max_len: 支持的最大序列长度
            dropout: Dropout 概率
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        # 创建位置编码矩阵 [max_len, d_model]
        pe = torch.zeros(max_len, d_model)

        # 位置索引 [max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 计算分母项：10000^(2i/d_model)
        # 等价于 exp(2i * -log(10000) / d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) *
            (-math.log(10000.0) / d_model)
        )

        # 偶数维度使用 sin
        pe[:, 0::2] = torch.sin(position * div_term)

        # 奇数维度使用 cos
        pe[:, 1::2] = torch.cos(position * div_term)

        # 添加 batch 维度 [1, max_len, d_model]
        pe = pe.unsqueeze(0)

        # 将位置编码注册为 buffer（不是模型参数，但会被保存和加载）
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        将位置编码加到输入上

        Args:
            x: 输入张量 [batch_size, seq_len, d_model]

        Returns:
            output: 加上位置编码的张量 [batch_size, seq_len, d_model]
        """
        # 根据输入序列长度截取相应的位置编码
        # self.pe[:, :x.size(1)]: [1, seq_len, d_model]
        x = x + self.pe[:, :x.size(1)]

        # 应用 dropout
        return self.dropout(x)


# ======================================================================
# Encoder Layer
# ======================================================================
class EncoderLayer(nn.Module):
    """
    Transformer 编码器层

    结构（按顺序）：
        1. Multi-Head Self-Attention
        2. Add & Norm（残差连接 + Layer Normalization）
        3. Position-wise Feed-Forward Network
        4. Add & Norm（残差连接 + Layer Normalization）

    残差连接的作用：
        - 缓解深层网络的梯度消失问题
        - 允许信息直接流过多层
        - 使训练更加稳定

    Layer Normalization 的作用：
        - 对每个样本的特征维度进行归一化
        - 加速训练收敛
        - 减少对学习率的敏感度

    注意：
        - 原始 Transformer 使用 Post-LN（先子层后归一化）
        - 现代实践中 Pre-LN（先归一化后子层）更常用，训练更稳定
        - 这里实现 Post-LN 以遵循原始论文
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        初始化编码器层

        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            d_ff: 前馈网络中间层维度
            dropout: Dropout 概率
        """
        super(EncoderLayer, self).__init__()

        # Multi-Head Self-Attention
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)

        # Position-wise Feed-Forward Network
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        编码器层的前向传播

        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            mask: 可选的注意力掩码

        Returns:
            output: 编码器层输出 [batch_size, seq_len, d_model]
        """
        # Step 1: Multi-Head Self-Attention
        # 在自注意力中，Q=K=V=x
        attn_output = self.self_attn(x, x, x, mask)

        # Step 2: 残差连接 + Layer Norm
        x = self.norm1(x + self.dropout(attn_output))

        # Step 3: Position-wise Feed-Forward
        ff_output = self.feed_forward(x)

        # Step 4: 残差连接 + Layer Norm
        output = self.norm2(x + self.dropout(ff_output))

        return output


# ======================================================================
# Decoder Layer
# ======================================================================
class DecoderLayer(nn.Module):
    """
    Transformer 解码器层

    结构（按顺序）：
        1. Masked Multi-Head Self-Attention（带掩码的自注意力）
        2. Add & Norm
        3. Multi-Head Cross-Attention（编码器-解码器注意力）
        4. Add & Norm
        5. Position-wise Feed-Forward Network
        6. Add & Norm

    关键区别：
        - Self-Attention 使用因果掩码（causal mask），确保位置 i 只能关注 i 及之前的位置
        - Cross-Attention 中 Q 来自解码器，K 和 V 来自编码器输出

    用途：
        - 用于序列到序列任务（如机器翻译、文本摘要）
        - 在生成过程中，解码器逐个生成 token
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        初始化解码器层

        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            d_ff: 前馈网络中间层维度
            dropout: Dropout 概率
        """
        super(DecoderLayer, self).__init__()

        # Masked Multi-Head Self-Attention
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)

        # Multi-Head Cross-Attention
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)

        # Position-wise Feed-Forward Network
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        """
        解码器层的前向传播

        Args:
            x: 解码器输入 [batch_size, tgt_len, d_model]
            enc_output: 编码器输出 [batch_size, src_len, d_model]
            src_mask: 源序列掩码（用于 cross-attention）
            tgt_mask: 目标序列掩码（用于 self-attention，通常是因果掩码）

        Returns:
            output: 解码器层输出 [batch_size, tgt_len, d_model]
        """
        # Step 1: Masked Self-Attention（解码器自注意力）
        # Q=K=V=x，使用因果掩码防止看到未来信息
        self_attn_output = self.self_attn(x, x, x, tgt_mask)

        # Step 2: 残差连接 + Layer Norm
        x = self.norm1(x + self.dropout(self_attn_output))

        # Step 3: Cross-Attention（编码器-解码器注意力）
        # Q 来自解码器，K 和 V 来自编码器
        cross_attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)

        # Step 4: 残差连接 + Layer Norm
        x = self.norm2(x + self.dropout(cross_attn_output))

        # Step 5: Position-wise Feed-Forward
        ff_output = self.feed_forward(x)

        # Step 6: 残差连接 + Layer Norm
        output = self.norm3(x + self.dropout(ff_output))

        return output


# ======================================================================
# Full Transformer Model
# ======================================================================
class Transformer(nn.Module):
    """
    完整的 Transformer 模型

    支持两种模式：
        1. Encoder-only（仅编码器）：用于语言建模、分类等任务
        2. Encoder-Decoder（编码器-解码器）：用于序列到序列任务

    模型结构：
        Input Embedding
        + Positional Encoding
        → Encoder Layers (N×)
        [→ Decoder Layers (N×)]  # 可选
        → Linear (vocabulary projection)
        → Softmax（在损失函数中）
    """

    def __init__(
        self,
        vocab_size,
        d_model=512,
        num_heads=8,
        d_ff=2048,
        num_encoder_layers=6,
        num_decoder_layers=6,
        max_seq_len=5000,
        dropout=0.1,
        use_decoder=False
    ):
        """
        初始化 Transformer 模型

        Args:
            vocab_size: 词汇表大小
            d_model: 模型维度（Embedding 维度）
            num_heads: 多头注意力的头数
            d_ff: 前馈网络中间层维度
            num_encoder_layers: 编码器层数
            num_decoder_layers: 解码器层数（仅在 use_decoder=True 时使用）
            max_seq_len: 支持的最大序列长度
            dropout: Dropout 概率
            use_decoder: 是否使用解码器（True: Encoder-Decoder, False: Encoder-only）
        """
        super(Transformer, self).__init__()

        self.d_model = d_model
        self.use_decoder = use_decoder

        # Token Embedding 层
        # 将 token id 映射到 d_model 维的向量
        self.embedding = nn.Embedding(vocab_size, d_model)

        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # 编码器层堆叠
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])

        # 解码器层堆叠（可选）
        if use_decoder:
            self.decoder_layers = nn.ModuleList([
                DecoderLayer(d_model, num_heads, d_ff, dropout)
                for _ in range(num_decoder_layers)
            ])

        # 输出投影层：d_model -> vocab_size
        self.fc_out = nn.Linear(d_model, vocab_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # 初始化参数
        self._init_parameters()

    def _init_parameters(self):
        """
        初始化模型参数

        使用 Xavier uniform 初始化，这是 tensor2tensor 中的标准做法
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_square_subsequent_mask(self, size):
        """
        生成因果掩码（causal mask）

        用于解码器的自注意力，确保位置 i 只能看到 i 及之前的位置。

        Args:
            size: 序列长度

        Returns:
            mask: 下三角掩码矩阵 [size, size]
                  1 表示可以关注，0 表示不能关注

        示例（size=4）：
            [[1, 0, 0, 0],
             [1, 1, 0, 0],
             [1, 1, 1, 0],
             [1, 1, 1, 1]]
        """
        mask = torch.triu(torch.ones(size, size), diagonal=1).type(torch.uint8)
        return mask == 0

    def forward(self, src, tgt=None, src_mask=None, tgt_mask=None):
        """
        Transformer 前向传播

        Args:
            src: 源序列（编码器输入） [batch_size, src_len]
            tgt: 目标序列（解码器输入） [batch_size, tgt_len]，仅在 use_decoder=True 时需要
            src_mask: 源序列掩码
            tgt_mask: 目标序列掩码（通常是因果掩码）

        Returns:
            output: 模型输出 [batch_size, seq_len, vocab_size]
        """
        # ========== 编码器部分 ==========
        # Step 1: Embedding + Positional Encoding
        # src: [B, src_len] -> [B, src_len, d_model]
        src_embedded = self.embedding(src) * math.sqrt(self.d_model)  # 缩放 embedding
        src_embedded = self.pos_encoding(src_embedded)

        # Step 2: 通过所有编码器层
        enc_output = src_embedded
        for layer in self.encoder_layers:
            enc_output = layer(enc_output, src_mask)

        # ========== 解码器部分（可选）==========
        if self.use_decoder and tgt is not None:
            # Step 3: Target Embedding + Positional Encoding
            # tgt: [B, tgt_len] -> [B, tgt_len, d_model]
            tgt_embedded = self.embedding(tgt) * math.sqrt(self.d_model)
            tgt_embedded = self.pos_encoding(tgt_embedded)

            # Step 4: 通过所有解码器层
            dec_output = tgt_embedded
            for layer in self.decoder_layers:
                dec_output = layer(dec_output, enc_output, src_mask, tgt_mask)

            # Step 5: 投影到词汇表
            output = self.fc_out(dec_output)

        # ========== 仅编码器模式 ==========
        else:
            # 直接将编码器输出投影到词汇表（用于语言建模）
            output = self.fc_out(enc_output)

        return output


# ======================================================================
# 辅助函数：创建预定义配置的模型
# ======================================================================

def make_transformer_lm(vocab_size, d_model=128, num_heads=4, d_ff=512,
                        num_layers=2, dropout=0.1):
    """
    创建 Encoder-only 语言模型（用于 Tiny Shakespeare 等任务）

    Args:
        vocab_size: 词汇表大小
        d_model: 模型维度
        num_heads: 注意力头数
        d_ff: 前馈网络维度
        num_layers: 编码器层数
        dropout: Dropout 概率

    Returns:
        model: Transformer 语言模型
    """
    return Transformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_encoder_layers=num_layers,
        num_decoder_layers=0,
        dropout=dropout,
        use_decoder=False
    )


def make_transformer_seq2seq(vocab_size, d_model=256, num_heads=8, d_ff=1024,
                              num_encoder_layers=4, num_decoder_layers=4,
                              dropout=0.1):
    """
    创建 Encoder-Decoder 序列到序列模型

    Args:
        vocab_size: 词汇表大小
        d_model: 模型维度
        num_heads: 注意力头数
        d_ff: 前馈网络维度
        num_encoder_layers: 编码器层数
        num_decoder_layers: 解码器层数
        dropout: Dropout 概率

    Returns:
        model: Transformer Seq2Seq 模型
    """
    return Transformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dropout=dropout,
        use_decoder=True
    )


if __name__ == "__main__":
    """
    测试代码：验证模型的前向传播
    """
    # 测试参数
    vocab_size = 100
    batch_size = 2
    src_len = 10
    tgt_len = 8

    # 创建随机输入
    src = torch.randint(0, vocab_size, (batch_size, src_len))
    tgt = torch.randint(0, vocab_size, (batch_size, tgt_len))

    print("=" * 60)
    print("测试 Encoder-only 模型")
    print("=" * 60)
    model_lm = make_transformer_lm(vocab_size)
    output_lm = model_lm(src)
    print(f"输入形状: {src.shape}")
    print(f"输出形状: {output_lm.shape}")
    print(f"参数量: {sum(p.numel() for p in model_lm.parameters()):,}")

    print("\n" + "=" * 60)
    print("测试 Encoder-Decoder 模型")
    print("=" * 60)
    model_seq2seq = make_transformer_seq2seq(vocab_size)
    tgt_mask = model_seq2seq.generate_square_subsequent_mask(tgt_len).unsqueeze(0).unsqueeze(0)
    output_seq2seq = model_seq2seq(src, tgt, tgt_mask=tgt_mask)
    print(f"源序列形状: {src.shape}")
    print(f"目标序列形状: {tgt.shape}")
    print(f"输出形状: {output_seq2seq.shape}")
    print(f"参数量: {sum(p.numel() for p in model_seq2seq.parameters()):,}")
    print("\n✓ 所有测试通过！")
