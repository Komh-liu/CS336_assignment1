import torch
import torch.nn as nn
import math
from einops import rearrange

"""
全连接层
    1. init: 定义一个W矩阵
    2. forward: 如输入x进行矩阵乘法
"""

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        
        # 1. 定义权重 W (形状: out x in)
        # 注意要把 device 和 dtype 传进去，确保张量创建在正确位置
        factory_kwargs = {'device': device, 'dtype': dtype}
        
        # nn.Parameter 的作用是告诉 PyTorch：“这个张量是模型的一部分，它是需要通过训练来学习的权重（Weights）。”
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        
        # 2. 初始化权重 (截断正态分布)  Xavier 初始化
        # 根据 3.4.1: sigma^2 = 2 / (din + dout)
        
        '''
        Xavier 初始化源于 Glorot & Bengio (2010) 的经典论文。其推导基于以下假设：
        权重 W、输入 x、梯度均独立同分布,且期望为 0。
        线性层输出 y = Wx,则 Var(y) = din · Var(W) · Var(x)。
        为了在深层网络中保持信号尺度稳定，我们希望：
        前向传播:Var(y) ≈ Var(x) ⟹ Var(W) ≈ 1 / din
        反向传播：梯度回传时同理要求 Var(W) ≈ 1 / dout
        由于无法同时严格满足两个条件，论文提出取两者的折中值（调和平均形式）：
        '''
        std = (2.0 / (in_features + out_features)) ** 0.5
        # PDF 要求截断在 [-3sigma, 3sigma]
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 使用 einsum 处理，适应各种 Batch 维度情况
        # '...i' 表示输入 x 的最后一个维度 (in_features)
        # 'oi' 表示权重 W (out_features, in_features)
        # '-> ...o' 表示输出保留前面的维度，最后一个维度变成 out_features
        return torch.einsum('...i, oi -> ...o', x, self.weight)



class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), **factory_kwargs))
        std = 1.0
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:

        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        # 1. 必须初始化为全 1 (ones)
        self.weight = nn.Parameter(torch.ones(d_model, **factory_kwargs))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (batch_size, sequence_length, d_model)

        in_dtype = x.dtype
        # 2. 转换为 float32 以防平方计算时溢出
        x_float = x.to(torch.float32)

        # 3. 计算均方根 (Root Mean Square)
        # 公式: rms = sqrt( mean(x^2) + eps )
        # dim=-1 表示在隐藏层维度计算，keepdim=True 方便后续除法自动广播
        """
        在 PyTorch（以及 NumPy）中，广播（Broadcasting） 是指在对两个形状不同的张量进行算术运算时，系统自动“扩展”较小张量的维度，使其与较大张量匹配的机制。
        要使两个张量是可广播的（Broadcastable），必须满足以下核心规则：
        核心规则：从右往左看
            比较两个张量的形状时，要从最后一个维度（最右边）开始往前检查。对于每一对对应的维度，必须满足以下 两个条件之一：
                1.这两个维度的值相等。
                2.其中一个维度的值是 1。
            如果其中一个张量的维度较少，系统会自动在它的左侧补 1，直到两者的维度数量相等，然后再按上述规则检查。
        """
        ms = x_float.pow(2).mean(dim=-1, keepdim=True)
        rms = torch.sqrt(ms + self.eps)

        # 4. 归一化并乘以可学习的增益参数 g
        result = (x_float / rms) * self.weight

        # 5. 转回原始类型
        return result.to(in_dtype)
        

def silu_fn(in_features):

    return in_features * torch.sigmoid(in_features)



class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_ff = d_ff
        self.d_model = d_model
        # W1 和 W3 是并行升维层: d_model -> d_ff
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)
        # W2 是降维层: d_ff -> d_model
        self.w2 = Linear(d_ff, d_model, device, dtype)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        gate = silu_fn(self.w1(x))
        signal = self.w3(x)
        # 形状: [..., d_ff]
        return self.w2(gate * signal)


    

def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        # 1. 为了数值稳定性，减去指定维度上的最大值
        # dim=-1 通常是 Transformer 中的隐藏层或词表维度
        x_max = torch.max(x, dim=dim, keepdim=True).values
        x_stable = x - x_max
        
        # 2. 计算指数
        exp_x = torch.exp(x_stable)
        
        # 3. 计算分母的各指数之和
        sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)
        
        # 4. 计算最终结果
        return exp_x / sum_exp

def scaled_dot_product_attention(
    Q: torch.Tensor, 
    K: torch.Tensor, 
    V: torch.Tensor, 
    mask: torch.Tensor = None
) -> torch.Tensor:
    """
    Q: (batch_size, ..., n, d_k)
    K: (batch_size, ..., m, d_k)
    V: (batch_size, ..., m, d_v)
    mask: (..., n, m) 或者是可以广播到该形状的布尔张量 (True 表示关注, False 表示屏蔽)
    """
    d_k = Q.size(-1)
    
    # 1. 计算分数: Q @ K^T / sqrt(d_k)
    # 交换最后两个维度进行矩阵乘法
    # 形状变化: (..., n, d_k) @ (..., d_k, m) -> (..., n, m)
    scores = torch.einsum('...nk, ...mk-> ...nm', Q, K) / math.sqrt(d_k)
    
    # 2. 应用掩码
    if mask is not None:
        # PDF 要求: 把 mask 为 False 的地方填入 -inf
        # 注意: 使用一个足够小的负数，通常 float('-inf') 在 torch 中是安全的
        scores = scores.masked_fill(mask == False, float('-inf'))
    
    # 3. Softmax 归一化 (在最后一个维度 m 上)
    # 注意: 这里的 dim=-1 指向的是 Key 序列的长度维度
    probs = softmax(scores, dim=-1)
    
    # 4. 对 Value 加权求和
    # 形状变化: (..., n, m) @ (..., m, d_v) -> (..., n, d_v)
    output = torch.einsum('...nm, ...mk-> ...nk', probs, V)
    
    return output

