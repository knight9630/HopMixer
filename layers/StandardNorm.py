import torch
import torch.nn as nn


class Normalize(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=False, subtract_last=False, non_norm=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(Normalize, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.non_norm = non_norm
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.non_norm:
            return x
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.non_norm:
            return x
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


class DishTS(nn.Module):
    """
    Dish-TS: A General Paradigm for Alleviating Distribution Shift in Time Series Forecasting (AAAI 2023)
    用来替换普通的 Normalize (RevIN)。
    核心区别：均值(mean)和方差(scale)不再是计算出来的，而是通过网络学习出来的。
    """
    def __init__(self, num_features: int, seq_len: int, eps=1e-5, subtract_last=False, non_norm=False):
        """
        :param num_features: 通道数/特征数 (enc_in)
        :param seq_len: 输入序列长度 (seq_len)
        :param eps: 数值稳定性项
        """
        super(DishTS, self).__init__()
        self.num_features = num_features
        self.seq_len = seq_len
        self.eps = eps
        self.subtract_last = subtract_last
        self.non_norm = non_norm
        
        # Dish-TS 的核心：系数学习网络 (Coefficient Learning Network)
        # 输入: [Batch, Channel, Seq_Len] -> 输出: [Batch, Channel, 2] (2代表 mean 和 scale)
        # 这种设计利用了 Linear 对最后一维操作的特性，实现了 Channel Independence
        self.reduce_layer = nn.Sequential(
            nn.Linear(seq_len, 16),
            nn.Tanh(), # 论文中使用 Tanh 或 GELU 激活
            nn.Linear(16, 2)
        )
        
        # 初始化：为了让训练初期稳定，初始化为近似 Identity 映射
        # 即让网络初始输出的 mean 接近 0，scale 接近 1 (log(scale)接近0)
        # 这里不做复杂初始化也可以，PyTorch 默认初始化通常也能收敛
        
    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _get_statistics(self, x):
        """
        核心修改点：
        原版 RevIN: 直接计算 x.mean(), x.var()
        Dish-TS: 将 x 输入神经网络，预测出 mean 和 scale
        """
        if self.non_norm:
            return
            
        # x shape: [Batch, Seq_Len, Channels] -> [Batch, Channels, Seq_Len]
        x_trans = x.permute(0, 2, 1)
        
        # 经过网络学习统计量
        # output shape: [Batch, Channels, 2]
        stats = self.reduce_layer(x_trans)
        
        # 拆分 mean 和 scale
        # [Batch, Channels, 1] -> 转回 [Batch, 1, Channels] 以匹配原始 x 的广播维度
        self.mean = stats[:, :, 0:1].permute(0, 2, 1) 
        self.scale = stats[:, :, 1:2].permute(0, 2, 1) 
        
        # 使用 Softplus 或 Exp 保证 scale 是正数，避免除零
        self.scale = torch.nn.functional.softplus(self.scale) + self.eps

    def _normalize(self, x):
        if self.non_norm:
            return x
        # 公式: (x - learned_mean) / learned_scale
        x = (x - self.mean) / self.scale
        return x

    def _denormalize(self, x):
        if self.non_norm:
            return x
        # 公式: x * learned_scale + learned_mean
        x = x * self.scale + self.mean
        return x