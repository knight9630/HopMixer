import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F

class FrequencyResidualCalibrator(nn.Module):
    def __init__(self, seq_in_len, seq_out_len, enc_in, hidden_dim=64):
        """
        Args:
            seq_in_len: 输入序列长度
            seq_out_len: 预测序列长度
            enc_in: 变量维度 (Channels)
            hidden_dim: 频域映射的隐藏层维度
        """
        super().__init__()
        self.seq_in_len = seq_in_len
        self.seq_out_len = seq_out_len
        self.channels = enc_in
        
        # 1. 频域特征提取层 (处理 rfft 后的实部和虚部)
        # rfft 输出长度为 floor(seq_len/2) + 1
        self.freq_len_in = seq_in_len // 2 + 1
        self.freq_len_out = seq_out_len // 2 + 1
        
        # 输入是实部+虚部，所以维度 * 2
        self.freq_projector = nn.Sequential(
            nn.Linear(self.freq_len_in * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.freq_len_out * 2)
        )

    def input_processing(self, x_hist):
        # --- 步骤 1: 对历史数据做 FFT ---
        # 维度变换: [B, L, C] -> [B, C, L] 方便处理
        x_fft = torch.fft.rfft(x_hist.permute(0, 2, 1), dim=-1)
        
        # 提取实部和虚部并在最后一个维度拼接: [B, C, Freq_In] -> [B, C, Freq_In * 2]
        x_fft_feat = torch.cat([x_fft.real, x_fft.imag], dim=-1)
        
        # --- 步骤 2: 频域映射 (学习残差的频率分布) ---
        # [B, C, Freq_In * 2] -> [B, C, Freq_Out * 2]
        residual_freq = self.freq_projector(x_fft_feat)
        
        # 分离实部和虚部
        res_real, res_imag = torch.split(residual_freq, self.freq_len_out, dim=-1)
        # 重组为复数
        res_complex = torch.complex(res_real, res_imag)
        
        # --- 步骤 3: 逆 FFT 变回时域 ---
        # irfft 输出长度需要指定，否则可能有一位误差
        res_time = torch.fft.irfft(res_complex, n=self.seq_out_len, dim=-1)
        
        # 维度变回: [B, C, L] -> [B, L, C]
        res_time = res_time.permute(0, 2, 1)

        return res_time

    def forward(self, x_hist, y_pred_base):
        """
        x_hist: 历史输入 [Batch, Seq_In, Channel]
        y_pred_base: Base Model 的预测输出 [Batch, Seq_Out, Channel]
        """
        
        res_time = self.input_processing(x_hist)
        # --- 步骤 4: 叠加校准 ---
        return y_pred_base + res_time
    

class TrendAwareSinglePatchAugmenter(nn.Module):
    def __init__(self, patch_len, d_model, dropout=0.1):
        super().__init__()
        self.patch_len = patch_len
        
        # 1. 权重共享的投影层 (替代原本的 M*M 个 Linear)
        # 我们把每个 Patch 视为一个 token
        self.query = nn.Linear(patch_len, d_model)
        self.key = nn.Linear(patch_len, d_model)
        self.value = nn.Linear(patch_len, patch_len) # 保持输出长度一致用于残差
        
        self.dropout = nn.Dropout(dropout)
        # 一个可学习的门控系数，初始化很小，防止一开始破坏原始数据
        self.gate = nn.Parameter(torch.zeros(1)) 

    def forward(self, x):
        """
        x: [Batch, T, C]
        """
        B, T, C = x.shape
        stride = self.patch_len # 假设不重叠，或者你可以自己设 stride
        
        # --- 1. Patching (向量化) ---
        # 变成 [B, C, N, P] -> [B*C, N, P]
        # N 是 Patch 数量, P 是 Patch 长度
        x_patches = x.unfold(dimension=1, size=self.patch_len, step=stride)
        # print('x_patches shape:', x_patches.shape)
        N = x_patches.shape[1]
        x_patches = x_patches.permute(0, 2, 1, 3).reshape(B * C, N, self.patch_len)
        
        # --- 2. 计算趋势一致性 Mask (全矩阵操作) ---
        # 计算一阶差分符号: [B*C, N, P-1]
        diff = x_patches[:, :, 1:] - x_patches[:, :, :-1]
        sign = torch.sign(diff) # {-1, 0, 1}
        
        # 计算两两 Patch 的趋势相似度
        # 如果 sign 相同，乘积为 1，否则为 -1 或 0
        # [B*C, N, P-1] @ [B*C, P-1, N] -> [B*C, N, N]
        trend_sim = torch.matmul(sign, sign.transpose(-1, -2))
        
        # 生成 Mask: 只有当大部分时间点趋势一致时，才允许聚合
        # 归一化到 [0, 1]，趋近 1 表示趋势高度一致
        # trend_score: [B*C, N, N]
        trend_score = trend_sim / (self.patch_len - 1 + 1e-6)
        
        # 硬阈值过滤（可选，保留你的 idea）或者用 Softmax
        # 这里建议用 ReLU + Softmax 强调正相关
        mask = F.relu(trend_score) # 过滤掉负相关的（反向趋势）

        # --- 3. 语义聚类 (Attention 机制) ---
        # Q, K, V 变换
        Q = self.query(x_patches) # [B*C, N, D]
        K = self.key(x_patches)   # [B*C, N, D]
        V = self.value(x_patches) # [B*C, N, P] (注意这里 Value 映射回原长度或保持原样)
        
        # 计算内容相似度 Attention Score
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (Q.shape[-1] ** 0.5) # [B*C, N, N]
        
        # --- 4. 融合: 内容相似度 * 趋势一致性 ---
        # 这是核心：只有内容像，且趋势也像的，权重才大
        combined_scores = attn_scores * mask
        
        attn_weights = F.softmax(combined_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 聚合
        out_patches = torch.matmul(attn_weights, V) # [B*C, N, P]
        
        # --- 5. 还原与残差连接 ---
        # Reshape 回 [B, C, N, P]
        out_patches = out_patches.reshape(B, C, N, self.patch_len)
        
        # Fold (这一步取决于你的 stride，这里假设拼接回去)
        # 简单拼接回去，假设 stride == patch_len
        x_out = out_patches.permute(0, 2, 1, 3).reshape(B, N * self.patch_len, C)
        
        # 处理可能的长度不一致 (padding)
        if x_out.shape[1] < T:
            x_out = F.pad(x_out, (0, 0, 0, T - x_out.shape[1]))
        elif x_out.shape[1] > T:
            x_out = x_out[:, :T, :]
            
        # 关键：使用可学习的 Gate 做残差
        # 这样模型初始时可以选择忽视插件 (gate=0)，慢慢学习加入特征
        return x + self.gate * x_out    
    
class TrendAwareCrossPatchAugmenter(nn.Module):
    def __init__(self, patch_len, d_model, dropout=0.1):
        super().__init__()
        self.patch_len = patch_len
        
        # 1. 投影层
        # 输入维度是 patch_len，映射到 d_model 进行 Attention
        self.query = nn.Linear(patch_len, d_model)
        self.key = nn.Linear(patch_len, d_model)
        self.value = nn.Linear(patch_len, patch_len) # Value 保持原长，用于重构
        
        self.dropout = nn.Dropout(dropout)
        
        # 2. 可学习的门控系数
        # 初始化为 0，确保训练初期不干扰 Base Model
        self.gate = nn.Parameter(torch.zeros(1)) 

    def forward(self, x):
        """
        x: [Batch, T, C]
        """
        B, T, C = x.shape
        stride = self.patch_len 
        
        # --- 1. Patching (切片) ---
        # x.unfold(dimension=1, ...) 
        # 输出形状: [Batch, N, Channel, PatchLen]
        # N 是每个变量切出的 Patch 数量
        x_patches = x.unfold(dimension=1, size=self.patch_len, step=stride)
        
        N_per_channel = x_patches.shape[1]
        
        # --- 2. 维度重组 (Cross-Variable Flattening) ---
        # 目标：把所有 Channel 的 Patch 混在一起
        # 路径: [B, N, C, P] -> [B, C, N, P] -> [B, C * N, P]
        # 现在，第二维 M = C * N 代表了该样本中所有的 Patch (跨变量)
        x_patches = x_patches.permute(0, 2, 1, 3).reshape(B, C * N_per_channel, self.patch_len)
        
        # M: 总 Patch 数 (Cross-Variable)
        M = x_patches.shape[1] 
        
        # --- 3. 计算趋势一致性 Mask (全矩阵操作) ---
        # 输入: [B, M, P]
        # 差分: [B, M, P-1]
        diff = x_patches[:, :, 1:] - x_patches[:, :, :-1]
        sign = torch.sign(diff) # {-1, 0, 1}
        
        # 计算两两 Patch 的趋势相似度 (跨变量)
        # [B, M, P-1] @ [B, P-1, M] -> [B, M, M]
        trend_sim = torch.matmul(sign, sign.transpose(-1, -2))
        
        # 归一化分数，并过滤
        trend_score = trend_sim / (self.patch_len - 1 + 1e-6)
        mask = F.relu(trend_score) # [B, M, M]

        # --- 4. 语义聚类 (Attention 机制) ---
        # Q, K: [B, M, D]
        Q = self.query(x_patches) 
        K = self.key(x_patches)   
        # V: [B, M, P]
        V = self.value(x_patches) 
        
        # Attention Score: [B, M, M]
        # 这里 M = C * N，因此矩阵大小是 (C*N)^2，允许跨变量交互
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (Q.shape[-1] ** 0.5)
        
        # --- 5. 融合: 内容相似度 * 趋势一致性 ---
        combined_scores = attn_scores * mask
        
        attn_weights = F.softmax(combined_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 聚合: [B, M, M] @ [B, M, P] -> [B, M, P]
        out_patches = torch.matmul(attn_weights, V)
        
        # --- 6. 还原与重构 (Reconstruction) ---
        # 现在我们需要把混合在一起的 M 个 Patch 拆回 C 个通道
        # 输入: [B, C*N, P] -> [B, C, N, P]
        out_patches = out_patches.reshape(B, C, N_per_channel, self.patch_len)
        
        # 调整顺序以恢复时间轴: [B, C, N, P] -> [B, N, P, C]
        # 注意：permute 必须先把 Channel 移到最后，把 N 和 P 连在一起
        out_patches = out_patches.permute(0, 2, 3, 1) 
        
        # 合并 N 和 P 得到时间轴 T
        x_out = out_patches.reshape(B, N_per_channel * self.patch_len, C)
        
        # --- 7. Padding 处理 (如果 T 不是 patch_len 的整数倍) ---
        if x_out.shape[1] < T:
            x_out = F.pad(x_out, (0, 0, 0, T - x_out.shape[1]))
        elif x_out.shape[1] > T:
            x_out = x_out[:, :T, :]
            
        # --- 8. 残差连接 ---
        return x + self.gate * x_out    
    

class TrendAwarePearsonAugmenter_CI(nn.Module):
    def __init__(self, patch_len, d_model, dropout=0.1, temperature=10.0):
        super().__init__()
        self.patch_len = patch_len
        self.temperature = temperature 
        self.query = nn.Linear(patch_len, d_model)
        self.key = nn.Linear(patch_len, d_model)
        self.value = nn.Linear(patch_len, patch_len) 
        
        self.dropout = nn.Dropout(dropout)
        
        self.gate = nn.Parameter(torch.zeros(1)) 

    def compute_pearson_attention(self, q, k):
        """
        计算 batch 中所有向量两两之间的皮尔逊相关系数
        输入: q, k 形状为 [Batch_Size, N, D]
        输出: corr_matrix 形状为 [Batch_Size, N, N]
        """
        # 1. 去均值 (Centering)
        # mean: [Batch_Size, N, 1]
        q_mean = q.mean(dim=-1, keepdim=True)
        k_mean = k.mean(dim=-1, keepdim=True)
        
        q_centered = q - q_mean
        k_centered = k - k_mean
        
        # 2. 分子：协方差
        # [Batch_Size, N, D] @ [Batch_Size, D, N] -> [Batch_Size, N, N]
        numerator = torch.matmul(q_centered, k_centered.transpose(-2, -1))
        
        # 3. 分母：标准差之积
        # norm: [Batch_Size, N, 1]
        q_norm = torch.sqrt(torch.sum(q_centered ** 2, dim=-1, keepdim=True) + 1e-8)
        k_norm = torch.sqrt(torch.sum(k_centered ** 2, dim=-1, keepdim=True) + 1e-8)
        
        # [Batch_Size, N, 1] @ [Batch_Size, 1, N] -> [Batch_Size, N, N]
        denominator = torch.matmul(q_norm, k_norm.transpose(-2, -1))
        
        # 4. 计算相关系数 [-1, 1]
        corr_matrix = numerator / (denominator + 1e-8)
        
        return corr_matrix

    def forward(self, x):
        """
        x: [Batch, T, C]
        """
        B, T, C = x.shape
        stride = self.patch_len 
        
        # --- 1. Patching (切片) ---
        # x.unfold 输出形状: [B, N, C, P]
        x_patches = x.unfold(dimension=1, size=self.patch_len, step=stride)
        N = x_patches.shape[1]
        
        # --- 2. 维度重组 (Channel Independence) ---
        # 关键区别：我们要把 Batch 和 Channel 合并，让它们并行处理
        # [B, N, C, P] -> [B, C, N, P] -> [B*C, N, P]
        x_patches = x_patches.permute(0, 2, 1, 3).reshape(B * C, N, self.patch_len)
        
        # 此时 Batch_Size 变成了 B*C，N 是 Patch 数量
        
        # --- 3. 趋势 Mask (Trend Consistency) ---
        # diff: [B*C, N, P-1]
        diff = x_patches[:, :, 1:] - x_patches[:, :, :-1]
        # print('diff shape:', diff.shape)
        sign = torch.sign(diff) # {-1, 0, 1}
        # print('sign shape:', sign.shape)
        
        # Trend Sim: [B*C, N, N]
        trend_sim = torch.matmul(sign, sign.transpose(-1, -2))
        # print('trend_sim shape:', trend_sim.shape)
        trend_score = trend_sim / (self.patch_len - 1 + 1e-6)
        mask = F.relu(trend_score) # 只保留正相关
        # print(mask)

        # --- 4. 皮尔逊语义聚类 (Pearson Attention) ---
        # Q, K: [B*C, N, D]
        Q = self.query(x_patches) 
        K = self.key(x_patches)   
        # V: [B*C, N, P]
        V = self.value(x_patches) 
        
        # 计算 PCC: [B*C, N, N]
        pearson_scores = self.compute_pearson_attention(Q, K)
        # print('pearson_scores shape:', pearson_scores.shape) 
        
        # --- 5. 融合与加权 ---
        # 同样使用 temperature 放大差异
        combined_scores = pearson_scores * mask * self.temperature
        # combined_scores = pearson_scores * self.temperature
        
        attn_weights = F.softmax(combined_scores, dim=-1)

        attn_map_out = attn_weights.reshape(B, C, N, N)
        
        attn_weights = self.dropout(attn_weights)
        
        # 聚合: [B*C, N, N] @ [B*C, N, P] -> [B*C, N, P]
        out_patches = torch.matmul(attn_weights, V)
        
        # --- 6. 还原与重构 (Reconstruction) ---
        # 先拆分 B 和 C: [B*C, N, P] -> [B, C, N, P]
        out_patches = out_patches.reshape(B, C, N, self.patch_len)
        
        # 调整顺序以恢复时间轴: [B, C, N, P] -> [B, N, P, C]
        # (PatchTST 等模型常用的维度顺序)
        out_patches = out_patches.permute(0, 2, 3, 1) 
        
        # 合并 N 和 P 得到时间轴 T
        x_out = out_patches.reshape(B, N * self.patch_len, C)
        
        # --- 7. Padding 处理 ---
        if x_out.shape[1] < T:
            x_out = F.pad(x_out, (0, 0, 0, T - x_out.shape[1]))
        elif x_out.shape[1] > T:
            x_out = x_out[:, :T, :]
            
        # --- 8. 残差连接 ---
        return x + self.gate * x_out
    
class TrendAwarePearsonAugmenter_CD(nn.Module):
    def __init__(self, patch_len, d_model, dropout=0.1, temperature=10.0):
        super().__init__()
        self.patch_len = patch_len
        self.temperature = temperature # 用于控制 Softmax 的尖锐程度
        
        # 1. 投影层
        self.query = nn.Linear(patch_len, d_model)
        self.key = nn.Linear(patch_len, d_model)
        self.value = nn.Linear(patch_len, patch_len) # Value 保持原长
        
        self.dropout = nn.Dropout(dropout)
        
        # 2. 门控系数
        self.gate = nn.Parameter(torch.zeros(1)) 

    def compute_pearson_attention(self, q, k):
        """
        计算 batch 中所有向量两两之间的皮尔逊相关系数
        输入:
            q: [B, M, D]
            k: [B, M, D]
        输出:
            corr_matrix: [B, M, M] 范围在 [-1, 1]
        """
        # 1. 去均值 (Centering)
        # mean: [B, M, 1]
        q_mean = q.mean(dim=-1, keepdim=True)
        k_mean = k.mean(dim=-1, keepdim=True)
        
        q_centered = q - q_mean
        k_centered = k - k_mean
        
        # 2. 分子：协方差 (Covariance-like)
        # [B, M, D] @ [B, D, M] -> [B, M, M]
        numerator = torch.matmul(q_centered, k_centered.transpose(-2, -1))
        
        # 3. 分母：标准差之积 (Std dev product)
        # q_norm: [B, M, 1] (L2范数)
        q_norm = torch.sqrt(torch.sum(q_centered ** 2, dim=-1, keepdim=True) + 1e-8)
        k_norm = torch.sqrt(torch.sum(k_centered ** 2, dim=-1, keepdim=True) + 1e-8)
        
        # [B, M, 1] @ [B, 1, M] -> [B, M, M]
        denominator = torch.matmul(q_norm, k_norm.transpose(-2, -1))
        
        # 4. 计算相关系数
        corr_matrix = numerator / (denominator + 1e-8)
        
        return corr_matrix

    def forward(self, x):
        """
        x: [Batch, T, C]
        """
        B, T, C = x.shape
        stride = self.patch_len 
        
        # --- 1. Patching & Flattening ---
        x_patches = x.unfold(dimension=1, size=self.patch_len, step=stride)
        N_per_channel = x_patches.shape[1]
        
        # 跨变量混合 [B, C*N, P]
        x_patches = x_patches.permute(0, 2, 1, 3).reshape(B, C * N_per_channel, self.patch_len)
        M = x_patches.shape[1] 
        
        # --- 2. 趋势 Mask (Trend Consistency) ---
        diff = x_patches[:, :, 1:] - x_patches[:, :, :-1]
        sign = torch.sign(diff)
        trend_sim = torch.matmul(sign, sign.transpose(-1, -2))
        trend_score = trend_sim / (self.patch_len - 1 + 1e-6)
        mask = F.relu(trend_score) # [B, M, M]

        # --- 3. 皮尔逊语义聚类 (Pearson Attention) ---
        Q = self.query(x_patches) # [B, M, D]
        K = self.key(x_patches)   # [B, M, D]
        V = self.value(x_patches) # [B, M, P]
        
        # 【核心修改】这里不再用 Q@K.T，而是计算 PCC
        pearson_scores = self.compute_pearson_attention(Q, K) # [B, M, M], Range [-1, 1]
        
        # --- 4. 融合与加权 ---
        # 技巧：PCC 的值域是 [-1, 1]，直接做 Softmax 会过于平滑（Entropy高）。
        # 所以我们需要乘以一个 temperature (比如 10)，放大差异，让 Softmax 能够选出最相关的 Patch。
        
        combined_scores = pearson_scores * mask * self.temperature
        # combined_scores = pearson_scores * self.temperature
        
        attn_weights = F.softmax(combined_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 聚合
        out_patches = torch.matmul(attn_weights, V)
        
        # --- 5. 还原与重构 ---
        out_patches = out_patches.reshape(B, C, N_per_channel, self.patch_len)
        out_patches = out_patches.permute(0, 2, 3, 1) 
        x_out = out_patches.reshape(B, N_per_channel * self.patch_len, C)
        
        if x_out.shape[1] < T:
            x_out = F.pad(x_out, (0, 0, 0, T - x_out.shape[1]))
        elif x_out.shape[1] > T:
            x_out = x_out[:, :T, :]
            
        return x + self.gate * x_out
    
class PatchPearson_PointTrend_Augmenter_CI(nn.Module):
    def __init__(self, patch_len, d_model, dropout=0.1, temperature=10.0):
        super().__init__()
        self.patch_len = patch_len
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)
        
        # 1. 投影层 (用于计算 Patch 间的语义相似度)
        self.query = nn.Linear(patch_len, d_model)
        self.key = nn.Linear(patch_len, d_model)
        
        # Value 保持原长，因为我们要对原始数据点进行 Mask 和融合
        self.value = nn.Linear(patch_len, patch_len)
        
        # 门控
        self.gate = nn.Parameter(torch.zeros(1)) 

    def compute_pearson_matrix(self, q, k):
        """
        计算 Patch 级别的皮尔逊相关系数
        输入: q, k [B, N, D]
        输出: scores [B, N, N]
        """
        q_mean = q.mean(dim=-1, keepdim=True)
        k_mean = k.mean(dim=-1, keepdim=True)
        
        q_centered = q - q_mean
        k_centered = k - k_mean
        
        numerator = torch.matmul(q_centered, k_centered.transpose(-2, -1))
        
        q_norm = torch.sqrt(torch.sum(q_centered ** 2, dim=-1, keepdim=True) + 1e-8)
        k_norm = torch.sqrt(torch.sum(k_centered ** 2, dim=-1, keepdim=True) + 1e-8)
        
        denominator = torch.matmul(q_norm, k_norm.transpose(-2, -1))
        
        return numerator / (denominator + 1e-8)

    def get_pointwise_trend_mask(self, x_patches):
        """
        生成细粒度的点级 Mask
        输入: x_patches [B, N, P]
        输出: mask [B, N, N, P] (4D Tensor)
        含义: mask[b, i, j, t] 表示: 第i个patch的第t个点，与第j个patch的第t个点，趋势是否相同
        """
        B, N, P = x_patches.shape
        
        # 1. 计算 Patch 内每个点的瞬时趋势 (Padding第一个点以保持长度为P)
        # x_patches: [B, N, P]
        padded = F.pad(x_patches, (1, 0), mode='replicate') # [B, N, P+1]
        diff = padded[:, :, 1:] - padded[:, :, :-1]         # [B, N, P]
        sign = torch.sign(diff)                             # [B, N, P] 值为 -1, 0, 1
        
        # 2. 扩展维度以进行广播比较
        # 我们需要比较任意两个 Patch (i, j) 在同一个位置 t 的趋势
        
        # sign_i: [B, N, 1, P] (作为目标 Patch)
        sign_i = sign.unsqueeze(2) 
        
        # sign_j: [B, 1, N, P] (作为源 Patch)
        sign_j = sign.unsqueeze(1)
        
        # 3. 生成 Mask
        # 只有当 sign_i * sign_j > 0 时 (即同正或同负)，趋势才相同
        # 这里的广播机制会自动生成 [B, N, N, P] 的形状
        trend_mask = (sign_i * sign_j > 0).float()
        
        # 可选：如果希望保留自己 (对角线)，可以取消下面这行的注释
        # identity = torch.eye(N, device=x_patches.device).reshape(1, N, N, 1)
        # trend_mask = torch.maximum(trend_mask, identity)
        
        return trend_mask

    def forward(self, x):
        """
        x: [Batch, T, C]
        """
        B_orig, T, C = x.shape
        stride = self.patch_len
        
        # --- 1. Patching ---
        # 建议混合 Batch 和 Channel，这样每个变量独立计算趋势，互不干扰
        # [B, T, C] -> [B, C, T] -> [B*C, T]
        x_in = x.permute(0, 2, 1).reshape(B_orig * C, T)
        
        # Unfold: [B*C, N, P]
        x_patches = x_in.unfold(dimension=1, size=self.patch_len, step=stride)
        B_new, N, P = x_patches.shape
        
        # --- 2. Patch-wise Pearson Attention (宏观权重) ---
        Q = self.query(x_patches) # [B*C, N, D]
        K = self.key(x_patches)   # [B*C, N, D]
        V = self.value(x_patches) # [B*C, N, P]
        
        # 计算 Patch 间的相似度矩阵 [B*C, N, N]
        patch_scores = self.compute_pearson_matrix(Q, K)
        
        # Softmax 归一化 (基于 Patch 整体相似度)
        patch_attn = F.softmax(patch_scores * self.temperature, dim=-1) # [B*C, N, N]
        patch_attn = self.dropout(patch_attn)
        
        # --- 3. Point-wise Trend Masking (微观筛选) ---
        # 计算 [B*C, N, N, P] 的 Mask
        # 含义: 即使 Patch A 和 B 很像，如果在第 t 个点趋势相反，mask[..., t] 也会是 0
        point_mask = self.get_pointwise_trend_mask(x_patches) 
        
        # --- 4. 融合 (Hybrid Fusion) ---
        # 我们要计算: Out[i, t] = Sum_j ( Attn[i, j] * Mask[i, j, t] * V[j, t] )
        
        # 扩展 patch_attn 维度以匹配 mask: [B*C, N, N, 1]
        patch_attn_expanded = patch_attn.unsqueeze(-1)
        
        # 最终权重: 只有当 (Patch相似) 且 (点趋势相同) 时，权重才保留，否则为 0
        final_weights = patch_attn_expanded * point_mask # [B*C, N, N, P]
        
        # 聚合: Einstein Summation (爱因斯坦求和约定)
        # b: batch, n: target patch, k: source patch, p: point position
        # weights[b, n, k, p] * V[b, k, p] -> sum over k -> output[b, n, p]
        out_patches = torch.einsum('bnkp, bkp -> bnp', final_weights, V)
        
        # --- 5. 还原 ---
        # [B*C, N, P] -> [B*C, T]
        x_out = out_patches.reshape(B_new, N * P)
        
        # 处理可能的长度不一致 (padding/trimming)
        if x_out.shape[1] < T:
            x_out = F.pad(x_out, (0, T - x_out.shape[1]))
        elif x_out.shape[1] > T:
            x_out = x_out[:, :T]
            
        # 还原回 [B, T, C]
        x_out = x_out.reshape(B_orig, C, T).permute(0, 2, 1)
        
        return x + self.gate * x_out

class PatchPearson_PointTrend_Augmenter_CD(nn.Module):
    def __init__(self, patch_len, d_model, dropout=0.1, temperature=10.0):
        super().__init__()
        self.patch_len = patch_len
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)
        
        # 1. 投影层
        self.query = nn.Linear(patch_len, d_model)
        self.key = nn.Linear(patch_len, d_model)
        self.value = nn.Linear(patch_len, patch_len)
        
        # 门控系数
        self.gate = nn.Parameter(torch.zeros(1)) 

    def compute_pearson_matrix(self, q, k):
        """
        计算 Patch 间的皮尔逊相关系数
        注意：这里的维度 M = Channel * N_patches
        输入: [B, M, D]
        输出: [B, M, M]
        """
        q_mean = q.mean(dim=-1, keepdim=True)
        k_mean = k.mean(dim=-1, keepdim=True)
        
        q_centered = q - q_mean
        k_centered = k - k_mean
        
        # 分子：协方差
        numerator = torch.matmul(q_centered, k_centered.transpose(-2, -1))
        
        # 分母：标准差乘积
        q_norm = torch.sqrt(torch.sum(q_centered ** 2, dim=-1, keepdim=True) + 1e-8)
        k_norm = torch.sqrt(torch.sum(k_centered ** 2, dim=-1, keepdim=True) + 1e-8)
        denominator = torch.matmul(q_norm, k_norm.transpose(-2, -1))
        
        return numerator / (denominator + 1e-8)

    def get_pointwise_trend_mask(self, x_patches):
        """
        生成跨通道的逐点趋势 Mask
        输入: x_patches [B, M, P]  (M 是所有通道Patch的总和)
        输出: mask [B, M, M, P]
        """
        # 1. 计算一阶差分
        padded = F.pad(x_patches, (1, 0), mode='replicate') 
        diff = padded[:, :, 1:] - padded[:, :, :-1] # [B, M, P]
        sign = torch.sign(diff) # -1, 0, 1
        
        # 2. 广播比较 (全通道混合比较)
        # Target: [B, M, 1, P]
        sign_i = sign.unsqueeze(2) 
        # Source: [B, 1, M, P]
        sign_j = sign.unsqueeze(1) 
        
        # 3. 生成 Mask
        # 只要趋势同向，无论来自哪个通道，Mask都为1
        trend_mask = (sign_i * sign_j > 0).float()
        
        return trend_mask

    def forward(self, x):
        """
        x: [Batch, T, C]
        """
        B, T, C = x.shape
        stride = self.patch_len
        
        # --- 1. Patching & Flattening (关键步骤) ---
        # 变换为 [B, C, T]
        x_in = x.permute(0, 2, 1) 
        
        # Unfold: [B, C, N, P]
        # N 是每个通道切分出的 Patch 数量
        x_patches_orig = x_in.unfold(dimension=2, size=self.patch_len, step=stride)
        N = x_patches_orig.shape[2]
        
        # 【关键修改】混合 Channel 和 Patch 维度
        # 我们把所有通道的 Patch 展平到同一个维度 M
        # [B, C, N, P] -> [B, C*N, P]
        x_patches = x_patches_orig.reshape(B, C * N, self.patch_len)
        M = x_patches.shape[1] # M = C * N
        
        # --- 2. 跨通道 Pearson Attention ---
        Q = self.query(x_patches) # [B, M, D]
        K = self.key(x_patches)   # [B, M, D]
        V = self.value(x_patches) # [B, M, P]
        
        # 计算所有 Patch (不分通道) 两两之间的 PCC
        pearson_scores = self.compute_pearson_matrix(Q, K) # [B, M, M]
        
        # Softmax 归一化
        patch_attn = F.softmax(pearson_scores * self.temperature, dim=-1)
        patch_attn = self.dropout(patch_attn)
        
        # --- 3. 跨通道 Point-wise Trend Mask ---
        # Mask: [B, M, M, P]
        point_mask = self.get_pointwise_trend_mask(x_patches)
        
        # --- 4. 融合 ---
        # 扩展维度 [B, M, M, 1]
        patch_attn_expanded = patch_attn.unsqueeze(-1)
        
        # 结合权重
        final_weights = patch_attn_expanded * point_mask # [B, M, M, P]
        
        # 聚合: Sum over k (Source Patches)
        # b: Batch
        # n: Target Patch (in C*N range)
        # k: Source Patch (in C*N range) - 可以来自不同通道
        # p: Point Position
        out_patches_flat = torch.einsum('bnkp, bkp -> bnp', final_weights, V)
        
        # --- 5. 还原结构 ---
        # [B, C*N, P] -> [B, C, N, P]
        out_patches = out_patches_flat.reshape(B, C, N, self.patch_len)
        
        # [B, C, N, P] -> [B, C, N*P]
        # 注意：这里需要先把 N 和 P 拼起来
        x_out = out_patches.reshape(B, C, N * self.patch_len)
        
        # 处理长度 (Padding/Trimming)
        if x_out.shape[2] < T:
            x_out = F.pad(x_out, (0, T - x_out.shape[2]))
        elif x_out.shape[2] > T:
            x_out = x_out[:, :, :T]
            
        # [B, C, T] -> [B, T, C]
        x_out = x_out.permute(0, 2, 1)
        
        return x + self.gate * x_out

class PatchPearson_PointTrend_Augmenter_CI_TopK(nn.Module):
    def __init__(self, patch_len, d_model, top_k=16, dropout=0.1, temperature=10.0):
        super().__init__()
        self.patch_len = patch_len
        self.temperature = temperature
        self.top_k = top_k
        self.dropout = nn.Dropout(dropout)
        
        # 1. 投影层
        self.query = nn.Linear(patch_len, d_model)
        self.key = nn.Linear(patch_len, d_model)
        # Value 保持原长
        self.value = nn.Linear(patch_len, patch_len)
        
        self.gate = nn.Parameter(torch.zeros(1)) 

    def compute_pearson_matrix(self, q, k):
        q_mean = q.mean(dim=-1, keepdim=True)
        k_mean = k.mean(dim=-1, keepdim=True)
        q_centered = q - q_mean
        k_centered = k - k_mean
        numerator = torch.matmul(q_centered, k_centered.transpose(-2, -1))
        q_norm = torch.sqrt(torch.sum(q_centered ** 2, dim=-1, keepdim=True) + 1e-8)
        k_norm = torch.sqrt(torch.sum(k_centered ** 2, dim=-1, keepdim=True) + 1e-8)
        denominator = torch.matmul(q_norm, k_norm.transpose(-2, -1))
        return numerator / (denominator + 1e-8)

    def get_pointwise_trend_mask(self, x_patches):
        B, N, P = x_patches.shape
        padded = F.pad(x_patches, (1, 0), mode='replicate')
        diff = padded[:, :, 1:] - padded[:, :, :-1]
        sign = torch.sign(diff)
        sign_i = sign.unsqueeze(2) 
        sign_j = sign.unsqueeze(1) 
        
        # 趋势相同为1，不同为0
        trend_mask = (sign_i * sign_j > 0).float()
        
        # 【重要】强制保留对角线 (Self-Attention)
        # 因为在Top-k之后，如果某点的所有邻居趋势都不对，全被屏蔽，Softmax会出NaN。
        # 必须保证自己永远能看到自己。
        eye = torch.eye(N, device=x_patches.device).reshape(1, N, N, 1).expand(B, N, N, P)
        trend_mask = torch.maximum(trend_mask, eye)
        
        return trend_mask

    def forward(self, x):
        B_orig, T, C = x.shape
        stride = self.patch_len
        
        # --- 1. Patching (通道独立) ---
        # [B, T, C] -> [B*C, T]
        x_in = x.permute(0, 2, 1).reshape(B_orig * C, T)
        
        x_patches = x_in.unfold(dimension=1, size=self.patch_len, step=stride)
        B_new, N, P = x_patches.shape 
        
        # --- 2. 计算 Pearson ---
        Q = self.query(x_patches) 
        K = self.key(x_patches)   
        V = self.value(x_patches) 
        
        pearson_scores = self.compute_pearson_matrix(Q, K) # [B*C, N, N]
        
        # --- 3. Top-k 策略 (Patch 级别筛选) ---
        curr_k = min(self.top_k, N)
        topk_vals, topk_inds = torch.topk(pearson_scores, k=curr_k, dim=-1)
        
        # 初始化 Masked Scores 为 -inf
        scores_topk = torch.full_like(pearson_scores, -1e9)
        
        # 填回 Top-k 的分数
        scores_topk.scatter_(dim=-1, index=topk_inds, src=topk_vals)
        
        # --- 4. 趋势屏蔽 (Point 级别筛选) ---
        # 这一步去掉了值修正，改回了 Masking 逻辑
        
        # 计算 Mask [B*C, N, N, P]
        point_mask = self.get_pointwise_trend_mask(x_patches)
        
        # 扩展分数维度: [B*C, N, N] -> [B*C, N, N, P]
        scores_expanded = scores_topk.unsqueeze(-1).expand(B_new, N, N, P)
        
        # 应用趋势 Mask：
        # 如果 trend_mask 为 0 (趋势冲突)，将分数强行设为 -inf
        # 这样在 Softmax 时，该点的权重会变成 0
        final_scores = torch.where(point_mask.bool(), scores_expanded * self.temperature, torch.tensor(-1e9, device=x.device))
        
        # --- 5. Softmax & 聚合 ---
        # dim=-2 (Source Patch 维度)
        # 此时，只有 (Top-k 且 趋势一致) 的点才有非零权重
        attn_weights = F.softmax(final_scores, dim=-2) 
        attn_weights = self.dropout(attn_weights)
        
        # 直接聚合原始 V (没有 V_rectified 了)
        out_patches = torch.einsum('bnkp, bkp -> bnp', attn_weights, V)
        
        # --- 6. 还原 ---
        x_out = out_patches.reshape(B_new, N * P)
        if x_out.shape[1] < T:
            x_out = F.pad(x_out, (0, T - x_out.shape[1]))
        elif x_out.shape[1] > T:
            x_out = x_out[:, :T]
            
        x_out = x_out.reshape(B_orig, C, T).permute(0, 2, 1)
        
        return x + self.gate * x_out

class PatchPearson_PointTrend_Augmenter_CD_TopK(nn.Module):
    def __init__(self, patch_len, d_model, top_k=16, dropout=0.1, temperature=10.0):
        super().__init__()
        self.patch_len = patch_len
        self.temperature = temperature
        self.top_k = top_k  # 新增 Top-k 参数
        self.dropout = nn.Dropout(dropout)
        
        # 1. 投影层
        self.query = nn.Linear(patch_len, d_model)
        self.key = nn.Linear(patch_len, d_model)
        self.value = nn.Linear(patch_len, patch_len)
        
        # 门控系数
        self.gate = nn.Parameter(torch.zeros(1)) 

    def compute_pearson_matrix(self, q, k):
        # 输入: [B, M, D], M = C * N
        q_mean = q.mean(dim=-1, keepdim=True)
        k_mean = k.mean(dim=-1, keepdim=True)
        q_centered = q - q_mean
        k_centered = k - k_mean
        
        numerator = torch.matmul(q_centered, k_centered.transpose(-2, -1))
        
        q_norm = torch.sqrt(torch.sum(q_centered ** 2, dim=-1, keepdim=True) + 1e-8)
        k_norm = torch.sqrt(torch.sum(k_centered ** 2, dim=-1, keepdim=True) + 1e-8)
        denominator = torch.matmul(q_norm, k_norm.transpose(-2, -1))
        
        return numerator / (denominator + 1e-8)

    def get_pointwise_trend_mask(self, x_patches):
        # 输入: [B, M, P]
        # 输出: [B, M, M, P]
        B, M, P = x_patches.shape
        padded = F.pad(x_patches, (1, 0), mode='replicate') 
        diff = padded[:, :, 1:] - padded[:, :, :-1]
        sign = torch.sign(diff)
        
        sign_i = sign.unsqueeze(2) # Target
        sign_j = sign.unsqueeze(1) # Source
        
        # 趋势相同为1，不同为0
        trend_mask = (sign_i * sign_j > 0).float()
        
        # 【重要保护】强制保留对角线 (自己看自己)
        # 防止 Top-k 选出的邻居在某一点全军覆没导致 Softmax NaN
        eye = torch.eye(M, device=x_patches.device).reshape(1, M, M, 1).expand(B, M, M, P)
        trend_mask = torch.maximum(trend_mask, eye)
        
        return trend_mask

    def forward(self, x):
        """
        x: [Batch, T, C]
        """
        B, T, C = x.shape
        stride = self.patch_len
        
        # --- 1. Patching (通道依赖混合) ---
        x_in = x.permute(0, 2, 1) # [B, C, T]
        x_patches_orig = x_in.unfold(dimension=2, size=self.patch_len, step=stride)
        N = x_patches_orig.shape[2]
        
        # [B, C, N, P] -> [B, C*N, P]
        # 混合 C 和 N，实现跨通道交互
        x_patches = x_patches_orig.reshape(B, C * N, self.patch_len)
        _, M, P = x_patches.shape # M = C * N
        
        # --- 2. 计算 Pearson 系数 ---
        Q = self.query(x_patches) # [B, M, D]
        K = self.key(x_patches)   # [B, M, D]
        V = self.value(x_patches) # [B, M, P]
        
        pearson_scores = self.compute_pearson_matrix(Q, K) # [B, M, M]
        
        # --- 3. Top-k 策略 (Patch 级粗筛) ---
        # 只保留跨通道最相关的 k 个 Patch
        curr_k = min(self.top_k, M)
        topk_vals, topk_inds = torch.topk(pearson_scores, k=curr_k, dim=-1)
        
        # 初始化全 -inf 矩阵
        scores_sparse = torch.full_like(pearson_scores, -1e9)
        
        # 填回 Top-k 值
        scores_sparse.scatter_(dim=-1, index=topk_inds, src=topk_vals)
        
        # --- 4. 趋势屏蔽 (Point 级精筛) ---
        
        # 计算跨通道趋势 Mask [B, M, M, P]
        point_mask = self.get_pointwise_trend_mask(x_patches)
        
        # 扩展分数维度: [B, M, M] -> [B, M, M, P]
        scores_expanded = scores_sparse.unsqueeze(-1).expand(B, M, M, P)
        
        # 双重 Mask 应用：
        # 1. 如果不是 Top-k，已经是 -inf
        # 2. 如果是 Top-k 但趋势 Mask 为 0，也被强制设为 -inf
        final_scores = torch.where(point_mask.bool(), scores_expanded * self.temperature, torch.tensor(-1e9, device=x.device))
        
        # --- 5. Softmax & 聚合 ---
        # dim=-2 (Source 维度)
        # Softmax 会自动忽略 -inf 的项，重新归一化剩余项
        attn_weights = F.softmax(final_scores, dim=-2) 
        attn_weights = self.dropout(attn_weights)
        
        # 聚合: [B, M, M, P] * [B, M, P] -> [B, M, P]
        # 此时 attn_weights 是高度稀疏的 (只有 Top-k 且同趋势的点非零)
        out_patches_flat = torch.einsum('bnkp, bkp -> bnp', attn_weights, V)
        
        # --- 6. 还原结构 ---
        # [B, C*N, P] -> [B, C, N, P]
        out_patches = out_patches_flat.reshape(B, C, N, self.patch_len)
        
        # [B, C, N*P]
        x_out = out_patches.reshape(B, C, N * self.patch_len)
        
        # Padding 处理
        if x_out.shape[2] < T:
            x_out = F.pad(x_out, (0, T - x_out.shape[2]))
        elif x_out.shape[2] > T:
            x_out = x_out[:, :, :T]
            
        # [B, C, T] -> [B, T, C]
        x_out = x_out.permute(0, 2, 1)
        
        return x + self.gate * x_out
    
class SeasonRectifiedAugmenter_CI(nn.Module):
    def __init__(self, patch_len, d_model, dropout=0.1, temperature=10.0):
        super().__init__()
        self.patch_len = patch_len
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)
        
        self.query = nn.Linear(patch_len, d_model)
        self.key = nn.Linear(patch_len, d_model)
        self.value = nn.Linear(patch_len, patch_len)
        self.gate = nn.Parameter(torch.zeros(1)) 

    def compute_pearson_matrix(self, q, k):
        # ... (标准 Pearson 计算，保持不变) ...
        q_mean = q.mean(dim=-1, keepdim=True)
        k_mean = k.mean(dim=-1, keepdim=True)
        q_centered = q - q_mean
        k_centered = k - k_mean
        numerator = torch.matmul(q_centered, k_centered.transpose(-2, -1))
        q_norm = torch.sqrt(torch.sum(q_centered ** 2, dim=-1, keepdim=True) + 1e-8)
        k_norm = torch.sqrt(torch.sum(k_centered ** 2, dim=-1, keepdim=True) + 1e-8)
        denominator = torch.matmul(q_norm, k_norm.transpose(-2, -1))
        return numerator / (denominator + 1e-8)

    def get_pointwise_trend_mask(self, x_patches):
        # ... (标准 Trend Mask 计算，保持不变) ...
        # 注意：这里不需要再强制对角线为1了，因为我们不会让分母变0
        B, N, P = x_patches.shape
        padded = F.pad(x_patches, (1, 0), mode='replicate')
        diff = padded[:, :, 1:] - padded[:, :, :-1]
        sign = torch.sign(diff)
        sign_i = sign.unsqueeze(2) 
        sign_j = sign.unsqueeze(1) 
        trend_mask = (sign_i * sign_j > 0).float()
        return trend_mask

    def forward(self, x):
        B_orig, T, C = x.shape
        stride = self.patch_len
        
        # 1. 通道独立 Reshape
        x_in = x.permute(0, 2, 1).reshape(B_orig * C, T)
        
        # Patching
        x_patches = x_in.unfold(dimension=1, size=self.patch_len, step=stride)
        B_new, N, P = x_patches.shape # B_new = Batch * Channel
        
        # 2. 计算 Pearson 权重 (Global Semantic)
        Q = self.query(x_patches)
        K = self.key(x_patches)
        V = self.value(x_patches) # [B, N, P] (Source Values)
        
        pearson_scores = self.compute_pearson_matrix(Q, K) # [B, N, N]
        
        # 直接计算 Attention Weights (不做 Mask 屏蔽！)
        # 我们认为只要形状像，就有参考价值
        attn_weights = F.softmax(pearson_scores * self.temperature, dim=-1)
        attn_weights = self.dropout(attn_weights) # [B, N, N]
        
        # 3. 计算 Trend Mask
        # [B, N, N, P]
        point_mask = self.get_pointwise_trend_mask(x_patches)
        
        # --- 4. 【核心优化】 值修正 (Value Rectification) ---
        
        # 目标：构建一个修正后的 Value 矩阵 V_rectified
        # 维度: [B, N, N, P] 
        # 含义: 当 Target=n, Source=k 时，使用的 Values
        
        # V_source: [B, 1, N, P] (来自 k 的值)
        V_source = V.unsqueeze(1) 
        
        # V_target: [B, N, 1, P] (目标 n 自己的值)
        # 这是当 Source 趋势不对时，我们要用的“替补”
        V_target = V.unsqueeze(2)
        
        # 构造修正后的值：
        # 如果 Mask==1 (趋势一致) -> 用 V_source (引入增强信息)
        # 如果 Mask==0 (趋势冲突) -> 用 V_target (保持自我，避免噪声)
        V_rectified = torch.where(point_mask.bool(), V_source, V_target)
        
        # --- 5. 聚合 ---
        
        # 扩展权重维度以匹配 V_rectified: [B, N, N, 1]
        attn_weights_expanded = attn_weights.unsqueeze(-1)
        
        # 加权求和
        # weights[b, n, k, 1] * V_rectified[b, n, k, p] -> sum over k -> out[b, n, p]
        out_patches = torch.sum(attn_weights_expanded * V_rectified, dim=2)
        
        # 6. 还原
        x_out = out_patches.reshape(B_new, N * P)
        if x_out.shape[1] < T:
            x_out = F.pad(x_out, (0, T - x_out.shape[1]))
        elif x_out.shape[1] > T:
            x_out = x_out[:, :T]
            
        x_out = x_out.reshape(B_orig, C, T).permute(0, 2, 1)
        
        return x + self.gate * x_out
    
class SeasonRectifiedAugmenter_CD(nn.Module):
    def __init__(self, patch_len, d_model, dropout=0.1, temperature=10.0):
        super().__init__()
        self.patch_len = patch_len
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)
        
        # 投影层
        self.query = nn.Linear(patch_len, d_model)
        self.key = nn.Linear(patch_len, d_model)
        self.value = nn.Linear(patch_len, patch_len)
        self.gate = nn.Parameter(torch.zeros(1)) 

    def compute_pearson_matrix(self, q, k):
        # 这里的输入维度是 [B, M, D]，其中 M = C * N
        q_mean = q.mean(dim=-1, keepdim=True)
        k_mean = k.mean(dim=-1, keepdim=True)
        q_centered = q - q_mean
        k_centered = k - k_mean
        numerator = torch.matmul(q_centered, k_centered.transpose(-2, -1))
        q_norm = torch.sqrt(torch.sum(q_centered ** 2, dim=-1, keepdim=True) + 1e-8)
        k_norm = torch.sqrt(torch.sum(k_centered ** 2, dim=-1, keepdim=True) + 1e-8)
        denominator = torch.matmul(q_norm, k_norm.transpose(-2, -1))
        return numerator / (denominator + 1e-8)

    def get_pointwise_trend_mask(self, x_patches):
        # x_patches: [B, M, P]
        # 这里的 M 包含了所有通道的 Patch
        # 所以 mask 的维度是 [B, M, M, P]，实现了跨通道的趋势比较
        B, M, P = x_patches.shape
        padded = F.pad(x_patches, (1, 0), mode='replicate')
        diff = padded[:, :, 1:] - padded[:, :, :-1]
        sign = torch.sign(diff)
        sign_i = sign.unsqueeze(2) 
        sign_j = sign.unsqueeze(1) 
        trend_mask = (sign_i * sign_j > 0).float()
        return trend_mask

    def forward(self, x):
        """
        输入: x [Batch, T, C]
        """
        B, T, C = x.shape
        stride = self.patch_len
        
        # --- 1. Patching (关键改动：混合 C 和 N) ---
        
        # 变换为 [B, C, T]
        x_in = x.permute(0, 2, 1)
        
        # Unfold 得到 [B, C, N, P]
        # N 是单通道切出的 Patch 数量
        x_patches_orig = x_in.unfold(dimension=2, size=self.patch_len, step=stride)
        N = x_patches_orig.shape[2]
        
        # 【关键】将 Channel 维度和 N 维度合并
        # [B, C, N, P] -> [B, C*N, P]
        # 现在，维度 1 (M) 包含了该样本所有变量的所有 Patch
        x_patches = x_patches_orig.reshape(B, C * N, self.patch_len)
        M = x_patches.shape[1] # M = C * N
        
        # --- 2. 计算 Pearson (Global Cross-Channel Semantic) ---
        Q = self.query(x_patches) # [B, M, D]
        K = self.key(x_patches)   # [B, M, D]
        V = self.value(x_patches) # [B, M, P]
        
        # 计算所有 Patch (跨通道) 之间的相似度
        pearson_scores = self.compute_pearson_matrix(Q, K) # [B, M, M]
        
        # 计算权重 (不做 Mask 屏蔽)
        attn_weights = F.softmax(pearson_scores * self.temperature, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # --- 3. 计算 Cross-Channel Trend Mask ---
        # [B, M, M, P]
        point_mask = self.get_pointwise_trend_mask(x_patches)
        
        # --- 4. 值修正 (Value Rectification) ---
        
        # V_source: [B, 1, M, P] (别人)
        V_source = V.unsqueeze(1)
        
        # V_target: [B, M, 1, P] (自己)
        V_target = V.unsqueeze(2)
        
        # 逻辑：
        # 如果 Channel A 的 Patch 和 Channel B 的 Patch 趋势一致 -> 用 Channel B 的值
        # 如果 趋势冲突 -> 保持 Channel A 的原值
        V_rectified = torch.where(point_mask.bool(), V_source, V_target)
        
        # --- 5. 聚合 ---
        
        # [B, M, M, 1]
        attn_weights_expanded = attn_weights.unsqueeze(-1)
        
        # Sum over k (Source dimension)
        out_patches_flat = torch.sum(attn_weights_expanded * V_rectified, dim=2) # [B, M, P]
        
        # --- 6. 还原结构 ---
        
        # 先把 M 拆回 C 和 N: [B, C*N, P] -> [B, C, N, P]
        out_patches = out_patches_flat.reshape(B, C, N, self.patch_len)
        
        # 拼合 N 和 P: [B, C, N*P]
        x_out = out_patches.reshape(B, C, N * self.patch_len)
        
        # 处理 Padding
        if x_out.shape[2] < T:
            x_out = F.pad(x_out, (0, T - x_out.shape[2]))
        elif x_out.shape[2] > T:
            x_out = x_out[:, :, :T]
            
        # [B, C, T] -> [B, T, C]
        x_out = x_out.permute(0, 2, 1)
        
        return x + self.gate * x_out

class SeasonRectifiedAugmenter_CI_TopK(nn.Module):
    def __init__(self, patch_len, d_model, top_k=16, dropout=0.1, temperature=10.0):
        """
        top_k: 设定保留相关性最高的 k 个 Patch
        """
        super().__init__()
        self.patch_len = patch_len
        self.temperature = temperature
        self.top_k = top_k  # 新增
        self.dropout = nn.Dropout(dropout)
        
        # 1. 投影层
        self.query = nn.Linear(patch_len, d_model)
        self.key = nn.Linear(patch_len, d_model)
        self.value = nn.Linear(patch_len, patch_len)
        self.gate = nn.Parameter(torch.zeros(1)) 

    def compute_pearson_matrix(self, q, k):
        # [B, N, D] -> [B, N, N]
        q_mean = q.mean(dim=-1, keepdim=True)
        k_mean = k.mean(dim=-1, keepdim=True)
        q_centered = q - q_mean
        k_centered = k - k_mean
        
        numerator = torch.matmul(q_centered, k_centered.transpose(-2, -1))
        
        q_norm = torch.sqrt(torch.sum(q_centered ** 2, dim=-1, keepdim=True) + 1e-8)
        k_norm = torch.sqrt(torch.sum(k_centered ** 2, dim=-1, keepdim=True) + 1e-8)
        
        denominator = torch.matmul(q_norm, k_norm.transpose(-2, -1))
        return numerator / (denominator + 1e-8)

    def get_pointwise_trend_mask(self, x_patches):
        # [B, N, P] -> [B, N, N, P]
        B, N, P = x_patches.shape
        padded = F.pad(x_patches, (1, 0), mode='replicate')
        diff = padded[:, :, 1:] - padded[:, :, :-1]
        sign = torch.sign(diff)
        sign_i = sign.unsqueeze(2) 
        sign_j = sign.unsqueeze(1) 
        trend_mask = (sign_i * sign_j > 0).float()
        return trend_mask

    def forward(self, x):
        """
        x: [Batch, T, C]
        """
        B_orig, T, C = x.shape
        stride = self.patch_len
        
        # --- 1. Patching (通道独立) ---
        # [B, T, C] -> [B, C, T] -> [B*C, T]
        # 合并 Batch 和 Channel，实现独立处理
        x_in = x.permute(0, 2, 1).reshape(B_orig * C, T)
        
        x_patches = x_in.unfold(dimension=1, size=self.patch_len, step=stride)
        B_new, N, P = x_patches.shape # B_new = B * C
        
        # --- 2. 计算 Pearson 系数 ---
        Q = self.query(x_patches) # [B*C, N, D]
        K = self.key(x_patches)   # [B*C, N, D]
        V = self.value(x_patches) # [B*C, N, P]
        
        pearson_scores = self.compute_pearson_matrix(Q, K) # [B*C, N, N]
        
        # --- 3. 【Top-k 策略】 ---
        # 目的：稀疏化注意力矩阵，只关注最相关的 k 个 Patch
        curr_k = min(self.top_k, N)
        
        # 1. 找到前 k 大的值和索引
        topk_vals, topk_inds = torch.topk(pearson_scores, k=curr_k, dim=-1)
        
        # 2. 构造全 -inf 矩阵
        mask_scores = torch.full_like(pearson_scores, -1e9)
        
        # 3. 填回 Top-k 值
        mask_scores.scatter_(dim=-1, index=topk_inds, src=topk_vals)
        
        # 4. Softmax (只有 Top-k 位置有非零权重)
        patch_attn = F.softmax(mask_scores * self.temperature, dim=-1)
        patch_attn = self.dropout(patch_attn)
        
        # --- 4. 趋势 Mask 和 值修正 ---
        point_mask = self.get_pointwise_trend_mask(x_patches) # [B*C, N, N, P]
        
        # 【为什么这里要用值修正而不用原来的 Masking？】
        # 因为用了 Top-k 后，候选人变少了。如果 Mask 把剩下的候选人权重也置 0，
        # 会导致该点的总权重远小于 1，造成严重的噪声。
        # 所以我们采用“趋势不对就用 Target 值替补”的策略。
        
        V_source = V.unsqueeze(1) # [B*C, 1, N, P]
        V_target = V.unsqueeze(2) # [B*C, N, 1, P]
        
        # 趋势一致 -> 用 Source (增强)
        # 趋势冲突 -> 用 Target (保底)
        V_rectified = torch.where(point_mask.bool(), V_source, V_target)
        
        # --- 5. 融合 ---
        # [B*C, N, N, 1]
        patch_attn_expanded = patch_attn.unsqueeze(-1)
        
        # 加权求和
        # 此时 patch_attn 已经是 Top-k 稀疏的了，所以只聚合了 Top-k 个邻居
        out_patches = torch.sum(patch_attn_expanded * V_rectified, dim=2) # [B*C, N, P]
        
        # --- 6. 还原 ---
        x_out = out_patches.reshape(B_new, N * P)
        
        if x_out.shape[1] < T:
            x_out = F.pad(x_out, (0, T - x_out.shape[1]))
        elif x_out.shape[1] > T:
            x_out = x_out[:, :T]
            
        # 还原回 [B, T, C]
        x_out = x_out.reshape(B_orig, C, T).permute(0, 2, 1)
        
        return x + self.gate * x_out

class SeasonRectifiedAugmenter_CD_TopK(nn.Module):
    def __init__(self, patch_len, d_model, top_k=16, dropout=0.1, temperature=10.0):
        """
        top_k: 每个 Patch 只保留相关性最高的 k 个 Patch 参与融合
        """
        super().__init__()
        self.patch_len = patch_len
        self.temperature = temperature
        self.top_k = top_k  # 新增 Top-k 参数
        self.dropout = nn.Dropout(dropout)
        
        # 投影层
        self.query = nn.Linear(patch_len, d_model)
        self.key = nn.Linear(patch_len, d_model)
        self.value = nn.Linear(patch_len, patch_len)
        self.gate = nn.Parameter(torch.zeros(1)) 

    def compute_pearson_matrix(self, q, k):
        # 输入维度: [B, M, D], M = C * N
        q_mean = q.mean(dim=-1, keepdim=True)
        k_mean = k.mean(dim=-1, keepdim=True)
        q_centered = q - q_mean
        k_centered = k - k_mean
        
        numerator = torch.matmul(q_centered, k_centered.transpose(-2, -1))
        
        q_norm = torch.sqrt(torch.sum(q_centered ** 2, dim=-1, keepdim=True) + 1e-8)
        k_norm = torch.sqrt(torch.sum(k_centered ** 2, dim=-1, keepdim=True) + 1e-8)
        
        denominator = torch.matmul(q_norm, k_norm.transpose(-2, -1))
        return numerator / (denominator + 1e-8)

    def get_pointwise_trend_mask(self, x_patches):
        # [B, M, P] -> [B, M, M, P]
        B, M, P = x_patches.shape
        padded = F.pad(x_patches, (1, 0), mode='replicate')
        diff = padded[:, :, 1:] - padded[:, :, :-1]
        sign = torch.sign(diff)
        sign_i = sign.unsqueeze(2) 
        sign_j = sign.unsqueeze(1) 
        trend_mask = (sign_i * sign_j > 0).float()
        return trend_mask

    def forward(self, x):
        """
        输入: x [Batch, T, C]
        """
        B, T, C = x.shape
        stride = self.patch_len
        
        # --- 1. Patching (通道依赖混合) ---
        x_in = x.permute(0, 2, 1) # [B, C, T]
        x_patches_orig = x_in.unfold(dimension=2, size=self.patch_len, step=stride)
        N = x_patches_orig.shape[2]
        
        # [B, C, N, P] -> [B, C*N, P]
        x_patches = x_patches_orig.reshape(B, C * N, self.patch_len)
        M = x_patches.shape[1] # M = C * N
        
        # --- 2. 计算 Pearson 系数 ---
        Q = self.query(x_patches) # [B, M, D]
        K = self.key(x_patches)   # [B, M, D]
        V = self.value(x_patches) # [B, M, P]
        
        pearson_scores = self.compute_pearson_matrix(Q, K) # [B, M, M]
        
        # --- 3. 【核心新增】 Top-k 策略 ---
        # 目的：只保留最大的 k 个相关性，其余置为 -inf
        
        # 确定实际的 k (防止 k > 总Patch数 M)
        curr_k = min(self.top_k, M)
        
        # 找出每行最大的 k 个值及其索引
        # topk_vals: [B, M, k], topk_inds: [B, M, k]
        topk_vals, topk_inds = torch.topk(pearson_scores, k=curr_k, dim=-1)
        
        # 初始化一个全 -inf 的矩阵
        mask_scores = torch.full_like(pearson_scores, -1e9)
        
        # 使用 scatter 将 topk 的值填回对应位置
        # dim=-1 表示沿着最后一个维度填充
        mask_scores.scatter_(dim=-1, index=topk_inds, src=topk_vals)
        
        # 计算注意力权重 (此时只有 Top-k 的位置有值，其余为0)
        attn_weights = F.softmax(mask_scores * self.temperature, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # --- 4. 计算 Cross-Channel Trend Mask ---
        point_mask = self.get_pointwise_trend_mask(x_patches) # [B, M, M, P]
        
        # --- 5. 值修正 (Value Rectification) ---
        V_source = V.unsqueeze(1) # [B, 1, M, P]
        V_target = V.unsqueeze(2) # [B, M, 1, P]
        
        # 趋势一致用 Source，冲突用 Target
        V_rectified = torch.where(point_mask.bool(), V_source, V_target)
        
        # --- 6. 聚合 ---
        # [B, M, M, 1]
        attn_weights_expanded = attn_weights.unsqueeze(-1)
        
        # 注意：虽然 V_rectified 是满的 [M, M]，但 attn_weights 是 Top-k 稀疏的
        # 所以只有 Top-k 的 patch 真正参与了运算，其他位置权重为0
        out_patches_flat = torch.sum(attn_weights_expanded * V_rectified, dim=2) # [B, M, P]
        
        # --- 7. 还原 ---
        out_patches = out_patches_flat.reshape(B, C, N, self.patch_len)
        x_out = out_patches.reshape(B, C, N * self.patch_len)
        
        if x_out.shape[2] < T:
            x_out = F.pad(x_out, (0, T - x_out.shape[2]))
        elif x_out.shape[2] > T:
            x_out = x_out[:, :, :T]
            
        x_out = x_out.permute(0, 2, 1)
        
        return x + self.gate * x_out