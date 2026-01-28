import torch
import torch.nn as nn
from layers.Autoformer_EncDec import series_decomp
from layers.SelfAttention_Family import GapAttention, PatchAttention
from layers.Embed import DataEmbedding_wo_pos,DataEmbedding_inverted
from layers.StandardNorm import Normalize
from layers.Plugs import SeasonRectifiedAugmenter_CI, SeasonRectifiedAugmenter_CD, SeasonRectifiedAugmenter_CI_TopK, SeasonRectifiedAugmenter_CD_TopK
import matplotlib.pyplot as plt
import os
import numpy as np

# class DFT_series_decomp(nn.Module):
#     """
#     Series decomposition block
#     """

#     def __init__(self, top_k):
#         super(DFT_series_decomp, self).__init__()
#         self.top_k = top_k

#     def forward(self, x):
#         print("x shape:", x.shape)
#         x=x.permute(0,2,1)  # (B, C, L)
#         xf = torch.fft.rfft(x)
#         freq = abs(xf)
#         freq[0] = 0
#         top_k_freq, top_list = torch.topk(freq, self.top_k)
#         xf[freq <= top_k_freq.min()] = 0
#         x_season = torch.fft.irfft(xf)
#         x_trend = x - x_season
#         print(f"x: {x[0][0]}")
#         print(f"x_season: {x_season[0][0]}")
#         print(f"x_trend: {x_trend[0][0]}")
#         return x_season, x_trend

class DFT_series_decomp(nn.Module):
    """
    Series decomposition block using DFT (修复版)
    将序列分解为季节性成分(高频)和趋势成分(低频)
    """

    def __init__(self, top_k):
        super(DFT_series_decomp, self).__init__()
        self.top_k = top_k

    def forward(self, x):
        # x: 输入形状 (B, L, C) -> 调整为 (B, C, L) 便于处理
        x = x.permute(0, 2, 1)  # (B, C, L)
        B, C, L = x.shape
        
        # 1. 进行实数快速傅里叶变换
        xf = torch.fft.rfft(x)  # (B, C, L//2+1) 复数结果
        freq_amp = torch.abs(xf)  # 频率幅值 (B, C, L//2+1)
        
        # 2. 去除直流分量 (0频率)
        freq_amp = freq_amp.clone()
        freq_amp[..., 0] = 0  # 使用...适配任意维度
        
        # 3. 找到幅值最大的top-k个频率索引
        # top_k_values: (B, C, top_k), top_k_indices: (B, C, top_k)
        top_k_values, top_k_indices = torch.topk(freq_amp, self.top_k, dim=-1)
        
        # 4. 初始化全零的频域张量，只保留top-k个频率分量
        xf_season = torch.zeros_like(xf)
        # 使用scatter_将top-k的频率分量回填
        # 为每个样本-通道维度扩展索引
        batch_idx = torch.arange(B).unsqueeze(1).unsqueeze(2).expand(-1, C, self.top_k)
        chan_idx = torch.arange(C).unsqueeze(0).unsqueeze(2).expand(B, -1, self.top_k)
        
        # 只保留top-k个频率分量
        xf_season[batch_idx, chan_idx, top_k_indices] = xf[batch_idx, chan_idx, top_k_indices]
        
        # 5. 逆傅里叶变换回到时域
        x_season = torch.fft.irfft(xf_season, n=L)  # n=L确保输出长度和输入一致
        x_trend = x - x_season

        # 还原原始维度 (B, C, L) -> (B, L, C)
        x_season = x_season.permute(0, 2, 1)
        x_trend = x_trend.permute(0, 2, 1)
        
        return x_season, x_trend

class ThreePartDFTDecomp(nn.Module):
    """
    基于DFT的三部分解：Season + Trend + Noise
    """
    def __init__(self, top_k_season, top_k_trend):
        super(ThreePartDFTDecomp, self).__init__()
        self.top_k_season = top_k_season # 例如 5
        self.top_k_trend = top_k_trend   # 例如 5 (保留最低的5个频率)

    def forward(self, x):
        # x: (B, L, C) -> (B, C, L)
        x_in = x.permute(0, 2, 1)
        B, C, L = x_in.shape
        
        # 1. FFT 变换
        xf = torch.fft.rfft(x_in) # (B, C, L//2+1)
        freq_amp = torch.abs(xf)
        
        # --- 提取 Season (振幅最大的 Top-k, 排除直流分量) ---
        # 暂时把直流分量(index 0)屏蔽掉，不让Season选它
        freq_amp_no_dc = freq_amp.clone()
        freq_amp_no_dc[..., 0] = 0 
        
        # 选出振幅最大的 k1 个频率
        _, season_indices = torch.topk(freq_amp_no_dc, self.top_k_season, dim=-1)
        
        # 构建 Season 频域信号
        xf_season = torch.zeros_like(xf)
        batch_idx = torch.arange(B).unsqueeze(1).unsqueeze(2).expand(-1, C, self.top_k_season)
        chan_idx = torch.arange(C).unsqueeze(0).unsqueeze(2).expand(B, -1, self.top_k_season)
        xf_season[batch_idx, chan_idx, season_indices] = xf[batch_idx, chan_idx, season_indices]
        
        # Season 变回时域
        x_season = torch.fft.irfft(xf_season, n=L)
        
        # --- 提取 Trend (剩下的信号中，频率最低的 Top-k) ---
        # 剩余信号的频谱 = 原始 - Season
        xf_residue = xf - xf_season
        
        # 强制选取最低频的 k2 个分量 (0, 1, 2 ... k2-1)
        # 注意：这里我们直接切片取前 k2 个，因为 rfft 的结果是按频率从低到高排列的
        # index 0 是直流分量(均值)，index 1 是最低频...
        trend_indices = torch.arange(self.top_k_trend, device=x.device)
        
        xf_trend = torch.zeros_like(xf)
        # 对于所有 Batch 和 Channel，都取前 k2 个频率
        xf_trend[..., :self.top_k_trend] = xf_residue[..., :self.top_k_trend]
        
        # Trend 变回时域
        x_trend = torch.fft.irfft(xf_trend, n=L)
        
        # --- 提取 Noise ---
        # 原始 - Season - Trend
        # 这里的 x_trend 和 x_season 还在 (B, C, L) 维度
        x_noise = x_in - x_season - x_trend
        
        # 还原维度 (B, L, C)
        return x_season.permute(0, 2, 1), x_trend.permute(0, 2, 1), x_noise.permute(0, 2, 1)
    
class MultiScaleSeasonMixing(nn.Module):
    """
    Bottom-up mixing season pattern
    """

    def __init__(self, configs):
        super(MultiScaleSeasonMixing, self).__init__()

        self.down_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        #configs.seq_len // (configs.down_sampling_window ** i),
                        #configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len ,
                        configs.seq_len 
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        #configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        #configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len ,
                        configs.seq_len 
                    ),

                )
                for i in range(configs.down_sampling_layers)
            ]
        )

    def forward(self, season_list):

        # mixing high->low
        out_high = season_list[0]
        out_low = season_list[1]
        out_season_list = [out_high.permute(0, 2, 1)]

        for i in range(len(season_list) - 1):
            out_low_res = self.down_sampling_layers[i](out_high)
            out_low = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_season_list.append(out_high.permute(0, 2, 1))

        return out_season_list

class MultiScaleTrendMixing(nn.Module):
    """
    Top-down mixing trend pattern
    """

    def __init__(self, configs):
        super(MultiScaleTrendMixing, self).__init__()

        self.up_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        #configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        #configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len ,
                        configs.seq_len 
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        #configs.seq_len // (configs.down_sampling_window ** i),
                        #configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len ,
                        configs.seq_len 
                    ),
                )
                for i in reversed(range(configs.down_sampling_layers))
            ])

    def forward(self, trend_list):

        # mixing low->high
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()
        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]
        out_trend_list = [out_low.permute(0, 2, 1)]

        for i in range(len(trend_list_reverse) - 1):
            out_high_res = self.up_sampling_layers[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 1))

        out_trend_list.reverse()
        return out_trend_list

class PastDecomposableMixing(nn.Module):
    def __init__(self, configs):
        super(PastDecomposableMixing, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window

        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)
        self.channel_independence = configs.channel_independence
        self.batch_num = 0

        if configs.decomp_method == 'moving_avg':
            self.decompsition = series_decomp(configs.moving_avg)
        elif configs.decomp_method == "dft_decomp":
            self.decompsition = DFT_series_decomp(configs.top_k)
        elif configs.decomp_method == "three_part_dft_decomp":
            self.decompsition = ThreePartDFTDecomp(configs.top_k_season, configs.top_k_trend)
        else:
            raise ValueError('decompsition is error')

        self.alpha = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(configs.down_sampling_layers + 1)])

        self.fusion_weights = nn.ParameterList([
            nn.Parameter(torch.ones(3)) for _ in range(configs.down_sampling_layers + 1)
        ])

        self.gating_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(configs.d_model * 3, configs.d_model), # 假设维度是 d_model
                nn.Tanh(),
                nn.Linear(configs.d_model, 3), # 输出三个系数
                nn.Softmax(dim=-1) # 归一化，让三者之和为 1 (或者用 Sigmoid)
            )
            for _ in range(configs.down_sampling_layers + 1)
        ])
            
        if configs.channel_independence==1:
            self.rectified_layer = torch.nn.ModuleList(
                [
                    # SeasonRectifiedAugmenter_CI_TopK(configs.agg_patch, configs.agg_patch, configs.agg_top_k)
                    # for _ in range(configs.down_sampling_layers + 1)
                    SeasonRectifiedAugmenter_CD(configs.agg_patch, configs.agg_patch)
                    for _ in range(configs.down_sampling_layers + 1)
                ]
            )
        else:
            self.rectified_layer = torch.nn.ModuleList(
                [
                    SeasonRectifiedAugmenter_CD_TopK(configs.agg_patch, configs.agg_patch, configs.agg_top_k)
                    for _ in range(configs.down_sampling_layers + 1)
                ]
            )
            
        if configs.channel_independence==1:
            self.out_cross_layer = nn.Sequential(
                nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
                nn.GELU(),
                nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
            )
        else:
            self.out_cross_layer = nn.Sequential(
                nn.Linear(in_features=configs.enc_in+configs.time_d, out_features=2*(configs.enc_in+configs.time_d)),
                nn.GELU(),
                nn.Linear(in_features=2*(configs.enc_in+configs.time_d), out_features=configs.enc_in+configs.time_d),
            )

        if configs.channel_independence==1:
            self.Season_MLP = torch.nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
                        nn.GELU(),
                        nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
                    )
                    for i in range(configs.down_sampling_layers + 1)
                ]
            )
        else:
            self.Season_MLP = torch.nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(in_features=configs.enc_in+configs.time_d, out_features=2*(configs.enc_in+configs.time_d)),
                        nn.GELU(),
                        nn.Linear(in_features=2*(configs.enc_in+configs.time_d), out_features=configs.enc_in+configs.time_d),
                    )
                    for i in range(configs.down_sampling_layers + 1)
                ]
            )
        
        if configs.channel_independence==1:
            self.Trend_MLP = torch.nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
                        nn.GELU(),
                        nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
                    )
                    for i in range(configs.down_sampling_layers + 1)
                ]
            )
        else:
            self.Trend_MLP = torch.nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(in_features=configs.enc_in+configs.time_d, out_features=2*(configs.enc_in+configs.time_d)),
                        nn.GELU(),
                        nn.Linear(in_features=2*(configs.enc_in+configs.time_d), out_features=configs.enc_in+configs.time_d),
                    )
                    for i in range(configs.down_sampling_layers + 1)
                ]
            )
            
        if configs.channel_independence==1:
            self.multigapattention1 = torch.nn.ModuleList(
                [
                    # GapAttention(configs.seq_len,24,3*(2**(configs.down_sampling_layers-i)))
                    # GapAttention(configs.seq_len,32,4*(2**i))
                    GapAttention(configs.seq_len,configs.global_patch,configs.local_patch)
                    for i in range(configs.down_sampling_layers + 1)
                ]
            )
            self.multigapattention2 = torch.nn.ModuleList(
                [
                    # GapAttention(configs.seq_len,24,3*(2**(configs.down_sampling_layers-i)))
                    # GapAttention(configs.seq_len,32,4*(2**i))
                    GapAttention(configs.seq_len,configs.global_patch,configs.local_patch)
                    for i in range(configs.down_sampling_layers + 1)
                ]
            )
        else:
            self.multigapattention1 = torch.nn.ModuleList(
                [
                    # GapAttention(configs.t_model,256,32*(2**(configs.down_sampling_layers-i)))
                    GapAttention(configs.t_model,configs.global_patch,configs.local_patch)#适应24，8，更改了3的倍数的长度嵌入长度：768
                    for i in range(configs.down_sampling_layers + 1)
                ]
            )
            self.multigapattention2 = torch.nn.ModuleList(
                [
                    # GapAttention(configs.t_model,256,32*(2**(configs.down_sampling_layers-i)))
                    GapAttention(configs.t_model,configs.global_patch,configs.local_patch)
                    for i in range(configs.down_sampling_layers + 1)
                ]
            )


    def plot_first_sample_curves(self, x, trend, season, layer_idx, save_dir="batch_plots"):
            """
            类内方法：仅绘制第一个样本（batch_idx=0）的所有N序列，保存到指定目录
            适配GPU/CPU张量，自动转换为NumPy数组
            """
            # 核心修复：将GPU张量转移到CPU并转为NumPy数组
            def tensor_to_numpy(tensor):
                # 如果是张量，先转CPU再转NumPy；如果已经是NumPy则直接返回
                if isinstance(tensor, torch.Tensor):
                    return tensor.detach().cpu().numpy()  # detach()避免梯度关联，cpu()转CPU，numpy()转数组
                return np.array(tensor)
            
            # 转换所有输入数据为NumPy数组（兼容GPU/CPU张量）
            x_np = tensor_to_numpy(x)
            trend_np = tensor_to_numpy(trend)
            season_np = tensor_to_numpy(season)
            # print(f"x: {x_np[0][0]}")
            # print(f"season: {season_np[0][0]}")
            # print(f"trend: {trend_np[0][0]}")
            
            # 1. 创建保存目录
            os.makedirs(save_dir, exist_ok=True)
            
            # 2. 提取第一个样本的数据（固定batch_idx=0）
            batch_idx = 13
            x_first = x_np[batch_idx]    # shape=(N, len)
            trend_first = trend_np[batch_idx]
            season_first = season_np[batch_idx]
            N_count = x_first.shape[0]
            
            # 3. 创建画布：每个N序列一个子图
            fig, axes = plt.subplots(N_count, 1, figsize=(12, 4*N_count), dpi=100)
            fig.suptitle(f'First Sample (Batch {batch_idx}): All N Sequences', 
                        fontsize=16, fontweight='bold', y=0.98)
            
            # 4. 遍历第一个样本的每个N序列绘图
            for n_idx in range(N_count):
                ax = axes[n_idx] if N_count > 1 else axes  # 兼容N=1的情况
                
                # 提取单条序列数据
                x_seq = x_first[n_idx]
                trend_seq = trend_first[n_idx]
                season_seq = season_first[n_idx]
                
                # 绘制三条曲线
                ax.plot(x_seq, label='Original (x)', color='blue', linewidth=2, alpha=0.8)
                ax.plot(trend_seq, label='Trend', color='red', linewidth=2, alpha=0.8, linestyle='--')
                ax.plot(season_seq, label='Season', color='green', linewidth=2, alpha=0.8, linestyle=':')
                
                # 子图样式设置
                ax.set_title(f'N = {n_idx}', fontsize=12)
                ax.set_xlabel('Sequence Index (len)', fontsize=10)
                ax.set_ylabel('Value', fontsize=10)
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)
            
            # 5. 调整布局并保存图片
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            save_path = os.path.join(save_dir, f'first_sample_batch_{self.batch_num}_{layer_idx}.png')
            plt.savefig(save_path, bbox_inches='tight')
            plt.close(fig)  # 释放内存
            
            print(f"第一个样本的图片已保存至: {save_path}")

    def forward(self, x_list):
        length_list = []
        for x in x_list:
            _, T, _ = x.size()
            length_list.append(T)

        # Decompose to obtain the season and trend
        out_season_list = []
        out_trend_list = []
        out_season_noise_list = []
        index=0
        for x in x_list:
            # print(f'xshape{x.shape}')
            season, trend, noise = self.decompsition(x)
            # print(f'seasonshape{season.shape}')
            # self.plot_first_sample_curves(x.permute(0, 2, 1), trend.permute(0, 2, 1), season.permute(0, 2, 1), index, save_dir="batch_plots_DFT_3")
            trend=self.Trend_MLP[index](trend)
            season=self.multigapattention2[index](season)
            season_with_noise = self.rectified_layer[index](season + noise)
            # season=self.Season_MLP[index](season)
            # trend=self.multigapattention2[index](trend)
            # patch_trend=self.patchattention[index](trend)
            # trend=sub_trend+patch_trend
            index=index+1

            out_season_list.append(season)
            out_trend_list.append(trend)
            out_season_noise_list.append(season_with_noise)
        self.batch_num += 1 

        out_list = []
        layer_num = 0
        for ori, out_season, out_trend, out_season_noise, length in zip(x_list, out_season_list, out_trend_list,out_season_noise_list,
                                                      length_list):
            out = out_season + out_trend + self.alpha[layer_num] * out_season_noise

            # w = torch.softmax(self.fusion_weights[layer_num], dim=0)
            # out = w[0] * out_trend + w[1] * out_season + w[2] * out_season_noise

            # combined = torch.cat([out_trend, out_season, out_season_noise], dim=-1)
    
            # # 2. 计算门控系数 [B, T, 3] (假设是对每个时间步动态加权)
            # # 你的 gating_layer 输入维度需要调整匹配
            # gate_weights = self.gating_layers[layer_num](combined) 
            
            # # 3. 分别提取权重
            # w_trend = gate_weights[..., 0:1]
            # w_season = gate_weights[..., 1:2]
            # w_sra = gate_weights[..., 2:3]
            
            # # 4. 动态融合
            # out = w_trend * out_trend + w_season * out_season + w_sra * out_season_noise
            out = ori + self.out_cross_layer(out)
            # out = ori + self.out_cross_layer(ori)
            out_list.append(out[:, :length, :])
            layer_num = layer_num + 1
        return out_list

class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window
        self.channel_independence = configs.channel_independence
        self.pdm_blocks = nn.ModuleList([PastDecomposableMixing(configs)
                                         for _ in range(configs.e_layers)])

        self.preprocess = series_decomp(configs.moving_avg)
        self.enc_in = configs.enc_in
        self.use_future_temporal_feature = configs.use_future_temporal_feature

        if self.channel_independence == 1:
            self.enc_embedding = DataEmbedding_wo_pos(1, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)
        else:
            self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.t_model, configs.embed, configs.freq,
                                                      configs.dropout)

        self.layer = configs.e_layers
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            if self.channel_independence == 1:
                self.predict_layers = torch.nn.ModuleList(
                    [
                        torch.nn.Linear(
                            configs.seq_len,
                            configs.pred_len,
                        )
                        for i in range(configs.down_sampling_layers + 1)
                    ]
                )
            else:
                self.predict_layers = torch.nn.ModuleList(
                    [
                        torch.nn.Linear(
                            configs.t_model,
                            configs.pred_len,
                        )
                        for i in range(configs.down_sampling_layers + 1)
                    ]
                )

            if self.channel_independence == 1:
                self.projection_layer = nn.Linear(
                    configs.d_model, 1, bias=True)
            else:
                self.projection_layer = nn.Linear(
                    configs.d_model, configs.c_out, bias=True)

                self.out_res_layers = torch.nn.ModuleList([
                    torch.nn.Linear(
                        configs.seq_len,
                        configs.seq_len,
                    )
                    for i in range(configs.down_sampling_layers + 1)
                ])

                self.regression_layers = torch.nn.ModuleList(
                    [
                        torch.nn.Linear(
                            #configs.seq_len // (configs.down_sampling_window ** i),
                            configs.seq_len,
                            configs.pred_len,
                        )
                        for i in range(configs.down_sampling_layers + 1)
                    ]
                )

            self.normalize_layers = torch.nn.ModuleList(
                [
                    Normalize(self.configs.enc_in, affine=True, non_norm=True if configs.use_norm == 0 else False)#configs.use_norm=1
                    for i in range(configs.down_sampling_layers + 1)
                ]
            )

            self.out_process =torch.nn.Linear((configs.down_sampling_layers+1),1)

    def __multi_scale_process_inputs(self, x_enc, x_mark_enc):
        if self.configs.down_sampling_method == 'max':
            down_pool = torch.nn.MaxPool1d(self.configs.down_sampling_window)
        elif self.configs.down_sampling_method == 'avg':
            #down_pool = torch.nn.AvgPool1d(self.configs.down_sampling_window)
            down_pool=nn.ModuleList()
            down_pool.append(torch.nn.AvgPool1d(kernel_size=2, stride=1))
            down_pool.append(torch.nn.AvgPool1d(kernel_size=4, stride=1))
            down_pool.append(torch.nn.AvgPool1d(kernel_size=8, stride=1))            

        elif self.configs.down_sampling_method == 'conv':
            padding = 1 if torch.__version__ >= '1.5.0' else 2
            down_pool = nn.Conv1d(in_channels=self.configs.enc_in, out_channels=self.configs.enc_in,
                                  kernel_size=3, padding=padding,
                                  stride=self.configs.down_sampling_window,
                                  padding_mode='circular',
                                  bias=False)
        else:
            return x_enc, x_mark_enc
        # B,T,C -> B,C,T
        x_enc = x_enc.permute(0, 2, 1)

        x_enc_ori = x_enc
        x_mark_enc_mark_ori = x_mark_enc

        x_enc_sampling_list = []
        x_mark_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
        x_mark_sampling_list.append(x_mark_enc)

        for i in range(self.configs.down_sampling_layers):
           
            #print(x_enc_ori.shape)
            dis=2**(i+1)-1
            mean_value=torch.mean(x_enc_ori,dim=2,keepdim=True)
            mean_value_repeated=mean_value.repeat(1,1,dis)
            x_enc_ori=torch.cat((mean_value_repeated,x_enc_ori),dim=2)
            #print(x_enc_ori.shape)
            x_enc_sampling = down_pool[i](x_enc_ori)
            #print(x_enc_sampling.shape)

            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling
            

            if x_mark_enc_mark_ori is not None:
                x_mark_sampling_list.append(x_mark_enc_mark_ori)
                #x_mark_sampling_list.append(x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :])
                #x_mark_enc_mark_ori = x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :]

        x_enc = x_enc_sampling_list
        if x_mark_enc_mark_ori is not None:
            x_mark_enc = x_mark_sampling_list
        else:
            x_mark_enc = x_mark_enc

        return x_enc, x_mark_enc

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x_enc, x_mark_enc = self.__multi_scale_process_inputs(x_enc, x_mark_enc)
        
        x_list = []
        x_mark_list = []
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                    x_mark = x_mark.repeat(N, 1, 1)
                x_list.append(x)
                x_mark_list.append(x_mark)
        else:
            for i, x in zip(range(len(x_enc)), x_enc, ):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm')  
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)
        
        # embedding
        enc_out_list = []
        #x_list = self.pre_enc(x_list)#单通道不处理；多通道会得到season和trend两个列表
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_list)), x_list, x_mark_list):
                #print(x.shape)
                #print(x_mark.shape)
                if self.channel_independence == 1:
                    enc_out = self.enc_embedding(x, x_mark)#[B,T,C]
                else:
                    enc_out = self.enc_embedding(x, x_mark).permute(0,2,1)#[B,T,C]
                #print(f'enc_out{enc_out.shape}')               
                #print(enc_out.shape)
                enc_out_list.append(enc_out)
        else:
            for i, x in zip(range(len(x_list)), x_list):
                if self.channel_independence == 1:
                    enc_out = self.enc_embedding(x, None)#[B,T,C]
                else:
                    enc_out = self.enc_embedding(x, None).permute(0,2,1)#[B,T,C]
                enc_out_list.append(enc_out)

        # Past Decomposable Mixing as encoder for past
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)


        # Future Multipredictor Mixing as decoder for future
        dec_out_list = self.future_multi_mixing(B, enc_out_list, x_list)

        #dec_out = torch.mean(torch.stack(dec_out_list, dim=-1),dim=-1)
        dec_out = torch.stack(dec_out_list, dim=-1).sum(-1)
        # dec_out=dec_out_list[3]
        #print(dec_out.shape)
        dec_out = self.normalize_layers[0](dec_out, 'denorm')
        return dec_out

    def future_multi_mixing(self, B, enc_out_list, x_list):
        dec_out_list = []
        _,_,N=x_list[0].size()
        if self.channel_independence == 1:
            for i, enc_out in zip(range(len(x_list)), enc_out_list):
                #print(enc_out.shape)
                if self.use_future_temporal_feature:
                    enc_out = enc_out + self.x_mark_dec
                    enc_out = self.projection_layer(enc_out)
                else:
                    enc_out = self.projection_layer(enc_out)
                
                enc_out = enc_out.reshape(B, self.configs.c_out, self.seq_len).permute(0, 2, 1).contiguous()

                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(0, 2, 1) # align temporal dimension
                
                #dec_out = dec_out.reshape(B, self.configs.c_out, self.pred_len).permute(0, 2, 1).contiguous()
                dec_out_list.append(dec_out)

        else:
            for i, enc_out in zip(range(len(x_list)), enc_out_list):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                    0, 2, 1)[:,:,:N]  # align temporal dimension
                #dec_out = self.projection_layer(dec_out)
                dec_out_list.append(dec_out)

        return dec_out_list

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            #时序数据（0，seq),时间数据（0，seq),时序数据（seq-label，seq+pred)，时间数据（seq-label，seq+pred)
            dec_out_list = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out_list
        else:
            raise ValueError('Only forecast tasks implemented yet')