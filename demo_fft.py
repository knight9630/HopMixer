import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

# 1. 定义DFT序列分解类（保留你提供的原始代码，仅修复关键维度bug）
class DFT_series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, top_k):
        super(DFT_series_decomp, self).__init__()
        self.top_k = top_k

    def forward(self, x):
        # 修复1：明确指定对最后一维（时间维度）做FFT，兼容(batch, N, len)维度
        xf = torch.fft.rfft(x, dim=-1)
        freq = torch.abs(xf)
        # 修复2：用...兼容任意批量维度，仅置零每个序列的直流分量
        freq[..., 0] = 0
        # 修复3：指定dim=-1，对时间维度取top_k
        top_k_freq, top_list = torch.topk(freq, self.top_k, dim=-1)
        # 修复4：keepdim=True避免维度广播错误，确保筛选逻辑正确
        xf[freq <= top_k_freq.min(dim=-1, keepdim=True)[0]] = 0
        x_season = torch.fft.irfft(xf, dim=-1)
        x_trend = x - x_season
        return x_season, x_trend

# 2. 定义绘图保存函数
def plot_and_save_curves(x, trend, season, save_dir="dft_demo_plots", top_k=5):
    """
    绘制并保存原始曲线、趋势项、季节项
    :param x: 原始序列张量 (batch_size, N, seq_len)
    :param trend: 趋势项张量 (batch_size, N, seq_len)
    :param season: 季节项张量 (batch_size, N, seq_len)
    :param save_dir: 图片保存目录
    :param top_k: top_k参数（用于图片命名）
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 辅助函数：将GPU/CPU张量转为NumPy数组
    def tensor2numpy(tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return np.array(tensor)
    
    # 转换为NumPy数组
    x_np = tensor2numpy(x)
    trend_np = tensor2numpy(trend)
    season_np = tensor2numpy(season)
    
    # 取第一个样本、第一个N序列（最易展示效果）
    batch_idx = 0
    n_idx = 0
    x_seq = x_np[batch_idx, n_idx, :]
    trend_seq = trend_np[batch_idx, n_idx, :]
    season_seq = season_np[batch_idx, n_idx, :]
    
    # 创建画布
    plt.figure(figsize=(12, 6), dpi=100)
    # 绘制三条曲线
    plt.plot(x_seq, label=f'Original (trend + season + noise)', 
             color='blue', linewidth=2, alpha=0.8)
    plt.plot(trend_seq, label=f'Trend (low frequency)', 
             color='red', linewidth=2, alpha=0.8, linestyle='--')
    plt.plot(season_seq, label=f'Season (top-{top_k} high frequency)', 
             color='green', linewidth=2, alpha=0.8, linestyle=':')
    
    # 图表样式设置
    plt.title(f'DFT Series Decomposition (top_k={top_k})', fontsize=14, fontweight='bold')
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图片
    save_path = os.path.join(save_dir, f'dft_decomp_topk_{top_k}.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
    print(f"图片已保存至：{save_path}")
    # 打印关键统计值，验证分解效果
    print(f"原始序列均值：{np.mean(x_seq):.4f}")
    print(f"趋势项均值：{np.mean(trend_seq):.4f}")
    print(f"季节项均值：{np.mean(season_seq):.4f}")
    print(f"trend + season 与原始序列的误差：{np.mean(np.abs(x_seq - (trend_seq + season_seq))):.6f}\n")

# 3. 主函数：生成数据 + 分解 + 绘图保存
if __name__ == "__main__":
    # ===================== 步骤1：生成模拟时间序列 =====================
    # 模拟参数
    batch_size = 2   # 批次大小
    N = 3            # 序列数量
    seq_len = 200    # 序列长度（时间步）
    top_k = 10        # 保留前5个高频分量
    
    # 生成时间轴
    t = torch.linspace(0, 20, seq_len)  # 0到20共200个时间步
    # 构造趋势项：线性增长 + 轻微波动（低频）
    trend = 0.5 * t + 2 * torch.sin(0.1 * torch.pi * t)  # 低频趋势
    # 构造季节项：高频波动（2Hz + 5Hz）
    season = 3 * torch.sin(2 * torch.pi * 2 * t) + 1.5 * torch.sin(2 * torch.pi * 5 * t)
    # 原始序列 = 趋势 + 季节 + 少量噪声
    x = trend + season + 0.2 * torch.randn_like(t)
    
    # 扩展维度到(batch_size, N, seq_len)（匹配类的输入要求）
    x = x.unsqueeze(0).unsqueeze(0).repeat(batch_size, N, 1)
    trend = trend.unsqueeze(0).unsqueeze(0).repeat(batch_size, N, 1)
    print(f"输入序列形状：{x.shape} (batch_size, N, seq_len)")
    
    # ===================== 步骤2：初始化分解器并执行分解 =====================
    decomp = DFT_series_decomp(top_k=top_k)
    x_season, x_trend = decomp(x)  # 分解得到季节项和趋势项
    
    # ===================== 步骤3：绘图并保存 =====================
    plot_and_save_curves(x, x_trend, x_season, top_k=top_k)