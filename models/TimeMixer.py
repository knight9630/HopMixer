import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from layers.Autoformer_EncDec import series_decomp
from layers.Embed import DataEmbedding_wo_pos
from layers.StandardNorm import Normalize


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
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
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
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** i),
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

        self.decompsition = series_decomp(configs.moving_avg)

        if configs.channel_independence == 0:
            self.cross_layer = nn.Sequential(
                nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
                nn.GELU(),
                nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
            )

        # Mixing season
        self.mixing_multi_scale_season = MultiScaleSeasonMixing(configs)

        # Mxing trend
        self.mixing_multi_scale_trend = MultiScaleTrendMixing(configs)

        self.out_cross_layer = nn.Sequential(
            nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
            nn.GELU(),
            nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
        )

    def forward(self, x_list):
        length_list = []
        for x in x_list:
            _, T, _ = x.size()
            length_list.append(T)

        # Decompose to obtain the season and trend
        season_list = []
        trend_list = []
        for x in x_list:
            season, trend = self.decompsition(x)
            if self.channel_independence == 0:
                season = self.cross_layer(season)
                trend = self.cross_layer(trend)
            season_list.append(season.permute(0, 2, 1))
            trend_list.append(trend.permute(0, 2, 1))

        # bottom-up season mixing
        out_season_list = self.mixing_multi_scale_season(season_list)
        # top-down trend mixing
        out_trend_list = self.mixing_multi_scale_trend(trend_list)

        out_list = []
        for ori, out_season, out_trend, length in zip(x_list, out_season_list, out_trend_list,
                                                      length_list):
            out = out_season + out_trend
            if self.channel_independence:
                out = ori + self.out_cross_layer(out)
            out_list.append(out[:, :length, :])
        return out_list

class PatchMLP(nn.Module):
    def __init__(self, dim, hidden=None, dropout=0.0):
        super(PatchMLP, self).__init__()
        if hidden is None:
            hidden = max(16, dim // 2)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: [L]
        return self.norm(x + self.net(x))
    
class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.channel = configs.enc_in
        self.down_sampling_window = configs.down_sampling_window
        self.channel_independence = configs.channel_independence
        self.pdm_blocks = nn.ModuleList([PastDecomposableMixing(configs)
                                         for _ in range(configs.e_layers)])

        self.preprocess = series_decomp(configs.moving_avg)
        self.enc_in = configs.enc_in
        self.global_patch = configs.global_patch

        self.P_per_channel = configs.seq_len // configs.global_patch
        self.patch_count = configs.enc_in * self.P_per_channel  # configs.enc_in * (seq_len//global_patch)
         # 为每个目标 patch m 初始化 self.patch_count 个线性层：self.patch_linears[m][j]
        # 结构为 ModuleList[ M ]，每项为 ModuleList[ M ] 的 nn.Linear(L->L)
        self.patch_linears = nn.ModuleList(
            [nn.ModuleList([PatchMLP(self.global_patch, hidden=self.global_patch//2, dropout=0.1)
                            for _ in range(self.patch_count)])
             for _ in range(self.patch_count)]
        )

        if self.channel_independence:
            self.enc_embedding = DataEmbedding_wo_pos(1, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)
        else:
            self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)

        self.layer = configs.e_layers
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.predict_layers = torch.nn.ModuleList(
                [
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.pred_len,
                    )
                    for i in range(configs.down_sampling_layers + 1)
                ]
            )

            if self.channel_independence:
                self.projection_layer = nn.Linear(
                    configs.d_model, 1, bias=True)
            else:
                self.projection_layer = nn.Linear(
                    configs.d_model, configs.c_out, bias=True)

                self.out_res_layers = torch.nn.ModuleList([
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    )
                    for i in range(configs.down_sampling_layers + 1)
                ])

                self.regression_layers = torch.nn.ModuleList(
                    [
                        torch.nn.Linear(
                            configs.seq_len // (configs.down_sampling_window ** i),
                            configs.pred_len,
                        )
                        for i in range(configs.down_sampling_layers + 1)
                    ]
                )

            self.normalize_layers = torch.nn.ModuleList(
                [
                    Normalize(self.configs.enc_in, affine=True, non_norm=True if configs.use_norm == 0 else False)
                    for i in range(configs.down_sampling_layers + 1)
                ]
            )

    def out_projection(self, dec_out, i, out_res):
        dec_out = self.projection_layer(dec_out)
        out_res = out_res.permute(0, 2, 1)
        out_res = self.out_res_layers[i](out_res)
        out_res = self.regression_layers[i](out_res).permute(0, 2, 1)
        dec_out = dec_out + out_res
        return dec_out

    def pre_enc(self, x_list):
        if self.channel_independence:
            return (x_list, None)
        else:
            out1_list = []
            out2_list = []
            for x in x_list:
                x_1, x_2 = self.preprocess(x)
                out1_list.append(x_1)
                out2_list.append(x_2)
            return (out1_list, out2_list)

    def __multi_scale_process_inputs(self, x_enc, x_mark_enc):
        if self.configs.down_sampling_method == 'max':
            down_pool = torch.nn.MaxPool1d(self.configs.down_sampling_window, return_indices=False)
        elif self.configs.down_sampling_method == 'avg':
            down_pool = torch.nn.AvgPool1d(self.configs.down_sampling_window)
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
            x_enc_sampling = down_pool(x_enc_ori)

            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling

            if x_mark_enc_mark_ori is not None:
                x_mark_sampling_list.append(x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :])
                x_mark_enc_mark_ori = x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :]

        x_enc = x_enc_sampling_list
        if x_mark_enc_mark_ori is not None:
            x_mark_enc = x_mark_sampling_list
        else:
            x_mark_enc = x_mark_enc

        return x_enc, x_mark_enc
    
    # ...existing code...
    def group_patches(self, corr, method='agglomerative', threshold=0.6, n_clusters=4, distance_threshold=None, min_size=1):
        """
        基于相关矩阵把每个样本的 M 个 patch 分组。
        参数:
          corr: torch.Tensor [B, M, M]
          method: 'threshold'|'agglomerative'|'kmeans'|'spectral'
          threshold: 用于 'threshold' 方法的二值化门限（|r| >= threshold）
          n_clusters: 聚类数（用于 kmeans / agglomerative 若未用 distance_threshold）
          distance_threshold: 若指定，agglomerative 使用 distance_threshold（并且 n_clusters=None）
          min_size: 过滤掉小于该大小的组
        返回:
          groups_all: list len B，每项是 list of groups（group 为 python list of int）
        """
        B, M, _ = corr.shape
        corr_np = corr.detach().cpu().numpy()
        groups_all = []

        for b in range(B):
            Cb = corr_np[b].copy()  # shape (M,M)
            # guard numerical issues
            Cb = np.nan_to_num(Cb, nan=0.0)
            if method == 'threshold':
                adj = (np.abs(Cb) >= float(threshold)).astype(np.uint8)
                np.fill_diagonal(adj, 1)
                visited = np.zeros(M, dtype=bool)
                comps = []
                for i in range(M):
                    if not visited[i]:
                        stack = [i]
                        members = []
                        while stack:
                            v = stack.pop()
                            if visited[v]:
                                continue
                            visited[v] = True
                            members.append(int(v))
                            neighs = np.where(adj[v])[0]
                            for u in neighs:
                                if not visited[u]:
                                    stack.append(u)
                        if len(members) >= min_size:
                            comps.append(members)
                groups_all.append(comps)

            elif method == 'agglomerative':
                # dist should be non-negative, smaller means more similar
                dist = 1.0 - Cb
                dist = np.clip(dist, 0.0, 2.0)
                # sklearn's AgglomerativeClustering with precomputed distance
                if distance_threshold is not None:
                    model = AgglomerativeClustering(n_clusters=None, affinity='precomputed',
                                                    linkage='average', distance_threshold=distance_threshold)
                else:
                    model = AgglomerativeClustering(n_clusters=int(n_clusters), affinity='precomputed',
                                                    linkage='average')
                labels = model.fit_predict(dist)
                comps = []
                for k in np.unique(labels):
                    members = np.where(labels == k)[0].tolist()
                    if len(members) >= min_size:
                        comps.append(members)
                groups_all.append(comps)

            elif method == 'kmeans':
                # use corr rows as features (each patch's similarity vector)
                Xfeat = Cb.copy()  # (M,M)
                # optional: normalize rows to zero mean
                Xfeat = Xfeat - Xfeat.mean(axis=1, keepdims=True)
                # run kmeans
                model = KMeans(n_clusters=int(n_clusters), n_init=10)
                labels = model.fit_predict(Xfeat)
                comps = []
                for k in np.unique(labels):
                    members = np.where(labels == k)[0].tolist()
                    if len(members) >= min_size:
                        comps.append(members)
                groups_all.append(comps)

            elif method == 'spectral':
                # spectral expects affinity matrix in [0,1]
                Aff = (Cb + 1.0) / 2.0
                Aff = np.clip(Aff, 0.0, 1.0)
                model = SpectralClustering(n_clusters=int(n_clusters), affinity='precomputed', assign_labels='kmeans')
                labels = model.fit_predict(Aff)
                comps = []
                for k in np.unique(labels):
                    members = np.where(labels == k)[0].tolist()
                    if len(members) >= min_size:
                        comps.append(members)
                groups_all.append(comps)

            else:
                raise ValueError(f"unknown grouping method: {method}")

        return groups_all
    
    def interp_fill(self, patch, lag):
        # patch: torch.Tensor (L,)
        arr = patch.detach().cpu().numpy().astype(np.float64)
        Lloc = arr.shape[0]
        filled = np.full(Lloc, np.nan, dtype=np.float64)
        if lag > 0:
            filled[lag:] = arr[:Lloc - lag]
        elif lag < 0:
            s = -lag
            filled[:Lloc - s] = arr[s:]
        else:
            return patch
        known = np.where(~np.isnan(filled))[0]
        if known.size == 0:
            # 全为空，退回原 patch
            return patch
        missing = np.where(np.isnan(filled))[0]
        # np.interp 要求 known 有序且至少一个点：对缺失位做线性插值/外插
        filled[missing] = np.interp(missing, known, filled[known])
        return torch.tensor(filled.astype(np.float32), device=patch.device, dtype=patch.dtype)
    
    # 端点线性外推（用邻近两个点的斜率）
    def extrap_fill(self, patch, lag):
        Lloc = patch.size(0)
        if lag == 0:
            return patch
        if lag > 0:
            body = patch[:Lloc - lag]
            if body.size(0) >= 2:
                slope = (body[1] - body[0]).item()
            elif body.size(0) == 1:
                slope = 0.0
            else:
                slope = 0.0
            left_vals = torch.tensor([body[0].item() - slope * (i + 1) for i in reversed(range(lag))],
                                        device=patch.device, dtype=patch.dtype)
            return torch.cat([left_vals, body])
        else:  # lag < 0
            s = -lag
            body = patch[s:]
            if body.size(0) >= 2:
                slope = (body[-1] - body[-2]).item()
            elif body.size(0) == 1:
                slope = 0.0
            else:
                slope = 0.0
            right_vals = torch.tensor([body[-1].item() + slope * (i + 1) for i in range(s)],
                                        device=patch.device, dtype=patch.dtype)
            return torch.cat([body, right_vals])

    def compute_group_reference_and_offsets(self, patches, groups_all, max_shift, ref_method='mean', fill_method = 'interp',eps=1e-8):
        """
        patches: Tensor [B, C, P, L]
        groups_all: list len B, 每项是 list of groups (group: list of patch indices in 0..M-1)
        max_shift: 最大搜索偏移（会在 [-max_shift, max_shift] 范围内搜索）
        ref_method: 'mean' 或 'medoid'
        返回:
          refs_all: list len B，每项 list of tensors (每个 group 的 reference, shape [L])
          offsets: Tensor [B, M] （对每个 patch 找到的最佳偏移，patch 索引 0..M-1）
          aligned: Tensor [B, M, L] （对齐后的 patch，超出用 0 填充）
        """
        B, C, P, L = patches.shape
        M = C * P
        X = patches.reshape(B, M, L).float()  # [B, M, L]
        device = X.device
        offsets = torch.zeros(B, M, dtype=torch.long, device=device)
        aligned = torch.zeros_like(X)
        refs_all = []

        for b in range(B):
            Xb = X[b]  # [M, L]
            groups = groups_all[b]
            refs_b = []
            # 构造索引到 group 的映射（若 patch 不属于任何组，assign -1）
            idx2grp = -torch.ones(M, dtype=torch.long, device=device)
            for gi, grp in enumerate(groups):
                for idx in grp:
                    idx2grp[idx] = gi

            # 计算每个 group 的 reference
            for gi, grp in enumerate(groups):
                members = torch.tensor(grp, device=device, dtype=torch.long)
                if len(members) == 0:
                    refs_b.append(torch.zeros(L, device=device))
                    continue
                if ref_method == 'mean':
                    ref = Xb[members].mean(dim=0)  # [L]
                else:  # medoid
                    Xm = Xb[members]  # [g, L]
                    Xm0 = Xm - Xm.mean(dim=1, keepdim=True)
                    S = torch.matmul(Xm0, Xm0.transpose(0, 1))  # [g,g]
                    stds = Xm0.norm(dim=1)
                    denom = (stds[:, None] * stds[None, :]).clamp(min=eps)
                    corrm = S / denom
                    sumcorr = corrm.sum(dim=1)
                    medoid_idx = torch.argmax(sumcorr)
                    ref = Xm[medoid_idx]
                refs_b.append(ref)
            
            # 对每个 patch 搜索最佳偏移
            for m in range(M):
                gi = int(idx2grp[m].item())
                if gi < 0:
                    ref = Xb[m]
                    best_lag = 0
                else:
                    ref = refs_b[gi]  # [L]
                    best_lag = 0
                    best_corr = -2.0
                    # 遍历 lag
                    for lag in range(-max_shift, max_shift + 1):
                        if lag >= 0:
                            n = L - lag
                            if n <= 1:
                                continue
                            a = ref[:n]
                            seg = Xb[m, lag:lag + n]   # 改名为 seg，避免覆盖 b
                        else:
                            s = -lag
                            n = L - s
                            if n <= 1:
                                continue
                            a = ref[s:s + n]
                            seg = Xb[m, :n]            # 改名为 seg
                        # Pearson numerator/denom
                        a_mean = a.mean()
                        seg_mean = seg.mean()        # 使用 seg_mean
                        a0 = a - a_mean
                        seg0 = seg - seg_mean        # 使用 seg0
                        num = (a0 * seg0).sum()
                        den = torch.sqrt((a0 * a0).sum() * (seg0 * seg0).sum()).clamp(min=eps)
                        corr = (num / den).item() if den > 0 else -2.0
                        if corr > best_corr:
                            best_corr = corr
                            best_lag = lag
                # 调试输出可选
                # print(f'best lag for sample patch {m}: {best_lag}')
                offsets[b, m] = int(best_lag)
                patch = Xb[m]
                lag = int(offsets[b, m].item())

                # 选择填充策略：'reference' 表示用组 reference 的对应片段（前面已有 refs_b）
                channel_idx = m // P
                if gi >= 0:
                    ref_vec = refs_b[gi]  # [L] torch tensor
                else:
                    ref_vec = None
                    ch_mean_val = float(Xb.view(C, P, L).mean(dim=2).mean(dim=1)[channel_idx].item())

                if fill_method == 'reference':
                    # 用组 reference 对齐片段填充
                    if lag > 0:
                        if ref_vec is not None:
                            fill = ref_vec[:lag]
                        else:
                            fill = torch.full((lag,), ch_mean_val, device=device, dtype=patch.dtype)
                        aligned_vec = torch.cat([fill, patch[:L - lag]])
                    elif lag < 0:
                        s = -lag
                        if ref_vec is not None:
                            fill = ref_vec[L - s:]
                        else:
                            fill = torch.full((s,), ch_mean_val, device=device, dtype=patch.dtype)
                        aligned_vec = torch.cat([patch[s:], fill])
                    else:
                        aligned_vec = patch

                elif fill_method == 'interp':
                    aligned_vec = self.interp_fill(patch, lag)

                elif fill_method == 'extrap':
                    aligned_vec = self.extrap_fill(patch, lag)

                else:
                    # 默认退回到 reference 填充
                    if lag > 0:
                        if ref_vec is not None:
                            fill = ref_vec[:lag]
                        else:
                            fill = torch.full((lag,), ch_mean_val, device=device, dtype=patch.dtype)
                        aligned_vec = torch.cat([fill, patch[:L - lag]])
                    elif lag < 0:
                        s = -lag
                        if ref_vec is not None:
                            fill = ref_vec[L - s:]
                        else:
                            fill = torch.full((s,), ch_mean_val, device=device, dtype=patch.dtype)
                        aligned_vec = torch.cat([patch[s:], fill])
                    else:
                        aligned_vec = patch

                aligned[b, m] = aligned_vec
            
            refs_all.append(refs_b)

        return refs_all, offsets, aligned
    
    def group_aggregate(self, aligned, groups_all):
        """
        aligned: Tensor [B, M, L]
        groups_all: list length B, each item is list of groups (group: list of ints in 0..M-1)
        对于每个 patch m：
          out[b,m] = aligned[b,m] + sum_{j in same_group, j!=m} patch_linears[j]( aligned[b,j] )
        """
        B, M, L = aligned.shape
        device = aligned.device
        out = torch.zeros_like(aligned)

        # 确认已按 configs 初始化好 self.patch_linears，数量应等于 M
        if self.patch_linears is None or len(self.patch_linears) != M:
            raise RuntimeError(f"patch_linears size ({None if self.patch_linears is None else len(self.patch_linears)}) != current M ({M}). "
                               "请在 __init__ 中按 configs 初始化 patch_linears 为长度 enc_in*(seq_len//global_patch)。")

        for b in range(B):
            groups = groups_all[b]
            # idx -> group 映射
            idx2grp = [-1] * M
            for gi, grp in enumerate(groups):
                for idx in grp:
                    idx2grp[int(idx)] = gi

            for m in range(M):
                self_vec = aligned[b, m]           # [L]
                gi = idx2grp[m]
                if gi == -1:
                    out[b, m] = self_vec
                    continue

                grp = groups[gi]  # Python list of ints
                # 对组内除自身外的每个 patch 用其各自的线性层变换并求和
                sum_trans = None
                for j in grp:
                    j = int(j)
                    if j == m:
                        continue
                    other_vec = aligned[b, j]   # [L]
                    trans = self.patch_linears[m][j](other_vec)  # [L]
                    if sum_trans is None:
                        sum_trans = trans
                    else:
                        sum_trans = sum_trans + trans
                if sum_trans is None:
                    out[b, m] = self_vec
                else:
                    out[b, m] = self_vec + sum_trans

        return out

    def lag_move(self, x):
        ori_x = x
        # print('x shape before lag move:', x.size())
        x=x.permute(0,2,1).contiguous()
        patches = x.unfold(dimension=-1, size=self.global_patch, step=self.global_patch)  # (B, C, num_patches, patch_size)
        # print('patches shape after lag move:', patches.size())

        B, C, P, L = patches.shape
        if L < 2:
            raise ValueError("patch length L must be >= 2 for correlation")
        # 合并 batch 和 channel 便于向量化： [B*C, P, L]
        X = patches.reshape(B , C*P, L).float()
        # 去均值
        mean = X.mean(dim=-1, keepdim=True)               # [B, C*P, 1]
        Xc = X - mean                                     # [B, C*P, L]
        # 协方差分子：每个样本内部做矩阵乘 -> [B, C*P, C*P]
        num = torch.matmul(Xc, Xc.transpose(1, 2)) / (L - 1)
        # 标准差： [B, C*P]
        std = Xc.std(dim=-1, unbiased=True).clamp(min=1e-8)
        denom = std[:, :, None] * std[:, None, :]         # [B, C*P, C*P]
        corr = num / denom                                # [B, C*P, C*P]
        # 皮尔逊相关系数矩阵 
        # corr=corr.reshape(B, C, P, C*P)
        # print('corr shape:', corr.size())
        patch_groups = self.group_patches(corr, method='threshold', threshold=0.6, min_size=1)
        
        # 基准向量，偏移量，和对齐后的 patch
        refs_all, offsets, aligned = self.compute_group_reference_and_offsets(patches, patch_groups, max_shift = 2, ref_method='medoid', fill_method='interp')

        processed_aligned = self.group_aggregate(aligned, patch_groups)

        self.patch_offsets = offsets  # [B, M]
        self.aligned_patches = processed_aligned.reshape(B, C, P, L).reshape(B,C,P*L)
        # print('aligned patches shape:', self.aligned_patches.size())

        lag_x = self.aligned_patches.permute(0,2,1).contiguous()
        return lag_x
        

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        #B,T,N
        x_enc = self.lag_move(x_enc)
        print('series lag moved')

        x_enc, x_mark_enc = self.__multi_scale_process_inputs(x_enc, x_mark_enc)

        x_list = []
        x_mark_list = []
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)
                x_mark = x_mark.repeat(N, 1, 1)
                x_mark_list.append(x_mark)
        else:
            for i, x in zip(range(len(x_enc)), x_enc, ):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)

        # embedding
        enc_out_list = []
        x_list = self.pre_enc(x_list)
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_list[0])), x_list[0], x_mark_list):
                enc_out = self.enc_embedding(x, x_mark)  # [B,T,C]
                enc_out_list.append(enc_out)
        else:
            for i, x in zip(range(len(x_list[0])), x_list[0]):
                enc_out = self.enc_embedding(x, None)  # [B,T,C]
                enc_out_list.append(enc_out)

        # Past Decomposable Mixing as encoder for past
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        # Future Multipredictor Mixing as decoder for future
        dec_out_list = self.future_multi_mixing(B, enc_out_list, x_list)

        dec_out = torch.stack(dec_out_list, dim=-1).sum(-1)
        dec_out = self.normalize_layers[0](dec_out, 'denorm')
        return dec_out

    def future_multi_mixing(self, B, enc_out_list, x_list):
        dec_out_list = []
        if self.channel_independence:
            x_list = x_list[0]
            for i, enc_out in zip(range(len(x_list)), enc_out_list):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                    0, 2, 1)  # align temporal dimension
                dec_out = self.projection_layer(dec_out)
                dec_out = dec_out.reshape(B, self.configs.c_out, self.pred_len).permute(0, 2, 1).contiguous()
                dec_out_list.append(dec_out)

        else:
            for i, enc_out, out_res in zip(range(len(x_list[0])), enc_out_list, x_list[1]):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                    0, 2, 1)  # align temporal dimension
                dec_out = self.out_projection(dec_out, i, out_res)
                dec_out_list.append(dec_out)

        return dec_out_list

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out_list = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out_list
        return None
