import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt

from einops import rearrange, repeat

from utils.masking import TriangularCausalMask, ProbMask
from reformer_pytorch import LSHSelfAttention

class GapAttention(nn.Module):
    def __init__(self,input_dim,daytime,gapdis):
        super(GapAttention, self).__init__()
        self.input_dim=input_dim
        self.daytime=daytime
        self.gapdis=gapdis
        self.multiattention_layers = torch.nn.ModuleList(
                [
                    nn.MultiheadAttention(embed_dim=self.gapdis, num_heads=1, batch_first=True)
                    for i in range(self.daytime//self.gapdis)
                ]
            )
    def forward(self,x):
        _,_,N=x.size()
        x_pre=x
        x=x.permute(0,2,1)
        x_days=[]
        for i in range(self.input_dim//self.daytime):
            x_days.append(x[:,:,i*self.daytime:(i+1)*self.daytime])

        x_gaps=[]
        for j in range(self.daytime//self.gapdis):
            x_gap=torch.cat([x_day[:,:,j*self.gapdis:(j+1)*self.gapdis] for x_day in x_days],dim=1)
            x_gaps.append(x_gap)

        
        for k in range(self.daytime//self.gapdis):
            x_gaps[k],_=self.multiattention_layers[k](x_gaps[k],x_gaps[k],x_gaps[k])
            for m in range(self.input_dim//self.daytime):
                x[:,:,m*self.daytime+k*self.gapdis:m*self.daytime+(k+1)*self.gapdis]=x_gaps[k][:,m*N:(m+1)*N,:]
        x=x.permute(0,2,1)
        return x

class SlicedAttention(nn.Module):
    def __init__(self, input_dim, daytime, gapdis):
        super(SlicedAttention, self).__init__()
        self.input_dim = input_dim
        self.daytime = daytime
        self.gapdis = gapdis

        # 构造每段内部使用的注意力层（一个 attention layer 用于每个 gap）
        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=gapdis, num_heads=1)
            for _ in range(daytime // gapdis)
        ])

    def forward(self, x):
        batch_size, input_dim, N = x.size()
        segment_len = 50

        num_segments = N // segment_len
        num_reminder = N % segment_len

        segments = [x[:, :, i * segment_len : (i + 1) * segment_len] for i in range(num_segments)]
        
        if num_reminder>0:
            reminder_segment = x[:, :, num_segments * segment_len :]
            segments.append(reminder_segment)

        processed_segments = []

        for segment in segments:
            # (B, input_dim, 25) -> (B, 25, input_dim)
            _ , _ , segment_N = segment.size()
            seg = segment.permute(0, 2, 1)

            # 按 day 切分
            x_days = [
                seg[:, :, i * self.daytime:(i + 1) * self.daytime]
                for i in range(self.input_dim // self.daytime)
            ]

            # 按 gapdis 切每个 day 同位置，拼接成 gap 输入
            x_gaps = []
            for j in range(self.daytime // self.gapdis):
                x_gap = torch.cat(
                    [x_day[:, :, j * self.gapdis:(j + 1) * self.gapdis] for x_day in x_days],
                    dim=1  # 拼接维度是“天”数上的 batch
                )  # (B, 25 * 天数, gapdis)
                x_gaps.append(x_gap)

            # 应用注意力
            for k in range(self.daytime // self.gapdis):
                x_gaps[k], _ = self.attn_layers[k](x_gaps[k], x_gaps[k], x_gaps[k])
                for m in range(self.input_dim // self.daytime):
                    seg[:, :, m * self.daytime + k * self.gapdis : m * self.daytime + (k + 1) * self.gapdis] = \
                        x_gaps[k][:, m * segment_N : (m + 1) * segment_N, :]

            # 转回原格式 (B, input_dim, 25)
            seg = seg.permute(0, 2, 1)
            processed_segments.append(seg)

        # 拼接所有段 (B, input_dim, 325)
        x = torch.cat(processed_segments, dim=2)
        return x

class PatchAttention(nn.Module):
    def __init__(self,input_dim,daytime):
        super(PatchAttention, self).__init__()
        self.input_dim=input_dim
        self.daytime=daytime
        self.patch_attention = nn.MultiheadAttention(embed_dim=self.daytime, num_heads=1)

    def forward(self,x):
        b,_,N=x.size()
        x_pre=x
        x=x.permute(0,2,1)
        # 检查 input_dim 是否可以被 daytime 整除
        assert self.input_dim % self.daytime == 0, "input_dim 必须是 daytime 的整数倍"

        # 计算分组的数量
        num_groups = self.input_dim // self.daytime

        # 调整 x 的维度以便切分和拼接
        # 先 reshape 为 (b, N, num_groups, daytime)
        patches = x.view(b, N, num_groups, self.daytime)

        # 调整维度为 (b, N * num_groups, daytime)
        aggregate_patches = patches.permute(0, 2, 1, 3).reshape(b, N * num_groups, self.daytime)
        # print(f'aggreted_shape{aggregate_patches.shape}')
        pro_patches,_=self.patch_attention(aggregate_patches,aggregate_patches,aggregate_patches)

        # 恢复 x 的维度到 (b, N, input_dim)
        pro_patches = pro_patches.view(b, num_groups, N, self.daytime).permute(0, 2, 1, 3)  # (b, N, num_groups, daytime)
        pro_patches = pro_patches.reshape(b, N, self.input_dim)  # (b, N, input_dim)
        x=pro_patches.permute(0,2,1)
        return x
        
            



class DSAttention(nn.Module):
    '''De-stationary Attention'''

    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(DSAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        tau = 1.0 if tau is None else tau.unsqueeze(
            1).unsqueeze(1)  # B x 1 x 1 x 1
        delta = 0.0 if delta is None else delta.unsqueeze(
            1).unsqueeze(1)  # B x 1 x 1 x S

        # De-stationary Attention, rescaling pre-softmax score with learned de-stationary factors
        scores = torch.einsum("blhe,bshe->bhls", queries, keys) * tau + delta

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        # real U = U_part(factor*ln(L_k))*L_q
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(
            L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(
            Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H,
                                                L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            # requires that L_Q == L_V, i.e. for self-attention only
            assert (L_Q == L_V)
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) /
                     L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[
                                                  None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * \
                 np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * \
            np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(
            queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(
            context, values, scores_top, index, L_Q, attn_mask)

        return context.contiguous(), attn

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn

class ReformerLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None, causal=False, bucket_size=4, n_hashes=4):
        super().__init__()
        self.bucket_size = bucket_size
        self.attn = LSHSelfAttention(
            dim=d_model,
            heads=n_heads,
            bucket_size=bucket_size,
            n_hashes=n_hashes,
            causal=causal
        )

    def fit_length(self, queries):
        # inside reformer: assert N % (bucket_size * 2) == 0
        B, N, C = queries.shape
        if N % (self.bucket_size * 2) == 0:
            return queries
        else:
            # fill the time series
            fill_len = (self.bucket_size * 2) - (N % (self.bucket_size * 2))
            return torch.cat([queries, torch.zeros([B, fill_len, C]).to(queries.device)], dim=1)

    def forward(self, queries, keys, values, attn_mask, tau, delta):
        # in Reformer: defalut queries=keys
        B, N, C = queries.shape
        queries = self.attn(self.fit_length(queries))[:, :N, :]
        return queries, None

class TwoStageAttentionLayer(nn.Module):
    '''
    The Two Stage Attention (TSA) Layer
    input/output shape: [batch_size, Data_dim(D), Seg_num(L), d_model]
    '''

    def __init__(self, configs,
                 seg_num, factor, d_model, n_heads, d_ff=None, dropout=0.1):
        super(TwoStageAttentionLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.time_attention = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                           output_attention=configs.output_attention), d_model, n_heads)
        self.dim_sender = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                       output_attention=configs.output_attention), d_model, n_heads)
        self.dim_receiver = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                         output_attention=configs.output_attention), d_model, n_heads)
        self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))

        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # Cross Time Stage: Directly apply MSA to each dimension
        batch = x.shape[0]
        time_in = rearrange(x, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model')
        time_enc, attn = self.time_attention(
            time_in, time_in, time_in, attn_mask=None, tau=None, delta=None
        )
        dim_in = time_in + self.dropout(time_enc)
        dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in))
        dim_in = self.norm2(dim_in)

        # Cross Dimension Stage: use a small set of learnable vectors to aggregate and distribute messages to build the D-to-D connection
        dim_send = rearrange(dim_in, '(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model', b=batch)
        batch_router = repeat(self.router, 'seg_num factor d_model -> (repeat seg_num) factor d_model', repeat=batch)
        dim_buffer, attn = self.dim_sender(batch_router, dim_send, dim_send, attn_mask=None, tau=None, delta=None)
        dim_receive, attn = self.dim_receiver(dim_send, dim_buffer, dim_buffer, attn_mask=None, tau=None, delta=None)
        dim_enc = dim_send + self.dropout(dim_receive)
        dim_enc = self.norm3(dim_enc)
        dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
        dim_enc = self.norm4(dim_enc)

        final_out = rearrange(dim_enc, '(b seg_num) ts_d d_model -> b ts_d seg_num d_model', b=batch)

        return final_out
