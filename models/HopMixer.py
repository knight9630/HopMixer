import torch
import torch.nn as nn
import numpy as np
import cv2
import torch.fft
from scipy.signal import medfilt2d
from layers.Autoformer_EncDec import series_decomp
from layers.SelfAttention_Family import GapAttention, PatchAttention, SlicedAttention
from layers.Embed import DataEmbedding_wo_pos,DataEmbedding_inverted
from layers.StandardNorm import Normalize

class DFT_series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, top_k=5):
        super(DFT_series_decomp, self).__init__()
        self.top_k = top_k

    def forward(self, x):
        xf = torch.fft.rfft(x)
        freq = abs(xf)
        freq[0] = 0
        top_k_freq, top_list = torch.topk(freq, 5)
        xf[freq <= top_k_freq.min()] = 0
        x_season = torch.fft.irfft(xf)
        x_trend = x - x_season
        return x_season, x_trend

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

        if configs.decomp_method == 'moving_avg':
            self.decompsition = series_decomp(configs.moving_avg)
        elif configs.decomp_method == "dft_decomp":
            self.decompsition = DFT_series_decomp(configs.top_k)
        else:
            raise ValueError('decompsition is error')
        
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
            self.Season_FC = torch.nn.ModuleList(
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
            self.Season_FC = torch.nn.ModuleList(
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
            self.multigapattention = torch.nn.ModuleList(
                [
                    GapAttention(configs.seq_len,configs.global_patch,configs.local_patch)
                    for i in range(configs.down_sampling_layers + 1)
                ]
            )
           
        else:
            self.multigapattention = torch.nn.ModuleList(
                [
                    GapAttention(configs.t_model,configs.global_patch,configs.local_patch)#适应24，8，更改了3的倍数的长度嵌入长度：768
                    for i in range(configs.down_sampling_layers + 1)
                ]
            )

        self.slicedattention = torch.nn.ModuleList(
                [
                    SlicedAttention(configs.t_model,256,32)
                    for i in range(configs.down_sampling_layers + 1)
                ]
            )

        # if configs.channel_independence==1:
        #     self.patchattention = torch.nn.ModuleList(
        #         [
        #             PatchAttention(configs.seq_len,24)
        #             for i in range(configs.down_sampling_layers + 1)
        #         ]
        #     )
        # else:
        #     self.patchattention = torch.nn.ModuleList(
        #         [
        #             PatchAttention(configs.t_model,128)
        #             for i in range(configs.down_sampling_layers + 1)
        #         ]
        #     )
    

    def forward(self, x_list):
        length_list = []
        for x in x_list:
            _, T, _ = x.size()
            length_list.append(T)

        # Decompose to obtain the season and trend
        out_season_list = []
        out_trend_list = []
        index=0
        for x in x_list:
            # print(f'xshape{x.shape}')
            season, trend = self.decompsition(x)
            # print(season.shape)
            season=self.Season_FC[index](season)
            trend=self.multigapattention[index](trend)
            # trend=self.slicedattention[index](trend)
            # patch_trend=self.patchattention[index](trend)
            # trend=sub_trend+patch_trend
            index=index+1

            out_season_list.append(season)
            out_trend_list.append(trend)
    

        out_list = []
        for ori, out_season, out_trend, length in zip(x_list, out_season_list, out_trend_list,
                                                      length_list):
            out = out_season + out_trend
            out = ori + self.out_cross_layer(out)
            # out = ori + self.out_cross_layer(ori)
            out_list.append(out[:, :length, :])
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

    def __multi_scale_process_inputs_avg(self, x_enc, x_mark_enc):
        if self.configs.down_sampling_method == 'max':
            down_pool = torch.nn.MaxPool1d(self.configs.down_sampling_window)
        elif self.configs.down_sampling_method == 'avg':
            #down_pool = torch.nn.AvgPool1d(self.configs.down_sampling_window)
            down_pool=[2,4,8]          

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
            # print(x_enc_ori.shape)
            x_enc_2d = x_enc_ori.cpu().numpy()  # 转换为 (batch_size, length)
            blurred = np.array([cv2.blur(x_i, (down_pool[i],down_pool[i])) for x_i in x_enc_2d])  # 应用均值滤波
            x_enc_sampling = torch.tensor(blurred).float().to(x_enc.device)  # 转换回 Tensor
            # print(x_enc_sampling.shape)

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

    def __multi_scale_process_inputs_mid(self, x_enc, x_mark_enc):
        if self.configs.down_sampling_method == 'max':
            down_pool = torch.nn.MaxPool1d(self.configs.down_sampling_window)
        elif self.configs.down_sampling_method == 'avg':
            #down_pool = torch.nn.AvgPool1d(self.configs.down_sampling_window)
            down_pool=[3,5,7]          

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
            # print(x_enc_ori.shape)
            x_enc_2d = x_enc_ori.cpu().numpy()  # 转换为 (batch_size, length)
            blurred = np.array([medfilt2d(x_i, down_pool[i]) for x_i in x_enc_2d])  # 应用中值滤波
            x_enc_sampling = torch.tensor(blurred).float().to(x_enc.device)  # 转换回 Tensor
            # print(x_enc_sampling.shape)

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

    def __multi_scale_process_inputs_gauss(self, x_enc, x_mark_enc):
        if self.configs.down_sampling_method == 'max':
            down_pool = torch.nn.MaxPool1d(self.configs.down_sampling_window)
        elif self.configs.down_sampling_method == 'avg':
            #down_pool = torch.nn.AvgPool1d(self.configs.down_sampling_window)
            down_pool=[3,5,7]          

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
            # print(x_enc_ori.shape)
            x_enc_2d = x_enc_ori.cpu().numpy()  # 转换为 (batch_size, length)
            blurred = np.array([cv2.GaussianBlur(x_i, (down_pool[i],down_pool[i]),0) for x_i in x_enc_2d])  # 应用高斯滤波
            x_enc_sampling = torch.tensor(blurred).float().to(x_enc.device)  # 转换回 Tensor
            # print(x_enc_sampling.shape)

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

    def __multi_scale_process_inputs_bilateral(self, x_enc, x_mark_enc):
        if self.configs.down_sampling_method == 'max':
            down_pool = torch.nn.MaxPool1d(self.configs.down_sampling_window)
        elif self.configs.down_sampling_method == 'avg':
            #down_pool = torch.nn.AvgPool1d(self.configs.down_sampling_window)
            down_pool=[2,4,8]          

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
            # print(x_enc_ori.shape)
            x_enc_2d = x_enc_ori.cpu().numpy()  # 转换为 (batch_size, length)
            blurred = np.array([cv2.bilateralFilter(x_i, d=down_pool[i], sigmaColor=75, sigmaSpace=75) for x_i in x_enc_2d])  # 应用均值滤波
            x_enc_sampling = torch.tensor(blurred).float().to(x_enc.device)  # 转换回 Tensor
            # print(x_enc_sampling.shape)

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

    def __multi_scale_process_inputs_guide(self, x_enc, x_mark_enc):
        if self.configs.down_sampling_method == 'max':
            down_pool = torch.nn.MaxPool1d(self.configs.down_sampling_window)
        elif self.configs.down_sampling_method == 'avg':
            #down_pool = torch.nn.AvgPool1d(self.configs.down_sampling_window)
            down_pool=[2,4,8]          

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
            # print(x_enc_ori.shape)
            x_enc_2d = x_enc_ori.cpu().numpy()  # 转换为 (batch_size, length)
            blurred = np.array([cv2.ximgproc.createGuidedFilter(guide=x_i, radius=down_pool[i], eps=0.1).filter(x_i) for x_i in x_enc_2d])  # 应用均值滤波
            x_enc_sampling = torch.tensor(blurred).float().to(x_enc.device)  # 转换回 Tensor
            # print(x_enc_sampling.shape)

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

    def __multi_scale_process_inputs_FFT_o(self, x_enc, x_mark_enc):
        # B,T,C -> B,C,T
        x_enc = x_enc.permute(0, 2, 1)

        x_enc_ori = x_enc
        x_mark_enc_mark_ori = x_mark_enc

        x_enc_sampling_list = []
        x_mark_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
        x_mark_sampling_list.append(x_mark_enc)

            # 对 x_enc 做快速傅里叶变换
        xf = torch.fft.rfft(x_enc, dim=-1)  # 在时间维度做 rFFT

        # 获取频率的平均幅值，用于选取主频分量
        freq_amplitude = torch.abs(xf).mean(dim=(0, 1))  # [Freq]

        # 忽略直流分量
        freq_amplitude[0] = 0

        # 找出 top-k 强频率位置
        k = self.configs.down_sampling_layers
        _, top_indices = torch.topk(freq_amplitude, k)

        # 构造掩码，仅保留 top-k 频率，其余置为 0
        device = xf.device
        mask = torch.zeros_like(xf, dtype=torch.bool)
        for idx in top_indices:
            mask[:, :, idx] = True

        # 对每个频率分量做 iFFT 还原回时域信号
        for idx in top_indices:
            filtered_xf = torch.zeros_like(xf)
            filtered_xf[:, :, idx] = xf[:, :, idx]  # 保留单个频率分量
            x_filtered = torch.fft.irfft(filtered_xf, n=x_enc.size(-1), dim=-1)
            x_enc_sampling_list.append(x_filtered.permute(0, 2, 1))  # 转为 [B, T, C]

        for i in range(self.configs.down_sampling_layers):
            if x_mark_enc_mark_ori is not None:
                x_mark_sampling_list.append(x_mark_enc_mark_ori)

        x_enc = x_enc_sampling_list
        if x_mark_enc_mark_ori is not None:
            x_mark_enc = x_mark_sampling_list
        else:
            x_mark_enc = x_mark_enc

        return x_enc, x_mark_enc
    
    def __multi_scale_process_inputs_FFT(self, x_enc, x_mark_enc):
        # B,T,C -> B,C,T
        x_enc = x_enc.permute(0, 2, 1)

        x_enc_ori = x_enc
        x_mark_enc_mark_ori = x_mark_enc

        x_enc_sampling_list = []
        x_mark_sampling_list = []
        # x_enc_sampling_list.append()
        x_mark_sampling_list.append(x_mark_enc)

            # 对 x_enc 做快速傅里叶变换
        xf = torch.fft.rfft(x_enc, dim=-1)  # 在时间维度做 rFFT

        # 获取频率的平均幅值，用于选取主频分量
        freq_amplitude = torch.abs(xf).mean(dim=(0, 1))  # [Freq]

        # 忽略直流分量
        freq_amplitude[0] = 0

        # 找出 top-k 强频率位置
        k = self.configs.down_sampling_layers+1
        _, top_indices = torch.topk(freq_amplitude, k)

        # 构造掩码，仅保留 top-k 频率，其余置为 0
        device = xf.device
        mask = torch.zeros_like(xf, dtype=torch.bool)
        for idx in top_indices:
            mask[:, :, idx] = True

        # 对每个频率分量做 iFFT 还原回时域信号
        for idx in top_indices:
            filtered_xf = torch.zeros_like(xf)
            filtered_xf[:, :, idx] = xf[:, :, idx]  # 保留单个频率分量
            x_filtered = torch.fft.irfft(filtered_xf, n=x_enc.size(-1), dim=-1)
            x_enc_sampling_list.append(x_filtered.permute(0, 2, 1))  # 转为 [B, T, C]

        for i in range(self.configs.down_sampling_layers):
            if x_mark_enc_mark_ori is not None:
                x_mark_sampling_list.append(x_mark_enc_mark_ori)

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