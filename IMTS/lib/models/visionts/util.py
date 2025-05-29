import inspect
import os
from typing import Optional, Callable
import requests

import pandas as pd
import numpy as np
from torchvision.transforms import Resize
from tqdm import tqdm
import torch
import einops
import torch.nn.functional as F
import torch.nn as nn

def download_file(url, local_filename):
    response = requests.get(url, stream=True)
    os.makedirs(os.path.dirname(local_filename), exist_ok=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(local_filename, 'wb') as file:
        with tqdm(
            desc=f"Download: {local_filename}",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
            dynamic_ncols=True
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
                    bar.update(len(chunk))


def safe_resize(size, interpolation):
    signature = inspect.signature(Resize)
    params = signature.parameters
    if 'antialias' in params:
        return Resize(size, interpolation, antialias=False)
    else:
        return Resize(size, interpolation)


POSSIBLE_SEASONALITIES = {
    "S": [3600],  # 1 hour
    "T": [1440, 10080],  # 1 day or 1 week
    "H": [24, 168],  # 1 day or 1 week
    "D": [7, 30, 365],  # 1 week, 1 month or 1 year
    "W": [52, 4], # 1 year or 1 month
    "M": [12, 6, 3], # 3 months, 6 months or 1 year
    "B": [5],
    "Q": [4, 2], # 6 months or 1 year
}


def norm_freq_str(freq_str: str) -> str:
    base_freq = freq_str.split("-")[0]
    if len(base_freq) >= 2 and base_freq.endswith("S"):
        return base_freq[:-1]
    return base_freq


def freq_to_seasonality_list(freq: str, mapping_dict=None) -> int:
    if mapping_dict is None:
        mapping_dict = POSSIBLE_SEASONALITIES
    offset = pd.tseries.frequencies.to_offset(freq)
    base_seasonality_list = mapping_dict.get(norm_freq_str(offset.name), [])
    seasonality_list = []
    for base_seasonality in base_seasonality_list:
        seasonality, remainder = divmod(base_seasonality, offset.n)
        if not remainder:
            seasonality_list.append(seasonality)
    seasonality_list.append(1) # we append P=1 for those without significant periodicity
    return seasonality_list


def get_prediction_from_reconstruction(y, tp_to_predict, start_tp, tp_predict_mask, tt_mask):
    # y: [B, L, D]
    # tp_to_predict: [B, T]
    B, L, D = y.shape
    min = start_tp
    max = (start_tp+1)
    scale = max - min
    mask = tp_predict_mask.max(dim=-1).values.long()
    index_predict = torch.clamp(torch.round(((tp_to_predict - min[:,None]) / scale[:,None] * L ) * mask),0,L-1).long()
    enable = ((tt_mask.sum(dim=1).sum(dim=1) > 0) & (tp_predict_mask.sum(dim=1).sum(dim=1) > 0)).long()
    index_predict = index_predict * enable[:,None]
    return y[torch.arange(B)[:,None].to(y.device), index_predict, :]


def grid_time_interpolation(tt, value, resolution_len:int, dataset_len=48, method="zero", tt_mask=None):
    '''
    tt: time series \\ [B, T]
    value: value series \\ [B, T, D]
    time_len: time length of the context window \\
    resolution_len: time resolution of the context \\
    method: interpolation method ['zero', 'linear']
    dataset_len: time length of the dataset, e.g. 48 months
    Explain: 将不规则采样的时间序列插值到固定的时间间隔上
    return: [B, D, context_len, resolution_len]
    '''
    device = tt.device
    tt_type = tt.dtype
    value_type = value.dtype
    tt = tt.cpu().numpy()      # [B, T]
    value = value.cpu().numpy()  # [B, T, D]
    
    
    B, T = tt.shape
    D = value.shape[2]
    
    # 确定每个批次的时间范围 0~1, 1/48为一个月
    unit_len = 1/dataset_len
    start_unit = np.floor(tt[:,0]/unit_len)  # [B]
    end_unit = np.floor(np.max(tt,axis=1)/unit_len)  # [B]
    
    num_periods = ((end_unit - start_unit) + 1).astype(int)  # [B]
    lengths = num_periods * resolution_len  # [B], integer
    max_length = np.max(lengths)      # scalar
    dt = 1 / resolution_len
    
    # 创建网格化的时间轴 new_tt [B, max_length]
    idx = np.arange(max_length)  # [max_length]
    new_tt = (idx[None, :] * dt + start_unit[:, None]) * unit_len  # [B, max_length]
    
    # # 创建一个掩码，指示每个批次的有效位置
    # mask = idx[None, :] < lengths[:, None]  # [B, max_length]
    # new_tt = np.where(mask, new_tt, 0)  # 可选：将无效位置设置为0或np.nan
    
    if method == "zero":
        # 初始化新的值向量 [B, max_length, D]
        new_value = np.zeros((B, max_length, D))
        
        # 计算每个 tt 对应的索引
        idx_assign = np.floor((tt - start_unit[:, None] * unit_len) * dataset_len * resolution_len + 0.5).astype(int)  # [B, T]
        
        # 确保 idx_assign 不超过每个批次的有效长度
        idx_assign = np.clip(idx_assign, 0, lengths[:, None] - 1)
        
        
        # 使用高级索引进行并行赋值
        new_value[np.arange(B)[:, None], idx_assign, :] = value
    
    elif method == "linear":
        
        idx_assign = np.floor((tt - start_unit[:, None] * unit_len) * dataset_len * resolution_len + 0.5).astype(int)  # [B, T]

        # 确保 idx_assign 不超过每个批次的有效长度
        idx_assign = np.clip(idx_assign, 0, lengths[:, None] - 1)

        # 初始化新的值向量 [B, max_length, D]
        new_value = np.zeros((B, max_length, D))

        # 使用高级索引进行并行赋值
        new_value[np.arange(B)[:, None], idx_assign, :] = value

        tt_mask = tt_mask.cpu().numpy() if tt_mask is not None else None
        # 为有效的 new_tt 之间的 new_value 进行线性插值
        for b in range(B):
            for d in range(D):
                valid_indices = idx_assign[b, tt_mask[b, :, d] != 0]
                valid_values = value[b, tt_mask[b, :, d] != 0, d]
                
                if len(valid_indices) == 0:
                    continue
                # 插值
                new_value[b, :, d] = torch.tensor(np.interp(np.arange(max_length), valid_indices, valid_values, left=0, right=0))

    else:
        raise ValueError("Unsupported interpolation method. Choose 'zero' or 'linear'.")
    
    # 转换回 PyTorch 张量
    new_tt = torch.tensor(new_tt, device=device, dtype=tt_type)
    new_value = torch.tensor(new_value, device=device, dtype=value_type) # [B, max_length, D]

    # leftpad
    new_value = einops.rearrange(new_value, 'b l n-> b n l') # [B, D, max_length]

    return new_value, new_tt


class IMTS_PatchEmbed(nn.Module):
    """ 
    Patch_feature 2 Patch Embedding
    [B*N,M,C] -> [B*N,M,768]
    """

    def __init__(
            self,
            # img_size: Optional[int] = 224,
            # patch_size: int = 16,
            input_dim: int = 32, # arg.hid_dim
            embed_dim: int = 768, # output_dim
            norm_layer: Optional[Callable] = None,
            bias: bool = True,
    ):
        super().__init__()


        # from time featrue to token
        self.proj = nn.Linear(in_features=input_dim, out_features=embed_dim, bias=bias)

        # layernorm
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()


    def forward(self, x):
        BN, M, C = x.shape
        
        # time to token
        x = self.proj(x)

        # normalization 
        x = self.norm(x)

        return x