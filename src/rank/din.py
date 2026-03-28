import torch
import numpy as np
from config import *
from src.data.load_data import get_phase_click, get_whole_phase_click
from src.data.convert_data import get_user_item_time_dict
import torch
import torch.nn as nn
import torch.nn.functional as F



torch.manual_seed(42)
np.random.seed(42)


np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


def get_init_item_embed(item_raw_id2_idx_dict, item_content_vec_dict):
    item_cnt = len(item_raw_id2_idx_dict)
    item_embed_np = np.zeros((item_cnt+1, 256), dtype=np.float32)
    for raw_id, idx in item_raw_id2_idx_dict.items():
        vec = item_content_vec_dict[int(raw_id)]
        item_embed_np[idx, :] = vec
    return item_embed_np


def get_init_user_embed(target_phase, user_raw_id2_idx_dict, item_content_vec_dict, is_use_whole_click=True):
    all_click, click_q_time = get_phase_click(target_phase)
    if is_use_whole_click:
        phase_click = get_whole_phase_click(target_phase)
    else:
        phase_click = all_click
    user_item_time_dict = get_user_item_time_dict(phase_click, click_q_time)

    def weighted_agg_content(hist_item_id_list):
        weighted_vec = np.zeros(128*2, dtype=np.float32)
        hist_num = len(hist_item_id_list)
        sum_weight = 0.0
        for loc, (item_id, click_time) in enumerate(hist_item_id_list):
            loc_weight = 0.9 ** (hist_num - loc)
            if item_id in item_content_vec_dict:
                sum_weight += loc_weight
                weighted_vec += loc_weight * item_content_vec_dict[item_id]
        
        if sum_weight != 0:
            weighted_vec /= sum_weight
            txt_norm = np.linalg.norm(weighted_vec[:128]) + 1e-12
            img_norm = np.linalg.norm(weighted_vec[128:]) + 1e-12
            txt_item_feat_np = weighted_vec[:128] / txt_norm
            img_item_feat_np = weighted_vec[128:] / img_norm
            weighted_vec = np.concatenate([txt_item_feat_np, img_item_feat_np]).astype(np.float32)
        else:
            print('zero weight for user, hist_num:', hist_num)
        return weighted_vec

    user_cnt = len(user_raw_id2_idx_dict)
    user_embed_np = np.zeros((user_cnt+1, 256), dtype=np.float32)
    for raw_id, idx in user_raw_id2_idx_dict.items():
        if int(raw_id) in user_item_time_dict:
            hist = user_item_time_dict[int(raw_id)]
            user_embed_np[idx, :] = weighted_agg_content(hist)
    return user_embed_np


class Dice(nn.Module):
    def __init__(self, input_dim, eps=1e-9):
        super(Dice, self).__init__()
        self.bn = nn.BatchNorm1d(input_dim, eps=eps)
        self.alpha = nn.Parameter(torch.zeros(input_dim))
    
    def forward(self, x):
        if x.dim() == 2:
            normed = self.bn(x)
            prob = torch.sigmoid(normed)
            return prob * x + (1-prob) * self.alpha * x
        
        shape = x.shape
        x_flat = x.reshape(-1, shape[-1])
        normed = self.bn(x_flat).reshape(shape)
        prob = torch.sigmoid(normed)
        return prob * x + (1-prob) * self.alpha.view(1, 1, -1) * x