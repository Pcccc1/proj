import torch
import numpy as np
from config import *
from src.data.load_data import get_phase_click, get_whole_phase_click
from src.data.convert_data import get_user_item_time_dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader



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
    

def get_activation(activation, hidden_size=None):
    name = activation.lower()
    if name == 'relu':
        return nn.ReLU()
    if name == 'prelu':
        return nn.PReLU()
    if name == 'sigmoid':
        return nn.Sigmoid()
    if name == 'Dice':
        return Dice(hidden_size)
    raise ValueError(f"Unsupported activation: {activation}")


class DNNBlock(nn.Module):
    def __init__(self, input_dim, hidden_units, activation='relu', use_bn=False, dropout_rate=0.0):
        super(DNNBlock, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_units:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(get_activation(activation, hidden_dim))
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        self.dnn = nn.Sequential(*layers)
    
    def forward(self, x):
        if len(self.dnn) == 0:
            return x
        return self.dnn(x)
        

class LocalActivationUnit(nn.Module):
    def __init__(self, input_dim, att_hidden_size=(80, 40), att_activation='dice'):
        super(LocalActivationUnit, self).__init__()
        self.att_dnn = DNNBlock(
            input_dim=input_dim * 4,
            hidden_units=att_hidden_size,
            activation=att_activation,
            use_bn=False,
            dropout_rate=0.0
        )
        final_dim = att_hidden_size[-1] if len(att_hidden_size) > 0 else input_dim * 4
        self.att_score = nn.Linear(final_dim, 1)
    
    def forward(self, query_emb, keys_emb):
        # q[B, 1, E] k[B, T, E]
        query_expand = query_emb.expand(-1, keys_emb.size(1), -1)
        att_input = torch.cat(
            [query_expand, keys_emb, query_expand - keys_emb, query_expand * keys_emb], dim=-1
        )
        att_hidden = self.att_dnn(att_input)
        score = self.att_score(att_hidden).squeeze(-1)
        return score


class DINNet(nn.Module):
    def __init__(
        self,
        dnn_feature_columns,
        history_feature_list,
        user_init_embedding=None,
        item_init_embedding=None,
        dnn_use_bn=False,
        dnn_hidden_units=(200, 80),
        dnn_activation='dice',
        att_hidden_size=(80, 40),
        att_weight_normlization=False,
        dnn_dropout_rate=0.0,
        init_std=0.001
    ):
        super(DINNet, self).__init__()

        self.sparse_features = list(dnn_feature_columns.get('sparse_features', []))
        self.dense_features = list(dnn_feature_columns.get('dense_features', []))
        self.varlen_sparse_features = list(dnn_feature_columns.get('varlen_sparse_features', []))
        self.embedding_dim_map = dict(dnn_feature_columns.get('embedding_dim_map', {}))
        self.embedding_name_map = dict(dnn_feature_columns.get('embedding_name_map', {}))
        self.history_feature_names = [f'hist_{feat}' for feat in history_feature_list]
        self.non_history_varlen_features = [
            feat for feat in self.varlen_sparse_features if feat not in self.history_feature_names
        ]
        self.sequence_length_name_map = dict(dnn_feature_columns.get('sequence_length_name_map', {}))
        self.att_weight_normlization = att_weight_normlization

        vocab_sizes = dict(dnn_feature_columns.get('vocab_sizes', {}))

        for feat in self.sparse_features:
            if feat not in self.embedding_dim_map:
                self.embedding_dim_map[feat] = feat
            
        for seq_feat in self.varlen_sparse_features:
            if seq_feat not in self.embedding_dim_map:
                if seq_feat.startwith('hist_'):
                    self.embedding_dim_map[seq_feat] = seq_feat[5:]
                else:
                    self .embedding_dim_map[seq_feat] = seq_feat
        
        base_embedding_features = set(self.embedding_dim_map.values())

        init_embedding_map = {}
        if user_init_embedding is not None:
            init_embedding_map['user_id'] = user_init_embedding
        
        if item_init_embedding is not None:
            init_embedding_map['item_id'] = item_init_embedding
        
        self.embedding_dict = nn.ModuleDict()
        for base_feat in base_embedding_features:
            if base_feat not in vocab_sizes:
                raise ValueError(f"Vocab size for feature '{base_feat}' is not specified in dnn_feature_columns['vocab_sizes']")
            embed_dim = self.embedding_dim_map.get(base_feat, EMBED_DIM)
            init_weight = init_embedding_map.get(base_feat, None)
            self.embedding_dict[base_feat] = self._build_embedding(
                vocab_size=vocab_sizes[base_feat],
                embed_dim=embed_dim,
                init_std=init_std,
                init_weight=init_weight
            )
        self.history_query_features = list(history_feature_list)
        if len(history_feature_list) == 0:
            raise ValueError("history_feature_list cannot be empty")
        
        history_input_dim = 0
        for feat in self.history_query_features:
            base_feat = self.embedding_name_map.get(feat, feat)
            history_input_dim += self.embedding_dim_map.get(base_feat, EMBED_DIM)
        
        self.attention_unit = LocalActivationUnit(
            input_dim=history_input_dim,
            att_hidden_size=att_hidden_size,
            att_activation=att_hidden_size
        )
        sparse_emb_dim = 0
        for feat in self.sparse_features:
            base_feat = self.embedding_name_map[feat]
            sparse_emb_dim += self.embedding_dim_map.get(base_feat, EMBED_DIM)
        
        varlen_pool_dim = 0
        for feat in self.non_history_varlen_features:
            base_feat = self.embedding_name_map[feat]
            varlen_pool_dim += self.embedding_dim_map.get(base_feat, EMBED_DIM)

        dnn_input_dim = sparse_emb_dim + varlen_pool_dim + history_input_dim
        self.dnn = DNNBlock(
            input_dim=dnn_input_dim,
            hidden_units=dnn_hidden_units,
            activation=dnn_activation,
            use_bn=dnn_use_bn,
            dropout_rate=dnn_dropout_rate
        )
        final_dim = dnn_hidden_units[-1] if len(dnn_hidden_units) > 0 else dnn_input_dim
        self.fc = nn.Linear(final_dim, 1, bias=False)


    @staticmethod
    def _build_embedding(vocab_size, embed_dim, init_std, init_weight=None):
        if init_weight is not None:
            return nn.Embedding.from_pretrained(
                torch.tensor(init_weight, dtype=torch.float32),
                freeze=False,
                padding_idx=0
            )
        emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        nn.init.normal_(emb.weight, mean=0.0, std=init_std)
        with torch.no_grad():
            emb.weight[0].fill_(0.0)
        return emb
    
    def _embed_sparse_feature(self, x, feat):
        base_feat = self.embedding_name_map[feat]
        return self.embedding_dict[base_feat](x[feat].long())
    
    def _get_sequence_length(self, x, seq_feat, seq_tensor):
        length_name = self.sequence_length_name_map.get(seq_feat, f'{seq_feat}_length')
        if length_name in x:
            return x[length_name].long()
        return (seq_tensor != 0).sum(dim=1)
    
    @staticmethod
    def _build_mask_from_length(length_tensor, max_len):
        arange_tensor = torch.arange(max_len, device=length_tensor.device).unsqueeze(0)
        return arange_tensor < length_tensor.unsqueeze(1)
    

    def _attention_pooling(self, query_emb, keys_emb, keys_length):
        score = self.attention_unit(query_emb, keys_emb)
        keys_mask = self._build_mask_from_length(keys_length, keys_emb.size(1))
        mask_float = keys_mask.float()

        if self.att_weight_normlization:
            score = score - score.max(dim=1, keepdim=True)[0]
            exp_score = torch.exp(score) * mask_float
            score_sum = exp_score.sum(dim=1, keepdim=True) + 1e-8
            att_weight = exp_score / score_sum
        else:
            att_weight = torch.sigmoid(score) * mask_float
        return torch.bmm(att_weight.unsqueeze(1), keys_emb).squeeze(1)
    
    
    def _varlen_mean_pooling(self, seq_emb, seq_length):
        mask = self._build_mask_from_length(seq_length, seq_emb.size(1)).unsqueeze(-1).float()
        seq_sum = (seq_emb * mask).sum(dim=1)
        denom = seq_length.float().clamp(min=1.0).unsqueeze(1)
        return seq_sum / denom
    
    def regularization_loss(self, l2_reg_dnn=0.0, l2_reg_embedding=0.0):
        reg_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if 'embedding_dict' in name:
                reg_loss = reg_loss + l2_reg_embedding * torch.sum(param * param)
            else:
                reg_loss = reg_loss + l2_reg_dnn * torch.sum(param * param)
        return reg_loss

    def forward(self, x):
        sparse_emb_list = [self._embed_sparse_feature(x, feat) for feat in self.sparse_features]

        query_emb_list = []
        keys_emb_list = []
        for query_feat, hist_feat in zip(self.history_query_features, self.history_feature_names):
            if query_feat not in x:
                raise ValueError(f"Query feature '{query_feat}' is missing in input")
            if hist_feat not in x:
                raise ValueError(f"History feature '{hist_feat}' is missing in input")
            
            query_emb = self._embed_sparse_feature(x, query_feat).unsqueeze(1)
            seq_ids = x[hist_feat].long()
            keys_emb = self._embed_sparse_feature(x, hist_feat)

            query_emb_list.append(query_emb)
            keys_emb_list.append(keys_emb)
            if 'history_length' not in x:
                x['history_length'] = self._get_sequence_length(x, hist_feat, seq_ids)
            
        query_emb = torch.cat(query_emb_list, dim=-1)
        keys_emb = torch.cat(keys_emb_list, dim=-1)
        hist_emb = self._attention_pooling(query_emb, keys_emb, x['history_length'])

        non_hist_varlen_pooled = []
        for seq_feat in self.non_history_varlen_features:
            if seq_feat not in x:
                continue
            seq_ids = x[seq_feat].long()
            seq_emb = self._embed_sparse_feature(x, seq_feat)
            seq_length = self._get_sequence_length(x, seq_feat, seq_ids)
            pooled = self.varlen_sparse_features(seq_emb, seq_length)
            non_hist_varlen_pooled.append(pooled)
        
        emb_input_list = sparse_emb_list + non_hist_varlen_pooled + [hist_emb]
        emb_input = torch.cat(emb_input_list, dim=1)

        dense_input_list = []
        for feat in self.dense_features:
            dense_value = x[feat].float()
            if dense_value.dim() == 1:
                dense_value = dense_value.unsqueeze(1)
            if dense_value.dim() > 2:
                dense_value = dense_value.reshape(dense_value.size(0), -1)
            dense_input_list.append(dense_value)
        
        if len(dense_input_list) > 0:
            dense_input = torch.cat(dense_input_list, dim=1)
            dnn_input = torch.cat([emb_input, dense_input], dim=1)
        else:
            dnn_input = emb_input
        dnn_out = self.dnn(dnn_input)
        return self.fc(dnn_out).squeeze(1)


class DictFeatureDataset(Dataset):
    def __init__(self, features, long_features_name, labels=None):
        self.features = {}
        self.long_features_name = set(long_features_name)
        self.labels = None if labels is None else np.asarray(labels, dtype=np.float32)

        for name, value in features.items():
            raw = value.tolist() if hasattr(value, 'tolist') else list(value)
            arr = np.asarray(raw)
            if arr.dtype == object:
                try:
                    arr = np.asarray(raw, dtype=np.int64)
                except Exception:
                    arr = np.asarray(raw, dtype=np.float32)
            self.features[name] = arr
        
        self.length = len(next(iter(self.features.values())))

    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        sample = {}
        for name, arr in self.features.items():
            val = arr[idx]
            if name in self.long_features_name:
                sample[name] = torch.as_tensor(val, dtype=torch.long)
            else:
                sample[name] = torch.as_tensor(val, dtype=torch.float32)
        
        if self.labels is not None:
            return sample
        else:
            return sample, torch.as_tensor(self.labels[idx], dtype=torch.float32)
        