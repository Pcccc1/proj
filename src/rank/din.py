import torch
import numpy as np
import inspect

from deepctr_torch.inputs import SparseFeat, DenseFeat, VarLenSparseFeat, get_feature_names
from deepctr_torch.models import DIN
from config import *

from src.data.load_data import get_phase_click, get_whole_phase_click
from src.data.convert_data import get_user_item_time_dict

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


def get_init_item_embed(item_raw_id2_idx_dict, item_content_vec_dict):
    global item_embed_np
    item_cnt = len(item_raw_id2_idx_dict)
    item_embed_np = np.zeros((item_cnt + 1, 256), dtype=np.float32)
    for raw_id, idx in item_raw_id2_idx_dict.items():
        vec = item_content_vec_dict[int(raw_id)]
        item_embed_np[idx, :] = vec
    return item_embed_np

def get_init_user_embed(target_phase, user_raw_id2_idx_dict, item_content_vec_dict, is_use_whole_click=True):
    global user_embed_np
    all_click, click_q_time = get_phase_click(target_phase)
    if is_use_whole_click:
        phase_click = get_whole_phase_click(all_click, click_q_time)
    else:
        phase_click = all_click

    user_item_time_hist_dict = get_user_item_time_dict(phase_click)

    def weighted_agg_content(hist_item_id_list):
        weighted_vec = np.zeros(128 * 2, dtype=np.float32)
        hist_num = len(hist_item_id_list)
        sum_weight = 0.0
        for loc, (item_id, click_t) in enumerate(hist_item_id_list):
            loc_weight = 0.9 ** (hist_num - loc)
            if item_id in item_content_vec_dict:
                sum_weight += loc_weight
                weighted_vec += loc_weight * item_content_vec_dict[item_id]

        if sum_weight != 0:
            weighted_vec /= sum_weight
            txt_item_feat_np = weighted_vec[0:128] / np.linalg.norm(weighted_vec[0:128])
            img_item_feat_np = weighted_vec[128:] / np.linalg.norm(weighted_vec[128:])
            weighted_vec = np.concatenate([txt_item_feat_np, img_item_feat_np]).astype(np.float32)
        else:
            print('zero weight...')
        return weighted_vec

    user_cnt = len(user_raw_id2_idx_dict)
    user_embed_np = np.zeros((user_cnt + 1, 256), dtype=np.float32)
    for raw_id, idx in user_raw_id2_idx_dict.items():
        if int(raw_id) in user_item_time_hist_dict:
            hist = user_item_time_hist_dict[int(raw_id)]
            user_embed_np[idx, :] = weighted_agg_content(hist)
    return user_embed_np


def _load_pretrained_embedding(model, embedding_name, pretrained_weight):
    if pretrained_weight is None:
        return
    if not hasattr(model, 'embedding_dict'):
        return
    if embedding_name not in model.embedding_dict:
        return
    
    emb_layer = model.embedding_dict[embedding_name]
    expect_shape = (emb_layer.num_embeddings, emb_layer.embedding_dim)
    if pretrained_weight.shape != expect_shape:
        print(
            'Pretrained weight shape {} does not match expected shape {} for embedding layer {}'.format(
                pretrained_weight.shape, expect_shape, embedding_name
            )
        )
        return

    with torch.no_grad():
        emb_layer.weight.data.copy_(torch.from_numpy(pretrained_weight))


def KDD_DIN(
    dnn_feature_columns,
    history_feature_list,
    dnn_use_bn=False,
    dnn_hidden_units=(200, 80),
    dnn_activation='relu',
    att_hidden_size=(90, 40),
    att_activation='Dice',
    att_weight_normalization=False,
    l2_reg_dnn=0,
    l2_reg_embedding=1e-6,
    dnn_dropout=0,
    init_std=0.0001,
    seed=42,
    task='binary',
    device='cpu',
    user_init_embedding=None,
    item_init_embedding=None,
):
    din_kwargs = {
        'dnn_feature_columns' : dnn_feature_columns,
        'history_feature_list' : history_feature_list,
        'dnn_use_bn' : dnn_use_bn,
        'dnn_hidden_units' : dnn_hidden_units,
        'dnn_activation' : dnn_activation,
        'att_activation' : att_activation,
        'att_weight_normalization' : att_weight_normalization,
        'l2_reg_dnn' : l2_reg_dnn,
        'l2_reg_embedding' : l2_reg_embedding,
        'dnn_dropout' : dnn_dropout,
        'init_std' : init_std,
        'seed' : seed,
        'task' : task,
        'device' : device,
    }
    din_init_params = inspect.signature(DIN.__init__).parameters
    if 'att_hidden_size' in din_init_params:
        din_kwargs['att_hidden_size'] = att_hidden_size
    elif 'att_hidden_units' in din_init_params:
        din_kwargs['att_hidden_units'] = att_hidden_size
    
    valid_kwargs = {k : v for k, v in din_kwargs.items() if k in din_init_params}
    model = DIN(**valid_kwargs)

    _load_pretrained_embedding(model, 'user_id', user_init_embedding)
    _load_pretrained_embedding(model, 'item_id', item_init_embedding)
    return model

HIDDEN_SIZE = (128, 128)
BATCH_SIZE = 1024
EPOCH = 1
EMBED_DIM = 256
TIME_EMBED_DIM = 16

def build_din_input(df, feature_names, seq_feature_name='hist_item_id', seq_length_name='hist_item_id_length'):
    model_input = {}
    for name in feature_names:
        if name in df.columns:
            model_input[name] = np.array(df[name].values.tolist())
            continue
    
        if name == seq_length_name:
            if seq_feature_name not in df.columns:
                raise ValueError(f"Sequence feature '{seq_feature_name}' not found in DataFrame columns.")
            seq_np = np.array(df[seq_feature_name].values.tolist())
            model_input[name] = np.sum(seq_np != 0, axis=1).astype(np.int32)
            continue
        
        raise ValueError(f"Feature '{name}' not found in DataFrame columns.")
    return model_input
        

def generate_din_feature_columns(sparse_features, dense_features, feat_lbe_dict):
    sparse_feature_columns = [
        SparseFeat(feat, vocabulary_size=len(feat_lbe_dict[feat].classes_) + 1, embedding_dim=EMBED_DIM) for feat in sparse_features if feat not in time_feat
    ]
    
    dense_feature_columns = [DenseFeat(feat, 1) for feat in dense_features]

    var_feature_columns = [
        VarLenSparseFeat(
            SparseFeat(
                'hist_item_id',
                vocabulary_size=len(feat_lbe_dict['item_id'].classes_) + 1,
                embedding_dim=EMBED_DIM,
                embedding_name='item_id'
            ),
            maxlen=10,
            length_name='hist_item_id_length',
        )
    ]
    dnn_feature_columns = sparse_feature_columns + dense_feature_columns + var_feature_columns
    linear_feature_columns = sparse_feature_columns + dense_feature_columns + var_feature_columns
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
    return feature_names, linear_feature_columns, dnn_feature_columns


def din_main(target_phase, train_final_df, item_raw_id2_idx_dict, user_raw_id2_idx_dict, item_content_vec_dict, feat_lbe_dict, val_final_df=None):
    print('generate din feature columns...')
    item_embed_np = get_init_item_embed(item_raw_id2_idx_dict, item_content_vec_dict)
    user_embed_np = get_init_user_embed(target_phase, user_raw_id2_idx_dict, item_content_vec_dict, is_use_whole_click=True)

    feature_names, linear_feature_columns, dnn_feature_columns = generate_din_feature_columns(
        ['user_id', 'item_id'],
        dense_features=item_dense_feat + sim_dense_feat + hist_time_diff_feat + hist_cnt_sim_feat + user_interest_dense_feat,
        feat_lbe_dict=feat_lbe_dict
    )

    train_input = build_din_input(train_final_df, feature_names)
    train_label = train_final_df['label'].values.astype(np.float32)

    if mode == 'offline':
        val_input = build_din_input(val_final_df, feature_names)
        val_label = val_final_df['label'].values.astype(np.float32)

    behavior_feature_list = ['item_id']
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    model = KDD_DIN(
        dnn_feature_columns,
        behavior_feature_list,
        dnn_hidden_units=HIDDEN_SIZE,
        att_hidden_size=(128, 64),
        att_weight_normalization=True,
        dnn_dropout=0.5,
        l2_reg_dnn=0,
        l2_reg_embedding=1e-6,
        user_init_embedding=user_embed_np,
        item_init_embedding=item_embed_np,
        device=device,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['binary_crossentropy', 'auc'],
    )
    if mode == 'offline':
        model.fit(
            train_input,
            train_label,
            batch_size=BATCH_SIZE,
            epochs=EPOCH,
            verbose=1,
            validation_data=(val_input, val_label)
        )
    else:
        model.fit(
            train_input,
            train_label,
            batch_size=BATCH_SIZE,
            epochs=EPOCH,
            verbose=1,
        )
    
    return model, feature_names
