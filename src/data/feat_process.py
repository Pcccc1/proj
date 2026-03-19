from src.data.convert_data import get_user_item_time_dict
from collections import defaultdict
from src.data.load_data import get_online_whole_click
import pandas as pd
import numpy as np
from config import *

def read_item_feat_df():
    print('begin read item feat df')
    item_feat_cols = ['item_id', ] + ['txt_embed_' + str(i) for i in range(128)] + ['img_embed_' + str(i) for i in range(128)]
    item_feat_df = pd.read_csv(
        f'{item_feat_file_path}',
        header=None,
        names=item_feat_cols,
    )
    item_feat_df['txt_embed_0'] = item_feat_df['txt_embed_0'].apply(lambda x: float(x[1:]))
    item_feat_df['img_embed_127'] = item_feat_df['img_embed_127'].apply(lambda x: float(x[:-1]))
    item_feat_df['txt_embed_127'] = item_feat_df['txt_embed_127'].apply(lambda x: float(x[:-1]))
    item_feat_df['img_embed_0'] = item_feat_df['img_embed_0'].apply(lambda x: float(x[1:]))
    return item_feat_df

def process_item_feat_df(item_feat_df: pd.DataFrame) -> pd.DataFrame:
    processed_item_feat_df = item_feat_df.copy()

    #norm
    txt_item_feat = processed_item_feat_df[txt_dense_feat].values
    img_item_feat = processed_item_feat_df[img_dense_feat].values
    txt_item_feat = txt_item_feat / np.linalg.norm(txt_item_feat, axis=1, keepdims=True)
    img_item_feat = img_item_feat / np.linalg.norm(img_item_feat, axis=1, keepdims=True)
    processed_item_feat_df[txt_dense_feat] = pd.DataFrame(txt_item_feat, columns=txt_dense_feat)
    processed_item_feat_df[img_dense_feat] = pd.DataFrame(img_item_feat, columns=img_dense_feat)

    return processed_item_feat_df

def fill_item_feat(processed_item_feat_df, item_content_vec_dict):
    online_total_click = get_online_whole_click()

    all_click_feat_df = pd.merge(online_total_click, processed_item_feat_df, on='item_id', how='left')
    missing_items = all_click_feat_df[all_click_feat_df['txt_embed_0'].isnull()]['item_id'].unique()
    print(f'number of missing items: {len(missing_items)}')
    user_item_time_hist_dict = get_user_item_time_dict(online_total_click)

    co_occur_dict = defaultdict(lambda : defaultdict(float))
    window = 5

    def cal_occ(sentence):
        for i, word in enumerate(sentence):
            hist_len = len(sentence)
            for j in range(max(0, i-window), min(hist_len, i+window)):
                if j == i or word == sentence[j]:
                    continue
                loc_weight = (0.9 ** abs(j - i))
                co_occur_dict[word][sentence[j]] += loc_weight
    
    for u, item_time in user_item_time_hist_dict.items():
        hist_items = [i for i, t in item_time]
        cal_occ(hist_items)

    miss_item_content_vec_dict = {}
    for miss_item in missing_items:
        co_occur_item_dict = co_occur_dict[miss_item]
        weight_vec = np.zeros(256)
        sum_weight = 0.0
        for co_item, weight in co_occur_item_dict.items():
            if co_item in item_content_vec_dict:
                sum_weight += weight
                co_item_vec = item_content_vec_dict[co_item]
                weight_vec += weight * co_item_vec
        
        weight_vec /= sum_weight + 1e-8
        txt_item_feat_vec = weight_vec[:128] / np.linalg.norm(weight_vec[:128])
        img_item_feat_vec = weight_vec[128:] / np.linalg.norm(weight_vec[128:])
        cnt_vec = np.concatenate([txt_item_feat_vec, img_item_feat_vec])
        miss_item_content_vec_dict[miss_item] = cnt_vec

    miss_item_feat_df = pd.DataFrame()
    miss_item_feat_df[item_dense_feat] = pd.DataFrame(miss_item_content_vec_dict.values(), columns=item_dense_feat)
    miss_item_feat_df['item_id'] = list(miss_item_content_vec_dict.keys())
    miss_item_feat_df = miss_item_feat_df[['item_id'] + item_dense_feat]

    return miss_item_feat_df, miss_item_content_vec_dict



def obtain_entire_item_feat_df():
    item_feat_df = read_item_feat_df()
    processed_item_feat_df = process_item_feat_df(item_feat_df)
    item_content_vec_dict = dict(zip(processed_item_feat_df['item_id'], processed_item_feat_df[item_dense_feat].values))
    miss_item_feat_df, miss_item_content_vec_dict = fill_item_feat(processed_item_feat_df, item_content_vec_dict)
    processed_item_feat_df = pd.concat([processed_item_feat_df, miss_item_feat_df], ignore_index=True)
    item_content_vec_dict.update(miss_item_content_vec_dict)

    return processed_item_feat_df, item_content_vec_dict