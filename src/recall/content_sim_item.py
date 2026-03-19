import numpy as np
import pickle
from collections import defaultdict
from config import *
import faiss


"""
基于内容的物品相似度计算
"""
def get_content_sim_item(
    item_feat_df,
    topk=100,
    is_load_from_file=True,
    txt_weight: float = 1.0,
    img_weight: float = 1.0,
):
    sim_path = os.path.join(user_data_dir, 'item_content_sim_dict.pkl')

    if is_load_from_file and os.path.exists(sim_path):
        with open(sim_path, 'rb') as f:
            return pickle.load(f)

    print('begin compute item content sim dict...')

    item_idx_2_rawid_dict = dict(zip(item_feat_df.index, item_feat_df['item_id']))
    txt_item_feat_df = item_feat_df.filter(regex='txt*')
    img_item_feat_df = item_feat_df.filter(regex='img*')

    txt_item_feat_np = np.ascontiguousarray(txt_item_feat_df.values, dtype='float32')
    ima_item_feat_np = np.ascontiguousarray(img_item_feat_df.values, dtype='float32')   

    txt_item_feat_np = txt_item_feat_np / np.linalg.norm(txt_item_feat_np, axis=1, keepdims=True)
    ima_item_feat_np = ima_item_feat_np / np.linalg.norm(ima_item_feat_np, axis=1, keepdims=True)

    txt_index = faiss.IndexFlatIP(128)
    txt_index.add(txt_item_feat_np)

    ima_index = faiss.IndexFlatIP(128)
    ima_index.add(ima_item_feat_np)

    item_sim_dict = defaultdict(lambda: defaultdict(float))

    def serach(feat_index, feat_np, weight):
        sim, idx = feat_index.search(feat_np, topk)
        for target_idx, sim_value_list, rele_idx_list in zip(range(len(feat_np)), sim, idx):
            target_raw_id = item_idx_2_rawid_dict[target_idx]
            for rele_idx, sim_value in zip(rele_idx_list[1:], sim_value_list[1:]):
                rele_raw_id = item_idx_2_rawid_dict[rele_idx]
                item_sim_dict[target_raw_id][rele_raw_id] = item_sim_dict.get(target_raw_id, {}).get(rele_raw_id, 0) + sim_value * weight

    serach(txt_index, txt_item_feat_np, txt_weight)
    serach(ima_index, ima_item_feat_np, img_weight)

    if is_load_from_file:
        with open(sim_path, 'wb') as f:
            pickle.dump(item_sim_dict, f)
    return item_sim_dict
