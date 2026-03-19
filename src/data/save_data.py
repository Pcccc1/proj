import pickle
import pandas as pd
import os
from collections import defaultdict
from config import *

def make_item_sim_tuple(group_df):
    score_col = 'sim' if 'sim' in group_df.columns else 'score'
    group_df = group_df.sort_values(by=[score_col], ascending=False)
    item_score_tuples = list(zip(group_df['item_id'], group_df[score_col]))
    return item_score_tuples

def recall_df2dict(phase_df):
    phase_df = phase_df.groupby('user_id').apply(lambda x : make_item_sim_tuple(x)).reset_index().rename(
        columns={0 : 'item_score_list'}
    )
    item_score_list = phase_df['item_score_list'].apply(lambda item_score_list : sorted(item_score_list, key=lambda x:x[1], reverse=True))
    phase_user_item_score_dict = dict(zip(phase_df['user_id'], item_score_list))
    return phase_user_item_score_dict

"""
确保defalutdict等非基本数据结构被转换为普通的dict或list等可序列化对象，以便pickle能够正确保存和加载。
"""
def to_plain_serializable_obj(obj):
    if isinstance(obj, defaultdict):
        return {k: to_plain_serializable_obj(v) for k, v in obj.items()}
    if isinstance(obj, dict):
        return {k: to_plain_serializable_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_plain_serializable_obj(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(to_plain_serializable_obj(v) for v in obj)
    return obj


def save_recall_df_as_user_tuples_dict(total_recom_df, phase_full_sim_dict, prefix=""):
    save_path = os.path.join(user_data_dir, 'recall', mode)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    with open(os.path.join(save_path, prefix+'_total_recall_df.pkl'), 'wb') as f:
        pickle.dump(total_recom_df, f)

    phase_list = sorted(total_recom_df['phase'].dropna().unique().tolist())
    for phase in phase_list:
        phase_df = total_recom_df[total_recom_df['phase'] == phase]
        phase_user_item_score_dict = recall_df2dict(phase_df)
        phase_sim_dict = to_plain_serializable_obj(phase_full_sim_dict.get(phase, {}))

        with open(os.path.join(save_path, '{}_phase_{}.pkl'.format(prefix, phase)), 'wb') as f:
            pickle.dump(phase_user_item_score_dict, f)
        with open(os.path.join(save_path, '{}_phase_{}_sim.pkl'.format(prefix, phase)), 'wb') as f:
            pickle.dump(phase_sim_dict, f)
