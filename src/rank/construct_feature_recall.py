import pandas as pd
from src.data.convert_data import time_info
from src.data.feat_process import read_item_feat_df
from src.recall.content_sim_item import get_content_sim_item
from config import *

def _coerce_recall_item_dict(recall_item_dict):
    """
    Backward compatibility:
    old cached recall files may store a DataFrame(user_id,item_id,sim),
    while ranking feature builder expects dict[user_id] -> [(item_id, sim)].
    """
    if isinstance(recall_item_dict, dict) and {'user_id', 'item_id', 'sim'}.issubset(set(recall_item_dict.keys())):
        # Some stale cache files may persist DataFrame-like "column dict" structure.
        recall_item_dict = pd.DataFrame(recall_item_dict)

    if isinstance(recall_item_dict, pd.DataFrame):
        required_cols = {'user_id', 'item_id', 'sim'}
        if not required_cols.issubset(set(recall_item_dict.columns)):
            raise ValueError(
                f"recall dataframe must contain columns {required_cols}, got {set(recall_item_dict.columns)}"
            )
        recall_item_dict = (
            recall_item_dict
            .sort_values(['user_id', 'sim'], ascending=[True, False], kind='mergesort')
            .groupby('user_id')
            .apply(lambda group: list(zip(group['item_id'], group['sim'])))
            .to_dict()
        )
    return recall_item_dict


def obtain_user_hist_feat(u, user_item_dict):
    user_hist_seq = [i for i, t in user_item_dict[u]]
    user_hist_time_seq = [t for i, t in user_item_dict[u]]
    user_hist_day_seq, user_hist_hour_seq, user_hist_min_seq = zip(*[time_info(t) for i, t in user_item_dict[u]])
    return [user_hist_seq, user_hist_time_seq, list(user_hist_day_seq), list(user_hist_hour_seq), list(user_hist_min_seq)]


def organize_user_feat_each_other(u, recall_items, user_item_dict, item_content_sim_dict, strategy_item_dict, phase):

    user_hist_info = obtain_user_hist_feat(u, user_item_dict)

    hist_num = 3
    recall_items_sum_cf_sim2_hist = []
    recall_items_max_cf_sim2_hist = []
    recall_items_cnt_sim2_hist = []

    user_hist_items = user_item_dict[u][-hist_num:]
    for recall_i, rating in recall_items:
        if rating > 0:
            max_cf_sim2_hist = []
            sum_cf_sim2_hist = []
            cnt_sim2_hist = []
            for hist_i, t in user_hist_items:
                sum_sim_value = 0.0
                max_sim_value = 0.0

                for strategy, item_sim_dict in strategy_item_dict.items():
                    strategy_sim_value = item_sim_dict.get(hist_i, {}).get(recall_i, 0.0) + item_sim_dict.get(recall_i, {}).get(hist_i, 0.0)
                    sum_sim_value += strategy_sim_value
                    max_sim_value = max(max_sim_value, strategy_sim_value)
                
                cnt_sim_value = item_content_sim_dict.get(hist_i, {}).get(recall_i, 0.0) + item_content_sim_dict.get(recall_i, {}).get(hist_i, 0.0)
                sum_cf_sim2_hist.append(sum_sim_value)
                max_cf_sim2_hist.append(max_sim_value)
                cnt_sim2_hist.append(cnt_sim_value)

            while len(sum_cf_sim2_hist) < hist_num:
                sum_cf_sim2_hist.append(0.0)
                max_cf_sim2_hist.append(0.0)
                cnt_sim2_hist.append(0.0)
        else:
            sum_cf_sim2_hist = [0.0] * hist_num
            max_cf_sim2_hist = [0.0] * hist_num
            cnt_sim2_hist = [0.0] * hist_num
        
        recall_items_sum_cf_sim2_hist.append(sum_cf_sim2_hist)
        recall_items_max_cf_sim2_hist.append(max_cf_sim2_hist)
        recall_items_cnt_sim2_hist.append(cnt_sim2_hist)
    
    recom_items = []
    for item_rating, sum_cf_sim2_hist, max_cf_sim2_hist, cnt_sim2_hist in zip(recall_items,
                                                                              recall_items_sum_cf_sim2_hist,
                                                                              recall_items_max_cf_sim2_hist,
                                                                              recall_items_cnt_sim2_hist):
        recom_items.append([u, item_rating[0], item_rating[1], phase] + sum_cf_sim2_hist + max_cf_sim2_hist + cnt_sim2_hist + user_hist_info)
    
    return recom_items
        

def organize_recall_feat(recall_item_dict, user_item_dict, item_sim_dict, item_content_sim_dict, phase):
    recall_item_dict = _coerce_recall_item_dict(recall_item_dict)

    recom_columns = ['user_id', 'item_id', 'sim', 'phase'] + \
                    [f'sum_sim2int_{i}' for i in range(1, 4)] + \
                    [f'max_sim2int_{i}' for i in range(1, 4)] + \
                    [f'cnt_sim2int_{i}' for i in range(1, 4)] + \
                    ['hist_item_id', 'hist_time', 'hist_day_id', 'hist_hour_id', 'hist_minute_id']
    recom_item = []
    invalid_user_keys = 0
    for u, recall_items in recall_item_dict.items():
        if u not in user_item_dict:
            try:
                u = int(u)
            except (TypeError, ValueError):
                invalid_user_keys += 1
                continue
            if u not in user_item_dict:
                invalid_user_keys += 1
                continue
        recom_item.extend(organize_user_feat_each_other(u, recall_items, user_item_dict, item_content_sim_dict, item_sim_dict, phase))
    if invalid_user_keys:
        print(f'[warn] skip {invalid_user_keys} invalid users in recall_item_dict')

    recall_recom_df = pd.DataFrame(recom_item, columns=recom_columns)
    recall_recom_df['sim_rank_score'] = recall_recom_df.groupby('user_id')['sim'].rank(method='first', ascending=True) / topk_num

    return recall_recom_df
