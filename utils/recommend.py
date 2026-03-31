import numpy as np
from collections import defaultdict
import pandas as pd

def _safe_content_sim(item_content_sim_dict, left_item, right_item):
    left_sim_dict = item_content_sim_dict.get(left_item)
    if left_sim_dict is None:
        return 0.0
    return left_sim_dict.get(right_item, 0.0)

def item_based_recommend(sim_item_corr, user_item_time_dict, item_content_sim_dict, user_id, topk, item_num, alpha=15000,
                         item_cnt_dict=None, user_cnt_dict=None):
    rank = defaultdict(float)
    if user_id not in user_item_time_dict:
        return []
    interacted_item_times = user_item_time_dict[user_id]
    min_time = min(time for item, time in interacted_item_times)
    interacted_items = set(item for item, time in interacted_item_times)

    miss_item_num = 0
    for loc, (i, time) in enumerate(interacted_item_times):
        if i not in sim_item_corr:
            miss_item_num += 1
            continue
        for j, wij in sorted(sim_item_corr[i].items(), key=lambda x: x[1], reverse=True)[:topk]:
            if j not in interacted_items:
                content_weight = 1.0
                content_weight += _safe_content_sim(item_content_sim_dict, i, j)
                content_weight += _safe_content_sim(item_content_sim_dict, j, i)

                time_weight = np.exp(alpha * (time - min_time))
                loc_weight = (0.9 ** (len(interacted_item_times) - loc))
                rank[j] += loc_weight * time_weight * content_weight * wij
    
    # if miss_item_num > 10:
    #     print(f"user_id {user_id} has {miss_item_num} items not in sim_item_corr, may cause bad recall performance")

    if item_cnt_dict is not None:
        for loc, item in enumerate(rank):
            rank[item] = re_rank(rank[item], item, user_id, item_cnt_dict, user_cnt_dict)

    sorted_rank_items = sorted(rank.items(), key=lambda x: x[1], reverse=True)
    return sorted_rank_items[0:item_num]

def user_based_recommend(sim_user_corr, user_item_time_dict, item_content_sim_dict, user_id, topk, item_num, alpha=15000,
                         item_cnt_dict=None, user_cnt_dict=None):
    rank = defaultdict(float)
    if user_id not in user_item_time_dict or user_id not in sim_user_corr:
        return []
    interacted_items = set(item for item, time in user_item_time_dict[user_id])
    interacted_item_time_list = user_item_time_dict[user_id]
    interacted_num = len(interacted_items)

    min_time = min(time for item, time in interacted_item_time_list)
    time_weight_dict = {item: np.exp(alpha * (time - min_time)) for item, time in interacted_item_time_list}
    loc_weight_dict = {item: 0.9 ** (interacted_num - loc) for loc, (item, time) in enumerate(interacted_item_time_list)}

    for sim_v, wuv in sorted(sim_user_corr[user_id].items(), key=lambda x: x[1], reverse=True)[:topk]:
        if sim_v not in user_item_time_dict:
            continue
        for j, j_time in user_item_time_dict[sim_v]:
            if j not in interacted_items:
                content_weight = 1.0
                for loc, (item, time) in enumerate(interacted_item_time_list):
                    loc_weight = loc_weight_dict[item]
                    time_weight = time_weight_dict[item]
                    content_weight += _safe_content_sim(item_content_sim_dict, item, j) * loc_weight * time_weight
                
                rank[j] += wuv * content_weight
    if item_cnt_dict is not None:
        for loc, item in enumerate(rank):
            rank[item] = re_rank(rank[item], item, user_id, item_cnt_dict, user_cnt_dict)

    rec_items = sorted(rank.items(), key=lambda x: x[1], reverse=True)
    return rec_items[0:item_num]

""" 
重排
"""
def re_rank(sim, i, u, item_cnt_dict, user_cnt_dict):
    user_cnt = user_cnt_dict.get(u, 1.0)

    if item_cnt_dict.get(i, 1.0) < 4:
        heat = np.log(item_cnt_dict.get(i, 1.0) + 2)
    elif item_cnt_dict.get(i, 1.0) >= 4 and item_cnt_dict.get(i, 1.0) < 10:
        if user_cnt > 50:
            heat = item_cnt_dict.get(i, 1.0) * 1
        elif user_cnt > 25:
            heat = item_cnt_dict.get(i, 1.0) * 1.2
        else:
            heat = item_cnt_dict.get(i, 1.0) * 1.6
    else:
        if user_cnt > 50:
            user_cnt_k = 0.4
        elif user_cnt > 10:
            user_cnt_k = 0.1
        else:
            user_cnt_k = 0
        heat = item_cnt_dict.get(i, 1.0) ** user_cnt_k + 10 - 10 ** user_cnt_k
    sim *= 2.0 / heat
    return sim


def get_predict(df, pred_col, top_fill):
    top_fill = [int(t) for t in top_fill.split(',')]
    scores = [-1 * i for i in range(1, len(top_fill) + 1)]
    ids = list(df['user_id'].unique())
    fill_df = pd.DataFrame(ids * len(top_fill), columns=['user_id'])
    fill_df.sort_values(by='user_id', inplace=True)
    fill_df['item_id'] = top_fill * len(ids)
    fill_df[pred_col] = scores * len(ids)
    df = pd.concat([df, fill_df], axis=0, ignore_index=True, sort=False)
    df.sort_values(pred_col, ascending=False, inplace=True)
    df = df.drop_duplicates(subset=['user_id', 'item_id'], keep='first')
    df['rank'] = df.groupby('user_id')[pred_col].rank(method='first', ascending=False)
    df = df[df['rank'] <= 50]
    df = df.groupby('user_id')['item_id'].apply(list).apply(pd.Series).reset_index()
    return df
