from src.data.convert_data import make_user_time_tuple
from collections import defaultdict
from tqdm import tqdm
import math

"""
swing算法
"""
def swing(df, user_col='user_id', item_col='item_id', time_col='time'):
    item_user_df = df.sort_values(by=[item_col, time_col])
    item_user_df = item_user_df.groupby(item_col).apply(
        lambda group : make_user_time_tuple(group)).reset_index().rename(columns={0 : 'user_time_list'})
    
    item_user_time_dict = dict(zip(item_user_df[item_col], item_user_df['user_time_list']))

    user_item_time_dict = defaultdict(list)

    u_u_cnt = defaultdict(list)
    item_cnt = defaultdict(int)
    for item, user_time_list in tqdm(item_user_time_dict.items()):
        for u, u_time in user_time_list:
            item_cnt[item] += 1
            user_item_time_dict[u].append((item, u_time))
            for rele_u, rele_time in user_time_list:
                if u == rele_u:
                    continue

                key = (u, rele_u) if u < rele_u else (rele_u, u)
                u_u_cnt[key].append(item)


    sim_item = defaultdict(lambda: defaultdict(float))
    alpha = 5.0
    for (u, rele_u), co_items in u_u_cnt.items():
        num_co_items = len(co_items)
        for i in co_items:
            for rele_i in co_items:
                if rele_i == i:
                    continue
                sim_item[i][rele_i] += 1 / (alpha + num_co_items)
    
    sim_item_corr = sim_item.copy()
    for i, rele_items in sim_item.items():
        for rele_i, cij in rele_items.items():
            sim_item_corr[i][rele_i] = cij / math.sqrt((item_cnt[i] * item_cnt[rele_i]))
    
    return sim_item_corr, user_item_time_dict