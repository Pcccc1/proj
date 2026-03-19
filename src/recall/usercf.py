from src.data.convert_data import get_user_item_time_dict, get_item_user_time_dict
import math
from collections import defaultdict
from tqdm import tqdm

"""
基于用户的协同过滤算法实现，计算用户之间的相似度，并根据相似度进行推荐。
"""
def user_cf(df):
    user_item_time_dict = get_user_item_time_dict(df)
    item_user_time_dict = get_item_user_time_dict(df)

    sim_user = defaultdict(lambda : defaultdict(float))
    user_cnt = defaultdict(int) 

    for item, user_time_list in tqdm(item_user_time_dict.items()):
        num_users = len(user_time_list)
        for u, t in user_time_list:
            user_cnt[u] += 1

            for rele_u, rele_t in user_time_list:
                if u == rele_u:
                    continue
                sim_user[u][rele_u] += 1.0 / math.log(1 + num_users)

    sim_user_corr = sim_user.copy()
    for u, rele_u in tqdm(sim_user.items()):
        for v, cuv in rele_u.items():
            sim_user_corr[u][v] = cuv / math.sqrt(user_cnt[u] * user_cnt[v])

    return sim_user_corr, user_item_time_dict