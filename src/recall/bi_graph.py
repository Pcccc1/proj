from src.data.convert_data import get_item_user_time_dict, get_user_item_time_dict
from collections import defaultdict
import numpy as np
import math
"""
基于用户-物品二分图的召回算法
"""
def bi_graph(df):
    item_user_time_dict = get_item_user_time_dict(df)
    user_item_time_dict = get_user_item_time_dict(df)


    sim_item = defaultdict(lambda: defaultdict(float))

    for item, user_time_lists in item_user_time_dict.items():

        for u, item_time in user_time_lists:
            user_plenty_len = len(user_item_time_dict[u])

            for rele_item, rele_item_time in user_item_time_dict[u]:
                weight = np.exp(-15000 * np.abs(rele_item_time - item_time))
                sim_item[item][rele_item] += weight / (math.log(1 + user_plenty_len) * math.log(1 + len(user_time_lists)))
    
    return sim_item, user_item_time_dict