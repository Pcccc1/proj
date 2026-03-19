from src.data.convert_data import get_user_item_time_dict
from collections import defaultdict
import numpy as np
import math
from tqdm import tqdm

"""
itemcf
"""
def item_cf(df):
    user_item_time_dict = get_user_item_time_dict(df)

    sim_item = defaultdict(lambda: defaultdict(float))
    item_cnt = defaultdict(int)

    for user, item_time_list in user_item_time_dict.items():
        for loc_1, (i, i_time) in enumerate(item_time_list):
            item_cnt[i] += 1
            for loc_2, (rele_i, rele_time) in enumerate(item_time_list):
                if i == rele_i:
                    continue
                loc_alpha = 1.0 if loc_2 > loc_1 else 0.7
                loc_weight = loc_alpha * (0.8 ** (np.abs(loc_2 - loc_1) - 1))
                time_weight = np.exp(-15000 * np.abs(rele_time - i_time))

                sim_item[i][rele_i] += loc_weight * time_weight / math.log(1 + len(item_time_list))
    
    sim_item_corr = sim_item.copy()
    for i, rele_items in tqdm(sim_item.items()):
        for rele_i, cij in rele_items.items():
            sim_item_corr[i][rele_i] = cij / math.sqrt(item_cnt[i] * item_cnt[rele_i])
    
    return sim_item_corr, user_item_time_dict