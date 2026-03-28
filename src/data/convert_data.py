import pandas as pd
import time

def make_item_time_tuple(group_df: pd.DataFrame) -> list:
    item_time_list = list(zip(group_df['item_id'], group_df['time']))
    return item_time_list

def make_user_time_tuple(group_df: pd.DataFrame) -> list:
    user_time_tuples = list(zip(group_df['user_id'], group_df['time']))
    return user_time_tuples

def get_user_item_time_dict(df: pd.DataFrame, drop_duplicates: bool = False) -> dict:
    x = df.sort_values(['user_id', 'time'], kind='mergesort')

    if drop_duplicates:
        print('drop duplicates user-item pairs, keep the last interaction')
        x = x.drop_duplicates(['user_id', 'item_id'], keep="last")

    x = x.groupby('user_id').apply(
        lambda group: make_item_time_tuple(group)).reset_index().rename(columns={0: 'item_time_list'})
   
    user_item_time_list = dict(zip(x['user_id'], x['item_time_list']))
    return user_item_time_list

def get_item_user_time_dict(df: pd.DataFrame)-> dict:
    item_user_df = df.sort_values(by=['item_id', 'time'])
    item_user_df = item_user_df.groupby('item_id').apply(
        lambda group : make_user_time_tuple(group)).reset_index().rename(columns={0 : 'user_time_list'})
    
    item_user_time_dict = dict(zip(item_user_df['item_id'], item_user_df['user_time_list']))
    return item_user_time_dict

def make_item_sim_tuple(group_df: pd.DataFrame):
    group_df = group_df.sort_values(by='sim', ascending=False)
    item_scores_tuples = list(zip(group_df['item_id'], group_df['sim']))
    return item_scores_tuples

def recall_df2dict(phase_df):
    phase_df = phase_df.groupby('user_id').apply(lambda group: make_item_sim_tuple(group)).reset_index().rename(columns={0: 'item_sim_list'})
    item_sim_list = phase_df['item_sim_list'].apply(
        lambda item_sim_list: sorted(item_sim_list, key=lambda x: x[1], reverse=True)
    )
    phase_user_item_sim_dict = dict(zip(phase_df['user_id'], item_sim_list))
    return phase_user_item_sim_dict

def recall_dict2df(recall_dict):
    recom_list = []
    for u, item_sim_list in recall_dict.items():
        for i, sim in item_sim_list:
            recom_list.append((u, i, sim))
    
    return pd.DataFrame(recom_list, columns=['user_id', 'item_id', 'sim'])

t = (2020, 4, 10, 0, 0, 0, 0, 0, 0)
time_end = time.mktime(t)
def time_info(time_delta):
    time_stamp = time_end * time_delta
    strcut_time = time.gmtime(time_stamp)
    day, hour, minu = strcut_time.tm_mday + 1, strcut_time.tm_hour + 1, strcut_time.tm_min + 1
    return day, hour, minu