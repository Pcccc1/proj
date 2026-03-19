import pandas as pd


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