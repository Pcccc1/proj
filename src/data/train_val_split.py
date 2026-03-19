import os
import numpy as np
import pandas as pd


from config import *

SAMPLE_USER_NUM = 1600
SEED = 42

os.makedirs(offline_answer_path, exist_ok=True)
os.makedirs(offline_test_path, exist_ok=True)
os.makedirs(offline_train_path, exist_ok=True)

rng = np.random.default_rng(SEED)



"""
划分验证集
"""
def tr_val_split(sample_user_num: int = SAMPLE_USER_NUM):
    for phase in range(now_phase + 1):
        df = pd.read_csv(
            f'{online_train_path}/{train_file_prefix}-{phase}.csv',
            header=None,
            names=['user_id', 'item_id', 'time'],
            dtype={'user_id': 'int32', 'item_id': 'int32', 'time': 'float64'},
        )

        df = df.drop_duplicates(['user_id', 'item_id', 'time'])

        user_cnt = df.groupby('user_id').size()
        eligible_users = user_cnt[user_cnt >= 2].index.to_numpy()

        if len(eligible_users) == 0:
            raise ValueError(f'No eligible users in phase {phase} for validation split.')
        
        k = min(sample_user_num, len(eligible_users))
        sample_user_ids = rng.choice(eligible_users, size=k, replace=False)

        is_val_user = df['user_id'].isin(sample_user_ids)
        train_df = df[~is_val_user].copy()
        val_df = df[is_val_user].copy()

        val_df = val_df.sort_values(['user_id', 'time'], kind='mergesort')
        
        last_pos = val_df.groupby('user_id').cumcount()
        user_size = val_df.groupby('user_id')['user_id'].transform('size')
        is_last = last_pos == (user_size - 1)

        answer_df = val_df[is_last].copy()
        his_df = val_df[~is_last].copy()
        q_time_df = answer_df[['user_id', 'time']].copy()

        train_df.to_csv(f'{offline_train_path}/{train_file_prefix}-{phase}.csv', header=False, index=False)
        answer_df.to_csv(f'{offline_answer_path}/{infer_answer_file_prefix}-{phase}.csv', header=False, index=False)

        phase_test_dir = f'{offline_test_path}/{test_file_prefix}-{phase}'
        os.makedirs(phase_test_dir, exist_ok=True)
        his_df.to_csv(f'{phase_test_dir}/{test_file_prefix}-{phase}.csv', header=False, index=False)
        q_time_df.to_csv(f'{phase_test_dir}/{infer_test_file_prefix}-{phase}.csv', header=False, index=False)



tr_val_split()