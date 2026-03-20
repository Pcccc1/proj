import os
from pathlib import Path
import pickle
from src.data.convert_data import recall_df2dict, recall_dict2df, get_user_item_time_dict
from src.data.load_data import get_phase_click, get_whole_phase_click
from utils.recommend import re_rank
import pandas as pd
import time

def _recall_dnn_re_rank(recom_df, phase, mode='offline'):
    all_click, click_q_time = get_phase_click(phase=phase)
    phase_whole_click = get_whole_phase_click(all_click=all_click, click_q_time=click_q_time)

    if mode == 'online':
        user_item_hist_dict = get_user_item_time_dict(phase_whole_click)
    else:
        phase_click = get_user_item_time_dict(all_click)
    
    item_cnt_dict = all_click.groupby('item_id')['user_id'].count().to_dict()
    user_cnt_dict = all_click.groupby('user_id')['item_id'].count().to_dict()

    recom_list = []
    for row in recom_df.itertuples(index=False):
        uid = int(row.user_id)
        iid = int(row.item_id)
        sim = row.sim
        sim = re_rank(row.sim, iid, uid, item_cnt_dict, user_cnt_dict)
        recom_list.append((uid, iid, sim, row.phase))

    return pd.DataFrame(recom_list, columns=['user_id', 'item_id', 'sim', 'phase'])



def _read_dnn_results(phase, data_path='data/user_data/recall/offline/') -> dict:
    recall_user_item_score_dict = {}
    today = time.strftime("%Y%m%d")
    file_path = Path(f"{data_path}/two_tower-{today}_phase_{phase}.pkl")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"DNN results file not found: {file_path}")
    
    with open(file_path, 'rb') as f:
        recall_user_item_score_dict = pickle.load(f)
    recom_df = recall_dict2df(recall_user_item_score_dict)
    recom_df['phase'] = phase
    recom_df = _recall_dnn_re_rank(recom_df, phase, mode='offline')
    recall_user_item_score_dict = recall_df2dict(recom_df)

    return recall_user_item_score_dict
