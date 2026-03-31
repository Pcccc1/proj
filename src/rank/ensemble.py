import pickle
import os
import pandas as pd
from config import *
from src.data.load_data import obtain_topk_click
from utils.recommend import get_predict


def norm_sim(sim_df, weight=0.0):
    min_sim = sim_df.min()
    max_sim = sim_df.max()
    if max_sim == min_sim:
        sim_df = sim_df.apply(lambda x : 1.0)
    else:
        sim_df = sim_df.apply(lambda x : 1.0 * (x - min_sim) / (max_sim - min_sim))
    
    sim_df = sim_df.apply(lambda x : weight + x)
    return sim_df


def ensemble(output_ranking_filename):
    rank_output_dir = os.path.join(user_data_dir, 'rank')
    lgb_output_file = 'ranker-' + output_ranking_filename + '.csv-pkl'
    lgb_ranker_df = pickle.load(open(f'{rank_output_dir}/{lgb_output_file}', 'rb'))
    lgb_ranker_df['sim'] = lgb_ranker_df.groupby('user_id')['sim'].transform(lambda x : norm_sim(x))
    print('read LightGBM ranking completed.')

    din_output_file = 'din-' + output_ranking_filename + '.csv-pkl'
    din_df = pickle.load(open(f'{rank_output_dir}/{din_output_file}', 'rb') )
    din_df['sim'] = din_df.groupby('user_id')['sim'].transform(lambda x : norm_sim(x))
    print('read DIN ranking completed.')

    din_lgb_full_df = pd.concat([lgb_ranker_df, din_df], axis=0)
    din_lgb_full_df = din_lgb_full_df.groupby(['user_id', 'item_id', 'phase'])['sim'].sum().reset_index()

    _, top50_click = obtain_topk_click()
    res3 = get_predict(din_lgb_full_df, 'sim', top50_click)
    res3.to_csv(output_path + '/result.csv', index=False, header=None)