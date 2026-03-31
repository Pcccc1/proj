from src.recall.itemcf import item_cf
from src.recall.bi_graph import bi_graph
from src.recall.swing import swing
from src.recall.usercf import user_cf
from utils.recommend import item_based_recommend, user_based_recommend
from src.recall.TwoTower.readDNNresults import _read_dnn_results
from src.data.load_data import obtain_topk_click
from tqdm import tqdm
from collections import defaultdict
import pandas as pd

def get_multi_source_sim_dict_results(history_df, recall_methods={'item_cf', 'bi-graph', 'swing', 'user_cf'}):
    recall_sim_pair_dict = {}
    if 'item_cf' in recall_methods:
        print('item_cf recall...')
        item_sim_dict, _ = item_cf(history_df)
        recall_sim_pair_dict['item_cf'] = item_sim_dict
        print('item_cf recall done!, pair_num={}'.format(len(item_sim_dict)))

    if 'bi-graph' in recall_methods:
        print('bi-graph recall...')
        bi_graph_sim_dict, _ = bi_graph(history_df)
        recall_sim_pair_dict['bi-graph'] = bi_graph_sim_dict
        print('bi-graph recall done!, pair_num={}'.format(len(bi_graph_sim_dict)))
    
    if 'swing' in recall_methods:
        print('swing recall...')
        swing_sim_dict, _ = swing(history_df)
        recall_sim_pair_dict['swing'] = swing_sim_dict
        print('swing recall done!, pair_num={}'.format(len(swing_sim_dict)))
    
    if 'user_cf' in recall_methods:
        print('user_cf recall...')
        user_sim_dict, _ = user_cf(history_df)
        recall_sim_pair_dict['user_cf'] = user_sim_dict
        print('user_cf recall done!, pair_num={}'.format(len(user_sim_dict)))
    return recall_sim_pair_dict


def get_recall_results(item_sim_dict, user_item_dict, item_content_sim_dict, target_user_ids=None, item_based=True,
                       item_cnt_dict=None, user_cnt_dict=None):
    if target_user_ids is None:
        target_user_ids = user_item_dict.keys()
    recall_item_dict = {}

    top50_click_items, _ = obtain_topk_click()

    for u in tqdm(target_user_ids):
        if item_based:
            recall_items = item_based_recommend(item_sim_dict, user_item_dict, item_content_sim_dict, u,
                                                topk=800, item_num=200, item_cnt_dict=item_cnt_dict, user_cnt_dict=user_cnt_dict)
        else:
            recall_items = user_based_recommend(item_sim_dict, user_item_dict, item_content_sim_dict, u,
                                                topk=800, item_num=200, item_cnt_dict=item_cnt_dict, user_cnt_dict=user_cnt_dict)
        if len(recall_items) == 0:
            recall_items = [(top50_click_items[0], 0.0)]
        
        recall_item_dict[u] = recall_items

    return recall_item_dict


def norm_recall_item_score_list(sorted_recall_item_list):
    if len(sorted_recall_item_list) == 0:
        return sorted_recall_item_list
    assert sorted_recall_item_list[0][1] >= sorted_recall_item_list[-1][1]
    max_sim = sorted_recall_item_list[0][1]
    min_sim = sorted_recall_item_list[-1][1]

    norm_sorted_recall_item_list = []
    for item, score in sorted_recall_item_list:
        if max_sim > 0:
            norm_socre = 1.0 * (score - min_sim) / (max_sim - min_sim) if max_sim > min_sim else 1.0
        else:
            norm_socre = 0.0
        norm_sorted_recall_item_list.append((item, norm_socre))
    return norm_sorted_recall_item_list

def norm_user_recall_item_dict(recall_item_dict):
    norm_recall_item_dict = defaultdict(lambda: defaultdict(float))
    for u, sorted_recall_item_list in recall_item_dict.items():
        norm_recall_item_dict[u] = norm_recall_item_score_list(sorted_recall_item_list)
    return norm_recall_item_dict

def agg_recall_results(recall_item_dict_list_dict, is_norm=True, weight_dict={}):
    agg_recall_item_dict = defaultdict(lambda: defaultdict(float))
    for name, recall_item_dict in recall_item_dict_list_dict.items():
        if is_norm:
            recall_item_dict = norm_user_recall_item_dict(recall_item_dict)
        weight = weight_dict.get(name, 1.0)
        for u, recall_items in recall_item_dict.items():
            for i, score in recall_items:
                agg_recall_item_dict[u][i] += weight * score
        
    recall_u_i_score_pair_list = []
    for u, recall_item_dict in agg_recall_item_dict.items():
        for i, score in recall_item_dict.items():
            recall_u_i_score_pair_list.append((u, i, score))
    
    recall_df = pd.DataFrame.from_records(recall_u_i_score_pair_list, columns=['user_id', 'item_id', 'sim'])
    return recall_df


def do_multi_recall_results(recall_sim_pair_dict, user_item_time_dict, item_content_sim_dict,
                            target_user_ids=None, phase=None, item_cnt_dict=None,
                            user_cnt_dict=None, recall_methods={'item_cf', 'bi-graph', 'swing', 'user_cf', 'TwoTower'}):
    if target_user_ids is None:
        target_user_ids = user_item_time_dict.keys()

    recall_item_list_dict = {}
    for name, sim_dict in recall_sim_pair_dict.items():
        if name in {'item_cf', 'bi-graph', 'swing'}:
            recall_item_dict = get_recall_results(sim_dict, user_item_time_dict, item_content_sim_dict, target_user_ids,
                                                  item_based=True, item_cnt_dict=item_cnt_dict, user_cnt_dict=user_cnt_dict)
        else:
            recall_item_dict = get_recall_results(sim_dict, user_item_time_dict, item_content_sim_dict, target_user_ids,
                                                  item_based=False, item_cnt_dict=item_cnt_dict, user_cnt_dict=user_cnt_dict)
        recall_item_list_dict[name] = recall_item_dict

    if 'TwoTower' in recall_methods:
        dnn_recall_item_dict = _read_dnn_results(phase)
        recall_item_list_dict['TwoTower'] = dnn_recall_item_dict
        print(f"TwoTower recall done!")
    
    return agg_recall_results(recall_item_list_dict, is_norm=True)


