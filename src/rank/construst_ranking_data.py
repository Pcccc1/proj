from src.rank .construct_feature_info_feat import organize_user_item_feat, get_word2vec_feat
from src.rank.construct_feature_recall import organize_recall_feat
from src.rank.construct_feature_label import organize_label_interact_feat_df
import os
from config import *
from src.data.load_data import get_phase_click, get_whole_phase_click
from src.data.feat_process import read_item_feat_df
from src.recall.recall import get_multi_source_sim_dict_results, do_multi_recall_results
import pickle
import pandas as pd
from src.data.convert_data import get_user_item_time_dict, time_info
from src.recall.content_sim_item import get_content_sim_item

def get_history_and_last_click_df(click_df):
    click_df = click_df.sort_values(by=['user_id', 'time'])
    click_last_df = click_df.groupby('user_id').tail(1)

    def hist_func(user_df):
        num = len(user_df)
        if num == 1:
            return user_df
        else:
            return user_df[:-1]
    
    click_history_df = click_df.groupby('user_id').apply(hist_func).reset_index(drop=True)

    return click_history_df, click_last_df


def sliding_obtain_training_df(phase, item_content_sim_dict, is_sliding_compute_sim=False):
    compute_mode = 'once' if not is_sliding_compute_sim else 'multi'
    saving_training_path = os.path.join(user_data_dir, 'training', mode, compute_mode, str(phase))

    if os.path.exists(os.path.join(saving_training_path, 'step_user_recall_item_dict.pkl')):
        print(f'phase{phase} training data already exists, loading...')
        return
    
    all_click, click_q_time = get_phase_click(phase)

    click_history_df = all_click
    recall_methods = {'item_cf', 'bi-graph', 'user_cf', 'swing'}

    if not os.path.exists(saving_training_path):
        os.makedirs(saving_training_path)

    total_step = 10
    step = 0
    full_sim_pair_dict = get_multi_source_sim_dict_results(click_history_df, recall_methods=recall_methods)
    pickle.dump(full_sim_pair_dict, open(os.path.join(saving_training_path, 'full_sim_pair_dict.pkl'), 'wb'))

    step_user_recall_item_dict = {}
    step_strategy_sim_pair_dict = {}

    while step < total_step:
        print(f'step={step} start...')
        click_history_df, click_last_df = get_history_and_last_click_df(click_history_df)
        user_item_time_dict = get_user_item_time_dict(click_history_df)

        if is_sliding_compute_sim:
            sim_pair_dict = get_multi_source_sim_dict_results(click_history_df, recall_methods=recall_methods)
        else:
            sim_pair_dict = full_sim_pair_dict
        
        user_recall_item_dict = do_multi_recall_results(sim_pair_dict, user_item_time_dict, item_content_sim_dict=item_content_sim_dict)
        step_user_recall_item_dict[step] = user_recall_item_dict
        if is_sliding_compute_sim:
            step_strategy_sim_pair_dict[step] = sim_pair_dict
        step += 1

        pickle.dump(step_user_recall_item_dict, open(os.path.join(saving_training_path, 'step_user_recall_item_dict.pkl'), 'wb'))

        if is_sliding_compute_sim:
            pickle.dump(step_strategy_sim_pair_dict, open(os.path.join(saving_training_path, 'step_strategy_sim_pair_dict.pkl'), 'wb'))
        
        if mode == 'offline':
            all_user_item_dict = get_user_item_time_dict(all_click)

            val_user_recall_item_dict = do_multi_recall_results(full_sim_pair_dict, all_user_item_dict, item_content_sim_dict=item_content_sim_dict, target_user_ids=click_q_time['user_id'].unique(), recall_methods=recall_methods)

            pickle.dump(val_user_recall_item_dict, open(os.path.join(saving_training_path, 'val_user_recall_item_dict.pkl'), 'wb'))
        


def organize_train_data(phase, item_content_vec_dict, is_sliding_compute_sim=False, load_from_file=True, total_step=10):
    compute_mode = 'once' if not is_sliding_compute_sim else 'multi'
    saving_training_path = os.path.join(user_data_dir, 'training', mode, compute_mode, str(phase))

    save_result_train_val_path = os.path.join(saving_training_path, 'train_val_label_target_id_data.pkl')
    if load_from_file and os.path.exists(save_result_train_val_path):
        return pickle.load(open(save_result_train_val_path, 'rb'))
    
    all_click, test_q_time = get_phase_click(phase)

    click_history_df = all_click

    full_sim_pair_dict = pickle.load(open(os.path.join(saving_training_path, 'full_sim_pair_dict.pkl'), 'rb'))
    
    step_user_recall_item_dict = pickle.load(open(os.path.join(saving_training_path, 'step_user_recall_item_dict.pkl'), 'rb'))

    if is_sliding_compute_sim:
        step_strategy_sim_pair_dict = pickle.load(open(os.path.join(saving_training_path, 'step_strategy_sim_pair_dict.pkl'), 'rb'))
    
    print('read recall data done, start organizing training data...')

    train_full_df_list = []
    for step in range(total_step):
        print(f'step={step} start...')
        click_history_df, click_last_df = get_history_and_last_click_df(click_history_df)
        user_recall_item_dict = step_user_recall_item_dict[step]
        strategy_sim_pair_dict = (
            step_strategy_sim_pair_dict[step]
            if is_sliding_compute_sim else full_sim_pair_dict
        )
        user_item_time_dict = get_user_item_time_dict(click_history_df)

        click_last_recall_recom_df = organize_recall_feat(
            recall_item_dict=user_recall_item_dict,
            user_item_dict=user_item_time_dict,
            item_sim_dict=strategy_sim_pair_dict,
            item_content_vec_dict=item_content_vec_dict,
            phase=phase
        )

        assert len(user_item_time_dict) == len(click_last_recall_recom_df['user_id'].unique()) == len(click_last_df['user_id'].unique())

        train_full_df = organize_label_interact_feat_df(
            click_last_df=click_last_df,
            click_last_recall_recom_df=click_last_recall_recom_df,
            phase=phase,
        )
        train_full_df['step'] = step
        assert 'sim' in train_full_df.columns
        train_full_df_list.append(train_full_df)
    
    print('organize training data done')

    train_full_df = pd.concat(train_full_df_list, ignore_index=True)

    if mode == 'offline':
        print('start organizing val data...')
        val_user_item_dict = get_user_item_time_dict(all_click)
        val_user_recall_item_dict = pickle.load(open(os.path.join(saving_training_path, 'val_user_recall_item_dict.pkl'), 'rb'))

        phase_val_last_click_answer_df = pd.read_csv(f'{offline_answer_path}/{infer_answer_file_prefix}-{phase}.csv', header=None, 
                                                        names=['user_id', 'item_id', 'time'])
        phase_val_last_click_recall_recom_df = organize_recall_feat(val_user_recall_item_dict, val_user_item_dict, full_sim_pair_dict, item_content_vec_dict, phase)

        val_full_df = organize_label_interact_feat_df(phase_val_last_click_answer_df, phase_val_last_click_recall_recom_df, phase, False)
        val_target_uids = phase_val_last_click_answer_df['user_id'].unique()
        save_train_val_path = os.path.join(saving_training_path, 'train_val_label_target_id_dataa.pkl')
        pickle.dump([train_full_df, val_full_df, val_target_uids], open(save_train_val_path, 'wb'))
        return train_full_df, val_full_df, val_target_uids
    else:
        print('online')
        save_train_val_path = os.path.join(saving_training_path, 'train_val_label_target_id_dataa.pkl')
        pickle.dump(train_full_df, open(save_train_val_path, 'wb'))
        return train_full_df
    

def organize_final_train_data_feat(target_phase, train_full_df_dict, processed_item_feat, item_content_vec_dict, item_raw_id2_idx_dict, feat_lbe_dict, val_full_df_dict=None, is_train_load_from_file=True, save_df_prefix=''):
    if mode == 'online':
        online_train_full_df_dict = train_full_df_dict
    else:
        train_full_df_dict = train_full_df_dict
        val_full_df_dict = val_full_df_dict
    
    ranking_final_data = os.path.join(user_data_dir, 'ranking')
    if not os.path.exists(ranking_final_data):
        os.makedirs(ranking_final_data)
    
    train_df_path = os.path.join(ranking_final_data, save_df_prefix + f'train_final_df_phase_{target_phase}.pkl')
    val_df_path = os.path.join(ranking_final_data, save_df_prefix + f'val_final_df_phase_{target_phase}.pkl')
    w2v_path = os.path.join(ranking_final_data, save_df_prefix + f'w2v_phase_{target_phase}.pkl')

    if is_train_load_from_file and os.path.exists(train_df_path):
        print('load train final df from file...')
        train_final_df = pickle.load(open(train_df_path, 'rb'))
        word2vec_item_embed_dict = pickle.load(open(w2v_path, 'rb'))
        if mode == 'offline':
            val_final_df = pickle.load(open(val_df_path), 'rb')
            return train_final_df, val_final_df, word2vec_item_embed_dict
        return train_final_df, word2vec_item_embed_dict
    else:
        if mode == 'online':
            train_full_df = online_train_full_df_dict[target_phase]
            if isinstance(train_full_df, list):
                train_full_df = train_full_df[0]
        else:
            train_full_df = train_full_df_dict[target_phase]
            val_full_df = val_full_df_dict[target_phase]
        
        word2vec_item_embed_dict = get_word2vec_feat(train_full_df)
        train_final_df = organize_user_item_feat(train_full_df, processed_item_feat, sparse_feat, dense_feat, item_content_vec_dict, item_raw_id2_idx_dict, feat_lbe_dict, is_w2v=True, word2vec_item_embed_dict=word2vec_item_embed_dict)
        pickle.dump(train_final_df[use_feats + ['label']], open(train_df_path, 'wb'))
        pickle.dump(word2vec_item_embed_dict, open(w2v_path, 'wb'))
        if mode == 'offline':
            val_final_df = organize_user_item_feat(val_full_df, processed_item_feat, sparse_feat, dense_feat, item_content_vec_dict, item_raw_id2_idx_dict, feat_lbe_dict, is_w2v=True, word2vec_item_embed_dict=word2vec_item_embed_dict)

            pickle.dump(val_final_df[use_feats + ['label']], open(val_df_path, 'wb'))
            return train_final_df, val_final_df, word2vec_item_embed_dict

        return train_final_df, word2vec_item_embed_dict



def infer_process(phase, processed_item_feat, item_content_sim_dict, item_content_vec_dict, item_raw_id2_idx_dict, feat_lbe_dict, load_from_file=True, is_use_whole_click=False, is_w2v=True, is_interest=True, word2vec_item_embed_dict=None, prefix=''):
    all_click, target_infer_user_df = get_phase_click(phase)
    recall_methods = {'item_cf', 'bi-graph', 'user_cf', 'swing'}
    if is_use_whole_click:
        print('use whole click')
        phase_whole_click = get_whole_phase_click(all_click=all_click, click_q_time=target_infer_user_df)
        infer_user_item_time_dict = get_user_item_time_dict(phase_whole_click)
        phase_click = phase_whole_click
    else:
        infer_user_item_time_dict = get_user_item_time_dict(all_click)
        phase_click = all_click
    
    saving_training_path = os.path.join(user_data_dir, 'recall', mode)
    sim_path = os.path.join(saving_training_path, prefix + f'phase_{phase}_sim.pkl')
    recall_path = os.path.join(saving_training_path, prefix + f'phase_{phase}.pkl')

    if load_from_file:
        print('load infer recall data from file...')
        full_sim_pair_dict = pickle.load(open(sim_path, 'rb'))
        infer_user_recall_item_dict = pickle.load(open(recall_path, 'rb'))

    else:
        item_cnt_dict = all_click.groupby('item_id')['user_id'].count().to_dict()
        user_cnt_dict = all_click.groupby('user_id')['item_id'].count().to_dict()
        full_sim_pair_dict = get_multi_source_sim_dict_results(phase_click, recall_methods=recall_methods)
        infer_user_recall_item_dict = do_multi_recall_results(full_sim_pair_dict, infer_user_item_time_dict, item_content_sim_dict, target_user_ids=target_infer_user_df['user_id'].unique(),
                                                              phase=phase, item_cnt_dict=item_cnt_dict, user_cnt_dict=user_cnt_dict, recall_methods=recall_methods | {'TwoTower'})

        pickle.dump(full_sim_pair_dict, open(sim_path, 'wb'))
        pickle.dump(infer_user_recall_item_dict, open(recall_path, 'wb'))

    infer_recall_recom_df = organize_recall_feat(infer_user_recall_item_dict, infer_user_item_time_dict, full_sim_pair_dict, item_content_sim_dict, phase)
    target_infer_user_df['day_id'], target_infer_user_df['hour_id'], target_infer_user_df['minute_id'] = zip(*target_infer_user_df['time'].apply(time_info))
    infer_recall_recom_df = pd.merge(infer_recall_recom_df, target_infer_user_df[['user_id', 'time', 'day_id', 'hour_id', 'minute_id']],
                                     on='user_id', how='left')
    
    infer_final_df = organize_user_item_feat(infer_recall_recom_df, processed_item_feat, sparse_feat, dense_feat, item_content_vec_dict, item_raw_id2_idx_dict, feat_lbe_dict, is_interest=is_interest,
                                             is_w2v=is_w2v, word2vec_item_embed_dict=word2vec_item_embed_dict)
    
    return infer_recall_recom_df, infer_final_df


def organize_infer_data(target_phase, word2vec_item_embed_dict, processed_item_feat, item_content_sim_dict, item_content_vec_dict, item_raw_id2_idx_dict, feat_lbe_dict, save_df_prefix, recall_prefix, is_infer_load_from_file=True):
    ranking_final_data = os.path.join(user_data_dir, 'ranking')
    infer_df_path = os.path.join(ranking_final_data, save_df_prefix + recall_prefix + f'infer_final_df_phase_{target_phase}.pkl')

    if is_infer_load_from_file and os.path.exists(infer_df_path):
        print('load infer final df from file...')
        infer_recall_recom_df = pickle.load(open(infer_df_path, 'rb'))
    else:
        infer_recall_recom_df, infer_df = infer_process(target_phase, processed_item_feat, item_content_sim_dict, item_content_vec_dict,
                                                        item_raw_id2_idx_dict, feat_lbe_dict, load_from_file=True, is_use_whole_click=True,
                                                        prefix=recall_prefix, is_w2v=True, word2vec_item_embed_dict=word2vec_item_embed_dict)
    
    return infer_recall_recom_df, infer_df