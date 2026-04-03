from src.data.feat_process import read_item_feat_df, obtain_entire_item_feat_df
from src.recall.content_sim_item import get_content_sim_item
from config import *
from src.rank.construst_ranking_data import sliding_obtain_training_df, organize_train_data, organize_final_train_data_feat, organize_infer_data
from src.rank.construct_feature_info_feat import sparse_feat_fit
from src.data.load_data import get_online_whole_click, obtain_topk_click
from src.data.convert_data import sub2_df
from src.rank.ensemble import ensemble
from utils.recommend import get_predict
from src.rank.lgb import lgb_main
from src.rank.din import din_main, build_din_input, BATCH_SIZE
import pandas as pd
import pickle


def rank_pipline(target_phase, train_full_df_dict, processed_item_feat, item_content_sim_dict, item_content_vec_dict, item_raw_id2_idx_dict, user_raw_id2_idx_dict, feat_lbe_dict, val_full_df_dict=None, output_ranking_filename=None, model_names=['ranker'], 
                 is_train_load_from_file=True, is_infer_load_from_file=True, recall_prefix='', save_df_prefix=''):
    global total_recom_lgb_df
    if mode == 'offline':
        train_final_df, val_final_df, word2vec_item_embed_dict = organize_final_train_data_feat(target_phase, train_full_df_dict, processed_item_feat, item_content_vec_dict,
                                                                      item_raw_id2_idx_dict, feat_lbe_dict, val_full_df_dict, is_train_load_from_file, save_df_prefix)
    else:
        train_final_df, word2vec_item_embed_dict = organize_final_train_data_feat(target_phase, train_full_df_dict, processed_item_feat, item_content_vec_dict,
                                                                      item_raw_id2_idx_dict, feat_lbe_dict, is_train_load_from_file=is_train_load_from_file, save_df_prefix=save_df_prefix)
    print('prepare rank training data done')

    infer_recall_recom_df, infer_df = organize_infer_data(target_phase, word2vec_item_embed_dict, processed_item_feat, item_content_sim_dict, item_content_vec_dict,
                                                          item_raw_id2_idx_dict, feat_lbe_dict, save_df_prefix, recall_prefix, is_infer_load_from_file)

    print('prepare rank infer data done')

    def gen_rec_results(output_model_name):
        global total_recom_lgb_df
        if 'user_id' in total_recom_lgb_df.columns:
            total_recom_lgb_df['user_id'] = total_recom_lgb_df['user_id'].astype(int)
        if 'item_id' in total_recom_lgb_df.columns:
            total_recom_lgb_df['item_id'] = total_recom_lgb_df['item_id'].astype(int)
        if 'sim' in total_recom_lgb_df.columns:
            total_recom_lgb_df['sim'] = total_recom_lgb_df['sim'].astype(float)

        online_infer_recall_df = infer_recall_recom_df[['user_id', 'item_id', 'prob']].rename(columns={'prob': 'sim'})
        online_infer_recall_df['phase'] = target_phase
        online_infer_recall_df['user_id'] = online_infer_recall_df['user_id'].astype(int)

        if mode == 'online':
            if 'phase' not in total_recom_lgb_df.columns:
                total_recom_lgb_df['phase'] = -1
            total_recom_lgb_df = total_recom_lgb_df[total_recom_lgb_df['phase'] != target_phase]
        else:
            # Offline user ids may appear in multiple phases; do replacement by target users instead of uid % 11 phase.
            target_users = set(online_infer_recall_df['user_id'].unique())
            total_recom_lgb_df = total_recom_lgb_df[~total_recom_lgb_df['user_id'].astype(int).isin(target_users)]

        total_recom_lgb_df = pd.concat([total_recom_lgb_df, online_infer_recall_df], axis=0, ignore_index=True)

        _, top50_click = obtain_topk_click()
        result = get_predict(total_recom_lgb_df, 'sim', top50_click)

        rank_output_dir = os.path.join(user_data_dir, 'rank')
        if not os.path.exists(rank_output_dir):
            os.makedirs(rank_output_dir)
        result.to_csv(f'{rank_output_dir}/{output_model_name}-{output_ranking_filename}', index=False, header=None)
        pickle.dump(total_recom_lgb_df, open(f'{rank_output_dir}/{output_model_name}-{output_ranking_filename}-pkl', 'wb'))

        if mode == 'offline':
            # Also save phase-local ranking file for offline evaluation/debugging.
            phase_result = get_predict(online_infer_recall_df[['user_id', 'item_id', 'sim']], 'sim', top50_click)
            phase_result.to_csv(f'{rank_output_dir}/{output_model_name}-phase_{target_phase}-{output_ranking_filename}', index=False, header=None)
        print('generate rank result done')

    if 'ranker' in model_names:
        lgb_ranker = lgb_main(train_final_df, val_final_df)
        lgb_rank_infer_ans = lgb_ranker.predict(infer_df[lgb_cols])
        infer_recall_recom_df['prob'] = lgb_rank_infer_ans
        gen_rec_results('ranker')

    if 'din' in model_names:
        din_model, feature_names = din_main(target_phase, train_final_df, item_raw_id2_idx_dict, user_raw_id2_idx_dict, item_content_vec_dict, feat_lbe_dict, val_final_df)
        infer_input = build_din_input(infer_df, feature_names)
        din_infer_ans = din_model.predict(infer_input, batch_size=BATCH_SIZE)
        infer_recall_recom_df['prob'] = din_infer_ans
        gen_rec_results('din')



if __name__ == '__main__':
    item_feat_df = read_item_feat_df()
    item_content_sim_dict = get_content_sim_item(item_feat_df, topk=200)
    print(len(item_content_sim_dict))

# 生成训练数据
    for i in range(start_phase, now_phase + 1):
        sliding_obtain_training_df(i, item_content_sim_dict)

# 特征工程
    processed_item_feat_df, item_content_vec_dict = obtain_entire_item_feat_df()

# 编码sparse特征
    online_total_click = get_online_whole_click()
    feat_lbe_dict, item_raw_id2_idx_dict, user_raw_id2_idx_dict = sparse_feat_fit(online_total_click)


# 构造除了feature之外的训练数据
    if mode == 'online':
        train_full_df_dict = {}
        for i in range(start_phase, now_phase + 1):
            if i in train_full_df_dict:
                continue
            train_full_df = organize_train_data(i, item_content_vec_dict, item_content_sim_dict, is_sliding_compute_sim=False, load_from_file=True)
            train_full_df_dict[i] = train_full_df
    else:
        train_full_df_dict = {}
        val_full_df_dict = {}
        for i in range(start_phase, now_phase + 1):
            train_full_df, val_full_df, val_target_uids = organize_train_data(i, item_content_sim_dict, is_sliding_compute_sim=False, load_from_file=True)
            train_full_df_dict[i] = train_full_df
            val_full_df_dict[i] = val_full_df
    
# 读取recall结果
    global total_recom_lgb_df
    total_recom_lgb_df = sub2_df(os.path.join('submit.csv'))

# rank pipeline
    today = time.strftime("%Y%m%d")
    output_ranking_filename = f'ranking-{today}'
    for i in range(start_phase, now_phase + 1):
        print(f"begin phase {i} ranking...")
        output_ranking_filename = output_ranking_filename + "_" + str(i)
        rank_pipline(i, train_full_df_dict, processed_item_feat_df, item_content_sim_dict, item_content_vec_dict, 
                     item_raw_id2_idx_dict, user_raw_id2_idx_dict, feat_lbe_dict, val_full_df_dict, output_ranking_filename=output_ranking_filename + '.csv', 
                     model_names=['ranker', 'din'], is_train_load_from_file=True, is_infer_load_from_file=True, 
                     recall_prefix=f'recall-{today}_',save_df_prefix=f'{today}_')
                    

    ensemble(output_ranking_filename)
