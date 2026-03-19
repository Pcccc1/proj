from src.data.feat_process import read_item_feat_df
from src.recall.content_sim_item import get_content_sim_item
from src.data.load_data import obtain_topk_click, get_phase_click, get_whole_phase_click
from src.data.convert_data import get_user_item_time_dict
from src.data.save_data import save_recall_df_as_user_tuples_dict
from src.recall.recall import get_multi_source_sim_dict_results, do_multi_recall_results, get_predict
from config import *
import pandas as pd

if __name__ == '__main__':
    item_feat_df = read_item_feat_df()
    item_content_sim_dict = get_content_sim_item(item_feat_df, topk=200)
    print(len(item_content_sim_dict))
    top50_click_np, top50_click = obtain_topk_click()

    total_recom_df = pd.DataFrame()
    phase_full_sim_dict = {}

    cf_methods = {'item_cf', 'bi-graph', 'swing', 'user_cf'}

    if is_multi_processing:
        raise NotImplementedError('`is_multi_processing=True` is not implemented yet, please set it to False.')
    else:
        print("recall with single process...")
        do_cf_sim_fun = get_multi_source_sim_dict_results
        do_recall_fun = do_multi_recall_results

    for c in range(start_phase, now_phase + 1):
        print(f"begin phase {c} recall...")
        all_click, click_q_time = get_phase_click(phase=c)
        if mode == "online":
            phase_click = get_whole_phase_click(all_click=all_click, click_q_time=click_q_time)
        else:
            # Keep offline evaluation phase-local to match original baseline behavior.
            phase_click = all_click
        item_cnt_dict = all_click.groupby('item_id')['user_id'].count().to_dict()
        user_cnt_dict = all_click.groupby('user_id')['item_id'].count().to_dict()

        recall_sim_pair_dict = do_cf_sim_fun(phase_click, recall_methods=cf_methods)
        user_item_time_dict = get_user_item_time_dict(phase_click, drop_duplicates=True)
        recall_df = do_recall_fun(recall_sim_pair_dict, user_item_time_dict, item_content_sim_dict,
                                  target_user_ids=click_q_time['user_id'].unique(), item_cnt_dict=item_cnt_dict, user_cnt_dict=user_cnt_dict,
                                  phase=c, recall_methods=cf_methods)
        recall_df['phase'] = c
        total_recom_df = pd.concat([total_recom_df, recall_df], axis=0)
        phase_full_sim_dict[c] = recall_sim_pair_dict

    today = time.strftime("%Y%m%d")
    save_recall_df_as_user_tuples_dict(total_recom_df, phase_full_sim_dict, prefix=f'recall-{today}')
    result = get_predict(total_recom_df, 'sim', top50_click)
    result.to_csv('submit.csv', index=False, header=None)
