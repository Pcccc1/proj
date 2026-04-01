import lightgbm as lgb
from config import *

def lgb_main(train_final_df, val_final_df=None):
    print('Training LightGBM model...')
    train_final_df.sort_values(by=['user_id'], inplace=True)
    g_train = train_final_df.groupby(['user_id'], as_index=False).count()['label'].values

    if mode == 'offline':
        val_final_df = val_final_df.sort_values(by=['user_id'])
        g_val = val_final_df.groupby(['user_id'], as_index=False).count()['label'].values

    lgb_ranker = lgb.LGBMRanker(
        boosting_type='gbdt',
        num_leaves=31,
        reg_alpha=0.0,
        reg_lambda=1,
        max_depth=-1,
        n_estimators=300,
        subsample=0.7,
        colsample_bytree=0.7,
        subsample_freq=1,
        learning_rate=0.01,
        min_child_weight=50,
        random_state=42,
        n_jobs=-1
    )

    if mode == 'offline':
        lgb_ranker.fit(
            train_final_df[lgb_cols],
            train_final_df['label'],
            group=g_train,
            eval_set=[val_final_df[lgb_cols], val_final_df['label']],
            eval_group=[g_val],
            eval_at=[50],
            eval_metric=['auc'],
            early_stopping_rounds=50,
        )
    else:
        lgb_ranker.fit(
            train_final_df[lgb_cols],
            train_final_df['label'],
            group=g_train
        )
    
    print('LightGBM model training completed.')
    return lgb_ranker