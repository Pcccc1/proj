import os
import time

data_dir = './data'
user_data_dir = './data/user_data'

online_train_path = os.path.join(data_dir, 'underexpose_train')
online_test_path = os.path.join(data_dir, 'underexpose_test')

offline_train_path = os.path.join(data_dir, 'offline_underexpose_train')
offline_test_path = os.path.join(data_dir, 'offline_underexpose_test')
offline_answer_path = os.path.join(data_dir, 'offline_underexpose_answer')

item_feat_file_path = os.path.join(online_train_path, 'underexpose_item_feat.csv')
user_feat_file_path = os.path.join(online_train_path, 'underexpose_user_feat.csv')


train_file_prefix = "underexpose_train_click"
test_file_prefix = "underexpose_test_click"
infer_test_file_prefix = "underexpose_test_qtime"
infer_answer_file_prefix = "underexpose_test_qtime_with_answer"


txt_dense_feat = ['txt_embed_' + str(i) for i in range(128)]
img_dense_feat = ['img_embed_' + str(i) for i in range(128)]
item_dense_feat = txt_dense_feat + img_dense_feat


is_multi_processing = False
now_phase = 9
start_phase = 7
mode = 'offline'  # online or offline