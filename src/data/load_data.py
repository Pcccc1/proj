from __future__ import annotations
from config import *
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import pandas as pd

CLICK_COLS = ['user_id', 'item_id', 'time']
QTIME_COLS = ['user_id', 'time']

DTYPES_CLICK = {'user_id': 'int32', 'item_id': 'int32', 'time': 'float64'}
DTYPES_QTIME = {'user_id': 'int32', 'time': 'float64'}

@dataclass(frozen=True)
class Paths:
    train_path : Path
    test_path : Path
    online_train_path: Path
    online_test_path: Path
    train_file_prefix : str
    test_file_prefix : str
    infer_file_prefix : str
    now_phase : int


def build_paths(run_mode: str = mode) -> Paths:
    is_online = run_mode == "online"
    train_path = Path(online_train_path) if is_online else Path(offline_train_path)
    test_path = Path(online_test_path) if is_online else Path(offline_test_path)
    return Paths(
        train_path=train_path,
        test_path=test_path,
        online_train_path=Path(online_train_path),
        online_test_path=Path(online_test_path),
        train_file_prefix=train_file_prefix,
        test_file_prefix=test_file_prefix,
        infer_file_prefix=infer_test_file_prefix,
        now_phase=now_phase,
    )


paths = build_paths()


def _read_click_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, header=None, names=CLICK_COLS, dtype=DTYPES_CLICK)


def _read_qtime_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, header=None, names=QTIME_COLS, dtype=DTYPES_QTIME)


"""
获取点击量最高的topk个item的id列表，以及以逗号分隔的字符串形式
"""
def obtain_topk_click(paths: Paths = paths, topk: int=50) -> tuple[pd.Index, str]:
    total_click = get_whole_click(paths)
    total_click = total_click.drop_duplicates(CLICK_COLS)

    tok_items = total_click['item_id'].value_counts().index[:topk]
    top_str = ','.join(map(str, tok_items.tolist()))
    return tok_items, top_str


"""
拿到某个phase的点击数据，包括train和test，以及qtime数据
"""
def get_phase_click(paths: Paths=paths, phase: int=now_phase) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_file = paths.train_path / f'{paths.train_file_prefix}-{phase}.csv'
    phase_test_dir = paths.test_path / f'{paths.test_file_prefix}-{phase}'
    test_file = phase_test_dir / f'{paths.test_file_prefix}-{phase}.csv'
    qtime_file = phase_test_dir / f'{paths.infer_file_prefix}-{phase}.csv'

    click_train = _read_click_csv(train_file)
    click_test = _read_click_csv(test_file)
    click_q_time = _read_qtime_csv(qtime_file)

    all_click = pd.concat([click_train, click_test], ignore_index=True)
    return all_click, click_q_time

def get_offline_evaluation_click(paths: Paths=paths, phase: int=now_phase) -> pd.DataFrame:
    file = Path(offline_answer_path) / f'{infer_answer_file_prefix}-{phase}.csv'
    click_answer = _read_click_csv(file)
    click_answer = click_answer.drop_duplicates(subset=['user_id'], keep='first')
    return click_answer


def get_online_whole_click(paths: Paths = paths) -> pd.DataFrame:
    whole_click = pd.DataFrame()
    frames = []
    for phase in range(paths.now_phase + 1):
        train_file = paths.online_train_path / f'{paths.train_file_prefix}-{phase}.csv'
        phase_test_dir = paths.online_test_path / f'{paths.test_file_prefix}-{phase}'
        test_file = phase_test_dir / f'{paths.test_file_prefix}-{phase}.csv'

        click_train = _read_click_csv(train_file)
        click_test = _read_click_csv(test_file)

        all_click = pd.concat([click_train, click_test], ignore_index=True)
        all_click['phase'] = phase
        frames.append(all_click)

    whole_click = pd.concat(frames, ignore_index=True)
    whole_click = whole_click.drop_duplicates(['user_id', 'item_id', 'time'])

    return whole_click

"""
拿到所有的点击数据，包括train和test
"""
@lru_cache(maxsize=2)
def get_whole_click(paths: Paths = paths) -> pd.DataFrame:
    frames = []
    for phase in range(paths.now_phase + 1):
        train_file = paths.train_path / f'{paths.train_file_prefix}-{phase}.csv'
        phase_test_dir = paths.test_path / f'{paths.test_file_prefix}-{phase}'
        test_file = phase_test_dir / f'{paths.test_file_prefix}-{phase}.csv'

        click_train = _read_click_csv(train_file)
        click_test = _read_click_csv(test_file)

        all_click = pd.concat([click_train, click_test], ignore_index=True)
        all_click['phase'] = phase
        frames.append(all_click)

    whole_click = pd.concat(frames, ignore_index=True)
    whole_click = whole_click.drop_duplicates(['user_id', 'item_id', 'time'])
    return whole_click


"""
根据某个phase的qtime，过滤掉在qtime之后的点击数据
如果filter_items_in_phase为True，则只保留在该phase中出现过的item
"""
def get_whole_phase_click(
    all_click: pd.DataFrame,
    click_q_time: pd.DataFrame,
    filter_items_in_phase: bool=True,
    paths: Paths = paths
) -> pd.DataFrame:
    whole_click = get_whole_click(paths)
    q_time = click_q_time.rename(columns={'time': 'q_time'})
    merged = whole_click.merge(q_time, on=['user_id'], how='left')

    mask = merged['q_time'].isna() | (merged['time'] <= merged['q_time'])
    phase_whole_click = merged.loc[mask, ['user_id', 'item_id', 'time', 'phase']]

    if filter_items_in_phase:
        phase_item_ids = set(all_click['item_id'].unique())
        phase_whole_click = phase_whole_click[phase_whole_click['item_id'].isin(phase_item_ids)]

    return phase_whole_click
