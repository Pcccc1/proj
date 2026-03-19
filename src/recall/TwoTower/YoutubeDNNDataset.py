from __future__ import annotations
from typing import Iterable
import torch
from torch.utils.data import Dataset
import numpy as np

def _left_pad_sequence(seq: list[int], max_len: int, pad_idx: int) -> tuple[list[int], list[int]]:
    seq = seq[-max_len:]
    real_len = len(seq)
    if real_len < max_len:
        pad_len = max_len - real_len
        return [pad_idx] * pad_len + seq, [0] * pad_len + [1] * real_len
    return seq, [1] * max_len

class YouTubeDNNDataset(Dataset):
    def __init__(
            self,
            user_item_time_dict: dict[int, list[tuple[int, float]]],
            item2idx: dict[int, int],
            max_seq_len: int = 30,
            last_k: int = 8,
            min_seq_len: int = 2,
            pad_idx: int = 0,
    ) -> None:
        self.max_seq_len = max_seq_len
        self.pad_idx = pad_idx
        self.samples : list[tuple[list[int], list[int], int]] = []

        for _, item_time_list in user_item_time_dict.items():
            if len(item_time_list) < min_seq_len:
                continue

            item_time_list = sorted(item_time_list, key=lambda x: x[1])
            item_seq = [item for item, _ in item_time_list if item in item2idx]

            if len(item_seq) < min_seq_len:
                continue

            start = 1 if last_k <= 0 else max(1, len(item_seq) - last_k)
            for t in range(start, len(item_seq)):
                pos_item = item_seq[t]
                hist_raw = item_seq[max(0, t-max_seq_len) : t]
                hist_idx = [item2idx[i] for i in hist_raw]
                if not hist_idx:
                    continue
                hist_ids, hist_mask = _left_pad_sequence(hist_idx, max_seq_len, pad_idx)
                self.samples.append((hist_ids, hist_mask, item2idx[pos_item]))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hist_ids, hist_mask, pos_item = self.samples[idx]
        return (
            torch.tensor(hist_ids, dtype=torch.long),
            torch.tensor(hist_mask, dtype=torch.float32),
            torch.tensor(pos_item, dtype=torch.long)
        )
    
@torch.no_grad()
def build_infer_tensors(
        user_item_time_dict: dict[int, list[tuple[int, float]]],
        target_user_ids: Iterable[int],
        item2idx: dict[int, int],
        max_seq_len: int = 30,
        pad_idx: int = 0,
) -> tuple[list[int], torch.Tensor, torch.Tensor, dict[int, set[int]]]:
    
    user_ids = []
    hist_ids_list = []
    hist_mask_list = []
    seen_item_dict = {}

    for user_id in target_user_ids:
        item_time_list = user_item_time_dict.get(user_id, [])
        if not item_time_list:
            continue
        item_time_list = sorted(item_time_list, key=lambda x: x[1])
        raw_items = [item for item, _ in item_time_list]
        mapped_items = [item2idx[item] for item in raw_items if item in item2idx]
        if not mapped_items:
            continue

        hist_ids, hist_mask = _left_pad_sequence(mapped_items, max_seq_len, pad_idx)
        user_ids.append(int(user_id))
        hist_ids_list.append(hist_ids)
        hist_mask_list.append(hist_mask)
        seen_item_dict[int(user_id)] = set(raw_items)

    if not user_ids:
        empty_hist = torch.empty((0, max_seq_len), dtype=torch.long)
        empty_mask = torch.empty((0, max_seq_len), dtype=torch.float32)
        return user_ids, empty_hist, empty_mask, seen_item_dict
    return (
        user_ids,
        torch.tensor(hist_ids_list, dtype=torch.long),
        torch.tensor(hist_mask_list, dtype=torch.float32),
        seen_item_dict,
    )