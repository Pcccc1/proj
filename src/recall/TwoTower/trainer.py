from __future__ import annotations
from dataclasses import asdict, dataclass, field, is_dataclass
import torch
from torch.utils.data import DataLoader
from typing import Iterable
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
from pathlib import Path
from src.recall.TwoTower.YoutubeDNN import YoutubeDNN
from src.recall.TwoTower.YoutubeDNNDataset import YouTubeDNNDataset, build_infer_tensors
from src.data.convert_data import get_user_item_time_dict
from src.data.feat_process import obtain_entire_item_feat_df


@dataclass
class YoutubeDNNConfig:
    max_seq_len: int = 30
    last_k: int = 8
    min_seq_len: int = 2

    batch_size: int = 1024
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-6
    num_workers: int = 4

    embedding_dim: int = 256
    user_hidden_dims: tuple[int, ...] = (512, 256)
    item_hidden_dims: tuple[int, ...] = (512, 256)
    output_dim: int = 128
    dropout: float = 0.1
    temperature: float = 0.05

    user_batch_size: int = 1024
    item_batch_size: int = 8192
    topk: int = 200
    exclude_history: bool = True

    seed: int = 42
    use_amp: bool = True
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    log_every: int = 200

@dataclass
class YoutubeDNNArtifact:
    model: YoutubeDNN
    item2idx: dict[int, int]
    idx2item: np.ndarray
    config: YoutubeDNNConfig
    train_losses: list[float] = field(default_factory=list)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _resolve_device(device: str) -> str:
    if device.startswith('cuda') and not torch.cuda.is_available():
        print('cuda is not available, using cpu instead.')
        return 'cpu'

    if device.startswith('mps'):
        mps_ok = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        if not mps_ok:
            print('mps is not available, using cpu instead.')
            return 'cpu'

    return device


def _build_item_vocab(user_item_time_dict: dict[int, list[tuple[int, float]]]) -> tuple[dict[int, int], np.ndarray]:
    all_items = []
    for item_time_list in user_item_time_dict.values():
        all_items.extend([int(item) for item, _ in item_time_list])
    uniq_items = sorted(set(all_items))
    item2idx = {item_id: idx+1 for idx, item_id in enumerate(uniq_items)}
    idx2item = np.zeros(len(uniq_items) + 1, dtype=np.int64)
    if uniq_items:
        idx2item[1:] = np.asarray(uniq_items, dtype=np.int64)
    return item2idx, idx2item


def train(
        user_item_time_dict: dict[int, list[tuple[int, float]]],
        config: YoutubeDNNConfig | None = None,
) -> tuple[YoutubeDNNArtifact, dict]:
    config = config or YoutubeDNNConfig()
    config.device = _resolve_device(config.device)
    _set_seed(config.seed)

    item2idx, idx2item = _build_item_vocab(user_item_time_dict)
    if len(item2idx) == 0:
        raise ValueError('No items found in the data.')
    
    dataset = YouTubeDNNDataset(
        user_item_time_dict=user_item_time_dict,
        item2idx=item2idx,
        max_seq_len=config.max_seq_len,
        last_k=config.last_k,
        min_seq_len=config.min_seq_len,
        pad_idx=0
    )
    if len(dataset) == 0:
        raise ValueError('No valid training samples found. Please check the data and config parameters.')
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.device.startswith('cuda'),
        drop_last=False
    )

    model = YoutubeDNN(
        num_items=len(idx2item),
        embedding_dim=config.embedding_dim,
        user_hidden_dims=config.user_hidden_dims,
        item_hidden_dims=config.item_hidden_dims,
        output_dim=config.output_dim,
        dropout=config.dropout,
        temperature=config.temperature,
        pad_idx=0
    ).to(config.device)

    _, item_content_vec_dict = obtain_entire_item_feat_df()

    model.init_item_embedding_from_content(
        item2idx=item2idx,
        item_content_dict=item_content_vec_dict,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp and config.device.startswith('cuda'))

    train_losses = []
    model.train()
    for epoch in range(1, config.epochs + 1):
        total_loss = 0.0
        total_cnt = 0

        for step, (hist_ids, hist_mask, pos_ids) in enumerate(dataloader, start=1):
            hist_ids = hist_ids.to(config.device, non_blocking=True)
            hist_mask = hist_mask.to(config.device, non_blocking=True)
            pos_ids = pos_ids.to(config.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=config.use_amp and config.device.startswith('cuda')):
                logits, labels = model(hist_ids, hist_mask, pos_ids)
                loss = F.cross_entropy(logits, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = hist_ids.size(0)
            total_loss += float(loss.item()) * bs
            total_cnt += bs

            if config.log_every > 0 and step % config.log_every == 0:
                print(f'epoch={epoch}/{config.epochs} step={step}/{len(dataloader)} loss={loss.item():.6f}')
    
        epoch_loss = total_loss / max(total_cnt, 1)
        train_losses.append(epoch_loss)
        print(f'epoch={epoch}/{config.epochs} avg_loss={epoch_loss:.6f}')

    model.eval()
    artifact = YoutubeDNNArtifact(
        model=model,
        item2idx=item2idx,
        idx2item=idx2item,
        config=config,
        train_losses=train_losses
    )
    info = {
        "num_users": int(len(user_item_time_dict)),
        "num_items": int(len(item2idx)),
        "num_train_samples": int(len(dataset)),
        "train_losses": train_losses
    }
    return artifact, info


@torch.no_grad()
def _encode_all_items(artifact: YoutubeDNNArtifact) -> tuple[torch.Tensor, torch.Tensor]:
    model = artifact.model
    config = artifact.config
    num_items = len(artifact.idx2item) - 1
    if num_items <= 0:
        raise ValueError('No items to encode.')
    
    all_idx = torch.arange(1, num_items + 1, device=config.device)
    item_vec_list = []
    for st in range(0, num_items, config.item_batch_size):
        ed = min(st + config.item_batch_size, num_items)
        batch_idx = all_idx[st:ed]
        item_vec = model.encode_item(batch_idx)
        item_vec_list.append(item_vec)
    
    all_item_vec = torch.cat(item_vec_list, dim=0)
    return all_idx, all_item_vec


@torch.no_grad()
def recall_topk(
    artifact: YoutubeDNNArtifact,
    user_item_time_dict: dict[int, list[tuple[int, float]]],
    target_user_ids: Iterable[int],
    topk: int | None = None,
    exclude_history: bool | None = None,
) -> pd.DataFrame:
    config = artifact.config
    model = artifact.model
    item2idx = artifact.item2idx
    idx2item = artifact.idx2item

    topk = int(topk if topk is not None else config.topk)
    exclude_history = bool(exclude_history if exclude_history is not None else config.exclude_history)

    user_ids, hist_ids, hist_mask, seen_items = build_infer_tensors(
        user_item_time_dict=user_item_time_dict,
        target_user_ids=target_user_ids,
        item2idx=item2idx,
        max_seq_len=config.max_seq_len,
        pad_idx=0
    )
    if not user_ids:
        return pd.DataFrame(columns=['user_id', 'item_id', 'sim'])
    
    hist_ids = hist_ids.to(config.device, non_blocking=True)
    hist_mask = hist_mask.to(config.device, non_blocking=True)

    item_idx_tensor, all_item_vec = _encode_all_items(artifact)
    n_items = all_item_vec.size(0)
    recall_topk = min(topk, n_items)

    row = []
    for st in range(0, len(user_ids), config.user_batch_size):
        ed = min(st + config.user_batch_size, len(user_ids))
        batch_uids = user_ids[st:ed]
        batch_hist = hist_ids[st:ed]
        batch_mask = hist_mask[st:ed]

        user_vec = model.encode_user(batch_hist, batch_mask) # (B, D)
        score_mat = torch.matmul(user_vec, all_item_vec.transpose(0, 1)) # (B, n_items)

        if exclude_history:
            for row_idx, uid in enumerate(batch_uids):
                seen_raw_items = seen_items.get(uid, set())
                if not seen_raw_items:
                    continue
                seen_idx = [item2idx[i] for i in seen_raw_items if i in item2idx]
                if not seen_idx:
                    continue
                col_idx = torch.as_tensor(np.asarray(seen_idx, dtype=np.int64) - 1, device=config.device)
                score_mat[row_idx, col_idx] = -torch.inf
        
        top_scores, top_cols = torch.topk(score_mat, k=recall_topk, dim=1)
        top_item_idx = item_idx_tensor[top_cols] # (B, topk)

        top_scores_np = top_scores.cpu().numpy()
        top_item_idx_np = top_item_idx.cpu().numpy()

        for row_idx, uid in enumerate(batch_uids):
            raw_items = idx2item[top_item_idx_np[row_idx]].tolist()
            sims = top_scores_np[row_idx].tolist()
            for item, sim in zip(raw_items, sims):
                if exclude_history and not np.isfinite(sim):
                    continue
                row.append((int(uid), int(item), float(sim)))
        

    recall_df = pd.DataFrame(row, columns=['user_id', 'item_id', 'sim'])
    return recall_df


def run_phase_youtube_dnn(
    phase_click: pd.DataFrame,
    target_user_ids: Iterable[int],
    config: YoutubeDNNConfig | None = None
) -> tuple[pd.DataFrame, YoutubeDNNArtifact, dict]:
    
    user_item_time_dict = get_user_item_time_dict(phase_click)
    artifact, train_info = train(user_item_time_dict=user_item_time_dict, config=config)
    recall_df = recall_topk(
        artifact=artifact,
        user_item_time_dict=user_item_time_dict,
        target_user_ids=target_user_ids
    )
    info = {**train_info, "num_recall_rows":int(len(recall_df))}
    return recall_df, artifact, info


def save_artifact(artifact: YoutubeDNNArtifact, file_path: str | Path) -> None:
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'model_state_dict' : artifact.model.state_dict(),
        'item2idx': artifact.item2idx,
        'idx2item' : artifact.idx2item,
        'config' : artifact.config,
        'train_losses' : artifact.train_losses
    }
    torch.save(payload, file_path)


def load_artifact(file_path: str | Path, map_location: str | None = None) -> YoutubeDNNArtifact:
    file_path = Path(file_path)
    payload = torch.load(file_path, map_location=map_location or 'cpu')

    cfg = payload['config']
    if isinstance(cfg, YoutubeDNNConfig):
        config = cfg
    else:
        if is_dataclass(cfg):
            cfg = asdict(cfg)
        if not isinstance(cfg, dict):
            raise TypeError(f'Unsupported config payload type: {type(cfg)}')
        cfg = dict(cfg)
        if 'user_hidden_dims' in cfg:
            cfg['user_hidden_dims'] = tuple(cfg['user_hidden_dims'])
        if 'item_hidden_dims' in cfg:
            cfg['item_hidden_dims'] = tuple(cfg['item_hidden_dims'])
        config = YoutubeDNNConfig(**cfg)

    if map_location is not None:
        config.device = map_location
    config.device = _resolve_device(config.device)
    
    idx2item = payload['idx2item']
    item2idx = payload['item2idx']

    model = YoutubeDNN(
        num_items=len(idx2item),
        embedding_dim=config.embedding_dim,
        user_hidden_dims=config.user_hidden_dims,
        item_hidden_dims=config.item_hidden_dims,
        output_dim=config.output_dim,
        dropout=config.dropout,
        temperature=config.temperature,
        pad_idx=0
    )
    model.load_state_dict(payload['model_state_dict'])
    model.to(config.device)
    model.eval()

    return YoutubeDNNArtifact(
        model=model,
        item2idx=item2idx,
        idx2item=idx2item,
        config=config,
        train_losses=payload.get('train_losses', [])
    )
